"""Build the fixed content pool in one run from the publishers in pool_sources.json.

RSS lists only the newest items, so instead of polling for days each publisher is backfilled
over a window: the WordPress posts API (title, date and full text in one call) where the site
offers it, else its sitemaps (URLs with dates), plus its RSS feed. General publishers keep
WINDOW_DAYS, hobby publishers HOBBY_WINDOW_DAYS since they publish little. Each publisher is
capped and its articles are taken evenly across the days. Then one SearXNG search per topic
(MIND categories, MovieLens genres, Amazon categories, common hobbies; never a persona's own
queries, so the pool stays shared) adds long-tail articles from any site.
Every fetched page must be allowed by robots.txt, each site is fetched at most once per
FETCH_DELAY seconds, and articles pass the same is_article check as the news.

Output (article text is copyrighted: local research use only, never commit or share):
  <out>/items.jsonl       one article per line
  <out>/embeddings.npy    row i = line i of items.jsonl (title + opening words)
  <out>/manifest.json     settings and counts
Rerunning the same --out resumes: URLs in items.jsonl / rejected.jsonl are not fetched again.

Example:
  python -m src.eval.build_content_pool --domains fieldandstream.com,variety.com --skip-search
  python -m src.eval.build_content_pool
"""
import argparse
import gzip
import hashlib
import html
import json
import math
import re
import threading
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import feedparser
import numpy as np
import requests
import trafilatura
from bs4 import BeautifulSoup

from src.eval.mind_personas import READABLE
from src.user_profile.fetch_breaking_news import CONSENT_NOTICE, HEADERS, clean, is_article, normalize_url, search
from src.user_profile.fetch_site_articles import LISTING_PATH, MAX_EMPTY_SEARCHES, SEARCH_DELAY
from src.utils import PROJECT_ROOT

SOURCES_FILE = Path(__file__).with_name("pool_sources.json")
OUT_DIR = PROJECT_ROOT / "data" / "eval" / "content_pool"
WINDOW_DAYS = 7
HOBBY_WINDOW_DAYS = 30
CAPS = {"general_news": 300, "existing": 120, "hobby": 150}
# Fetch attempts per publisher before giving up on its cap (failed pages, videos, old articles).
ATTEMPTS_PER_KEPT = 2
TOPIC_RESULTS = 8
FETCH_DELAY = 1.0
WORKERS = 10
MAX_SITEMAPS = 25
EMBED_WORDS = 200
SKIP_PATH = re.compile(r"/(video|videos|gallery|galleries|photos|slideshow|live|podcasts?|audio|newsletters?)/", re.I)
SKIP_SITEMAP = re.compile(r"video|image|tag|categor|author|topic|page-sitemap|product|store|shop|gallery", re.I)
URL_MONTH = re.compile(r"(20\d\d)[-/_]?(0[1-9]|1[0-2])(?!\d)")
MONTHS = "jan feb mar apr may jun jul aug sep oct nov dec".split()
URL_DATE = re.compile(r"/(20\d\d)[/-](0[1-9]|1[0-2]|" + "|".join(MONTHS) + r")[/-]([0-3]\d)(?:/|-|$)", re.I)
# Hub, landing and product pages also sit in sitemaps; og:type tells them apart when present.
ARTICLE_TYPES = re.compile(r"article|news|blog|post", re.I)
# Sportsbook promo codes and shopping deals are ads, not podcast material. Plain "deal" and "sale"
# are left alone: trade deals, contract deals and auction sales are news.
PROMO_TITLE = re.compile(
    r"promo code|bonus code|sportsbook promos?|betting promos?|bonus bets|betting apps|"
    r"black friday|cyber (monday|sale)|prime day|deals? of the day|\bdeals\b(?! with)|\bdeal (on|from|under)\b|"
    r"(?<![\d.])\d+\s*(%|percent) off|\$[\d,.]+ off\b|lowest prices?|price drop|best price|discount code|"
    r"exclusive discount|limited-time discount|(?<!goes )\bon[ -]sale\b|['’]s sale on\b|"
    r"(flash|warehouse|overstock|cyber) sale|sale section|savings event|"
    r"for (just|only|less than) \$|drops? to (a mere |as low as )?\$|starting at (just )?\$", re.I)
TRADE = re.compile(r"\btrade\b", re.I)
# Topics come from the datasets' category vocabularies and common hobbies, not from any persona.
MIND_SKIP = {"News photos", "Movie clips", "TV clips", "Viral videos"}
MOVIE_GENRES = ["Action", "Adventure", "Animated", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
                "Film noir", "Horror", "Musical", "Mystery", "Romantic", "Science fiction", "Thriller", "War", "Western"]
AMAZON_CATEGORIES = [
    "beauty products", "fashion and clothing", "home appliances", "arts, crafts and sewing supplies",
    "car accessories", "baby products", "books", "vinyl records", "smartphones", "consumer electronics",
    "gourmet food", "handmade gifts", "household products", "kitchen gadgets", "e-books", "movies on Blu-ray",
    "musical instruments", "office supplies", "lawn and garden equipment", "pet supplies", "software",
    "sports and outdoor gear", "power tools", "toys and games", "video games"]
HOBBIES = [
    "gardening", "fishing", "fly fishing", "hunting", "woodworking", "hiking", "camping", "chess", "photography",
    "birdwatching", "crossword puzzles", "sudoku", "yoga", "knitting", "quilting", "sewing", "crochet",
    "scrapbooking", "classic car restoration", "playing guitar", "piano", "genealogy", "model trains",
    "model building", "running", "cycling", "board games", "golf", "bowling", "tennis", "pickleball", "swimming",
    "kayaking", "skiing", "home brewing", "baking", "grilling and barbecue", "wine tasting", "painting",
    "watercolor painting", "drawing", "pottery", "jewelry making", "calligraphy", "creative writing",
    "book clubs", "poetry", "theater", "dancing", "salsa dancing", "line dancing", "karaoke", "volunteering",
    "meditation", "journaling", "astronomy", "coin collecting", "stamp collecting", "antiques", "thrift shopping",
    "motorcycle riding", "boating", "horseback riding", "dog training", "aquariums", "beekeeping",
    "home renovation", "interior decorating", "podcasts", "video gaming", "anime", "comic books", "magic tricks",
    "softball", "basketball", "soccer", "volleyball", "martial arts", "weightlifting", "running marathons"]


# One "<genre> movies" search left genres such as romance and film noir with a handful of
# articles, so the movie topics also search reviews, new releases, interviews and streaming picks.
GENRE_WORDS = {"Children's": "family", "Romantic": "romance", "Science fiction": "sci-fi", "War": "war",
               "Animated": "animated"}
MOVIE_TEMPLATES = ["{} movie review", "new {} movies", "{} film director interview", "best {} movies to stream"]


def topics():
    mind = sorted(set(READABLE.values()) - MIND_SKIP)
    return (mind + [f"{g} movies" for g in MOVIE_GENRES] + [f"{c} reviews" for c in AMAZON_CATEGORIES] +
            [h.capitalize() for h in HOBBIES] + movie_topics())


def movie_topics():
    return [template.format(GENRE_WORDS.get(g, g.lower())) for g in MOVIE_GENRES for template in MOVIE_TEMPLATES]


def promo(title):
    match = PROMO_TITLE.search(title or "")
    return bool(match) and not (match.group(0).lower() == "deals" and TRADE.search(title))


def utc(value):
    """A timezone-aware UTC datetime from feed tuples, ISO strings or None."""
    if not value:
        return None
    if isinstance(value, time.struct_time):
        return datetime(*value[:6], tzinfo=timezone.utc)
    try:
        moment = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def url_date(url):
    """The date written in an article URL (/2025/nov/08/, /2025/11/08/), if any."""
    match = URL_DATE.search(urlsplit(url).path)
    if not match:
        return None
    month = match.group(2).lower()
    month = MONTHS.index(month) + 1 if month in MONTHS else int(month)
    try:
        return datetime(int(match.group(1)), month, int(match.group(3)), tzinfo=timezone.utc)
    except ValueError:
        return None


def host(url):
    return urlsplit(url).netloc.lower().removeprefix("www.")


class Fetcher:
    """Polite HTTP: robots.txt checked, at most one request per FETCH_DELAY seconds per site."""

    def __init__(self):
        self.local = threading.local()
        self.locks = defaultdict(threading.Lock)
        self.last = {}
        self.robots = {}
        self.robots_lock = threading.Lock()

    def session(self):
        if not hasattr(self.local, "session"):
            self.local.session = requests.Session()
            self.local.session.headers.update(HEADERS)
        return self.local.session

    def get(self, url, **params):
        site = host(url)
        with self.locks[site]:
            wait = self.last.get(site, 0) + FETCH_DELAY - time.time()
            if wait > 0:
                time.sleep(wait)
            try:
                response = self.session().get(url, params=params or None, timeout=25)
            except requests.RequestException:
                response = None
            finally:
                self.last[site] = time.time()
        return response if response is not None and response.ok else None

    def allowed(self, url):
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        with self.robots_lock:
            parser = self.robots.get(origin)
        if parser is None:
            parser = RobotFileParser()
            response = self.get(origin + "/robots.txt")
            parser.parse(response.text.splitlines() if response is not None else [])
            with self.robots_lock:
                self.robots[origin] = parser
        return parser.can_fetch("*", url)


def rendered(value):
    """Plain text of a WordPress field: {"rendered": html} on most sites, a bare string on some."""
    markup = value.get("rendered", "") if isinstance(value, dict) else str(value or "")
    return BeautifulSoup(markup, "html.parser").get_text("\n")


def wordpress_posts(fetcher, base, since, until, limit):
    """Posts with full text from the WordPress REST API, an equal share asked for each day of the
    window (newest posts would otherwise fill the cap); [] when the site has no such API."""
    days = max(1, math.ceil((until - since).total_seconds() / 86400))
    per_day = min(100, math.ceil(limit / days))
    items = []
    for day in range(days):
        end = until - timedelta(days=day)
        response = fetcher.get(f"{base}/wp-json/wp/v2/posts", per_page=per_day,
                               after=(end - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S"),
                               before=end.strftime("%Y-%m-%dT%H:%M:%S"), _fields="link,date_gmt,title,content")
        try:
            posts = response.json() if response is not None else None
        except ValueError:
            posts = None
        if not isinstance(posts, list):
            break  # no WordPress API
        for post in posts:
            if not isinstance(post, dict):
                continue
            body, title = rendered(post.get("content")), " ".join(rendered(post.get("title")).split())
            items.append({"url": post.get("link", ""), "title": clean(html.unescape(title)),
                          "published": utc(post.get("date_gmt")), "text": "\n".join(
                              line.strip() for line in body.splitlines() if line.strip()), "via": "wp_json"})
    return items


def read_xml(fetcher, url):
    response = fetcher.get(url)
    if response is None:
        return None
    content = response.content
    if content[:2] == b"\x1f\x8b":
        content = gzip.decompress(content)
    try:
        return ET.fromstring(content)
    except ET.ParseError:
        return None


def sitemap_priority(loc, lastmod, since):
    """News sitemaps first, then ones dated (or named) inside the window, newest first."""
    named = URL_MONTH.search(loc)
    month = datetime(int(named.group(1)), int(named.group(2)), 1, tzinfo=timezone.utc) if named else None
    date = lastmod or month
    if "news" in loc.lower():
        rank = 0
    elif date is not None:
        rank = 1
    else:
        rank = 2
    return rank, -(date.timestamp() if date else 0)


def sitemap_entries(fetcher, base, since, limit):
    """(url, published, title) from the site's sitemaps, only entries dated inside the window."""
    response = fetcher.get(f"{base}/robots.txt")
    queue = re.findall(r"(?im)^sitemap:\s*(\S+)", response.text) if response is not None else []
    queue = queue or [f"{base}/sitemap.xml"]
    seen, entries, fetched = set(), [], 0
    while queue and fetched < MAX_SITEMAPS and len(entries) < limit:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        root = read_xml(fetcher, url)
        fetched += 1
        if root is None:
            continue
        if root.tag.endswith("sitemapindex"):
            children = []
            for node in root.findall("{*}sitemap"):
                loc, lastmod = node.findtext("{*}loc", "").strip(), utc(node.findtext("{*}lastmod"))
                named = URL_MONTH.search(loc)
                old_name = named and datetime(int(named.group(1)), int(named.group(2)), 1,
                                              tzinfo=timezone.utc) < since.replace(day=1)
                if loc and not SKIP_SITEMAP.search(loc) and not (lastmod and lastmod < since) and not old_name:
                    children.append((sitemap_priority(loc, lastmod, since), loc))
            queue = [loc for _, loc in sorted(children + [(sitemap_priority(u, None, since), u) for u in queue])]
            continue
        for node in root.findall("{*}url"):
            loc = node.findtext("{*}loc", "").strip()
            published = utc(node.findtext("{*}news/{*}publication_date")) or utc(node.findtext("{*}lastmod"))
            if loc and published and published >= since:
                entries.append({"url": loc, "published": published, "title": clean(node.findtext("{*}news/{*}title")),
                                "text": "", "via": "sitemap"})
    return entries


def feed_entries(fetcher, feed_url, since):
    response = fetcher.get(feed_url)
    if response is None:
        return []
    entries = []
    for entry in feedparser.parse(response.content).entries:
        published = utc(entry.get("published_parsed") or entry.get("updated_parsed"))
        if entry.get("link") and (published is None or published >= since):
            entries.append({"url": entry["link"], "published": published, "title": clean(entry.get("title")),
                            "text": "", "via": "rss"})
    return entries


def spread_by_day(candidates):
    """Candidates in turn from each day, newest day first, so a capped publisher covers the window."""
    days = defaultdict(list)
    for item in sorted(candidates, key=lambda c: c["published"] or datetime.min.replace(tzinfo=timezone.utc),
                       reverse=True):
        days[item["published"].date() if item["published"] else None].append(item)
    groups, ordered = list(days.values()), []
    while any(groups):
        for group in groups:
            if group:
                ordered.append(group.pop(0))
    return ordered


class Pool:
    def __init__(self, out_dir, built, window_days, hobby_window_days):
        self.out_dir, self.built = out_dir, built
        self.since, self.hobby_since = built - timedelta(days=window_days), built - timedelta(days=hobby_window_days)
        self.items_file, self.rejected_file = out_dir / "items.jsonl", out_dir / "rejected.jsonl"
        self.lock = threading.Lock()
        self.done, self.kept = set(), Counter()
        for path in (self.items_file, self.rejected_file):
            if path.exists():
                for line in path.read_text(encoding="utf-8").splitlines():
                    row = json.loads(line)
                    self.done.add(normalize_url(row["url"]))
                    if path == self.items_file:
                        self.kept[row["source_key"]] += 1

    def record(self, row, rejected=None):
        with self.lock:
            self.done.add(normalize_url(row["url"]))
            path = self.rejected_file if rejected else self.items_file
            if rejected:
                row = {"url": row["url"], "source_key": row["source_key"], "reason": rejected}
            else:
                self.kept[row["source_key"]] += 1
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def add(self, fetcher, candidate, meta, since):
        """Fetch (unless the text is already known), check and store one candidate; True if kept."""
        url = candidate["url"]
        row = {**meta, "url": url}
        if normalize_url(url) in self.done:
            return False
        if SKIP_PATH.search(urlsplit(url).path) or LISTING_PATH.search(url) or len(urlsplit(url).path) < 8:
            self.record(row, "not_article_url")
            return False
        title, text, published = candidate["title"], candidate["text"], candidate["published"]
        written = url_date(url)
        if written and (published is None or written.date() < published.date()):
            published = written
        if published is not None and published < since - timedelta(days=1):
            self.record(row, "outside_window")
            return False
        if not text:
            if not fetcher.allowed(url):
                self.record(row, "robots")
                return False
            response = fetcher.get(url)
            if response is None:
                self.record(row, "fetch_failed")
                return False
            extracted = trafilatura.extract(response.content, url=url, output_format="json", with_metadata=True,
                                            include_comments=False, include_tables=False)
            data = json.loads(extracted) if extracted else {}
            text = CONSENT_NOTICE.sub("", data.get("text") or "")
            if data.get("pagetype") and not ARTICLE_TYPES.search(data["pagetype"]):
                self.record(row, "not_article_page")
                return False
            title = title or clean(data.get("title"))
            page_date = utc(data.get("date"))
            # Sitemap lastmod can be an edit date: trust the page's own date when it is older.
            if page_date and (published is None or page_date.date() < published.date()):
                published = page_date
        if published is None or published < since - timedelta(days=1):
            self.record(row, "outside_window")
            return False
        if not title or not is_article(title, text):
            self.record(row, "not_article")
            return False
        if promo(title):
            self.record(row, "promo_or_deal")
            return False
        self.record({**row, "id": "pool_" + hashlib.sha256(normalize_url(url).encode()).hexdigest()[:16],
                     "domain": host(url), "title": title, "published": published.isoformat(),
                     "words": len(text.split()), "via": candidate["via"], "text": text})
        return True


def publisher_groups(sources):
    """One entry per domain: its feeds, its strongest group, and the name of its first source."""
    rank = {"general_news": 0, "hobby": 1, "existing": 2}
    merged = {}
    for source in sources:
        entry = merged.setdefault(source["domain"], {"domain": source["domain"], "name": source["name"],
                                                     "group": source["group"], "feeds": [], "method": source["method"]})
        if rank[source["group"]] < rank[entry["group"]]:
            entry["group"] = source["group"]
        if source.get("feed"):
            entry["feeds"].append(source["feed"])
    return list(merged.values())


def fill_publisher(pool, fetcher, publisher, cap):
    since = pool.hobby_since if publisher["group"] == "hobby" else pool.since
    key = publisher["domain"]
    if pool.kept[key] >= cap:
        return key, pool.kept[key], "already full"
    base = f"https://www.{key}" if fetcher.get(f"https://www.{key}/robots.txt") is not None else f"https://{key}"
    want = cap * ATTEMPTS_PER_KEPT
    candidates = wordpress_posts(fetcher, base, since, pool.built, want)
    if len(candidates) < cap:
        candidates += sitemap_entries(fetcher, base, since, want)
    for feed in publisher["feeds"]:
        candidates += feed_entries(fetcher, feed, since)
    unique = {}
    for item in candidates:
        if item["url"] and host(item["url"]).endswith(key.split("/")[0]):
            current = unique.get(normalize_url(item["url"]))
            if current is None or (item["text"] and not current["text"]):
                unique[normalize_url(item["url"])] = item
    meta = {"source_key": key, "source": publisher["name"], "group": publisher["group"], "topic": None}
    attempts = 0
    for item in spread_by_day(unique.values()):
        if pool.kept[key] >= cap or attempts >= want:
            break
        if normalize_url(item["url"]) in pool.done:
            continue
        attempts += 1
        pool.add(fetcher, item, meta, since)
    vias = Counter(item["via"] for item in unique.values())
    return key, pool.kept[key], f"{len(unique)} candidates {dict(vias)}"


def search_topics(pool, fetcher, topic_list, out_dir):
    """One SearXNG news search (past month) per topic; up to TOPIC_RESULTS new articles each."""
    searched_file = out_dir / "searched_topics.json"
    searched = set(json.loads(searched_file.read_text(encoding="utf-8"))) if searched_file.exists() else set()
    session = requests.Session()
    session.headers.update(HEADERS)
    empty, jobs = 0, []
    with ThreadPoolExecutor(WORKERS) as executor:
        for topic in topic_list:
            if topic in searched:
                continue
            try:
                results = search(session, topic, 1, "month")
            except (requests.RequestException, ValueError) as exc:
                print(f"  search failed for {topic}: {exc}", flush=True)
                results = []
            empty = empty + 1 if not results else 0
            if empty >= MAX_EMPTY_SEARCHES:
                print("  searches keep coming back empty (engines suspended?); stopping the topic search",
                      flush=True)
                break
            candidates = [{"url": r.get("url", ""), "title": clean(r.get("title")), "text": "",
                           "published": utc(r.get("publishedDate")), "via": "search"} for r in results if r.get("url")]
            jobs.append(executor.submit(fill_topic, pool, fetcher, topic, candidates))
            searched.add(topic)
            searched_file.write_text(json.dumps(sorted(searched)), encoding="utf-8")
            time.sleep(SEARCH_DELAY)
        kept = [job.result() for job in jobs]
    return sum(kept), len(searched)


def fill_topic(pool, fetcher, topic, candidates):
    meta = {"source_key": "search", "source": "SearXNG search", "group": "topic_search", "topic": topic}
    kept = 0
    for item in candidates[:TOPIC_RESULTS * ATTEMPTS_PER_KEPT]:
        if kept >= TOPIC_RESULTS:
            break
        kept += pool.add(fetcher, item, {**meta, "source_key": f"search:{host(item['url'])}"},
                         pool.hobby_since)
    return kept


def prune(out_dir):
    """Move stored articles that fail today's checks (promo titles) to rejected.jsonl, keeping
    embeddings.npy row-aligned; returns the number removed."""
    items_file, vectors_file = out_dir / "items.jsonl", out_dir / "embeddings.npy"
    if not items_file.exists():
        return 0
    rows = [json.loads(line) for line in items_file.read_text(encoding="utf-8").splitlines()]
    keep = [not promo(row["title"]) for row in rows]
    if all(keep):
        return 0
    with (out_dir / "rejected.jsonl").open("a", encoding="utf-8") as handle:
        for row, kept in zip(rows, keep):
            if not kept:
                handle.write(json.dumps({"url": row["url"], "source_key": row["source_key"],
                                         "reason": "promo_or_deal"}) + "\n")
    items_file.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row, kept in zip(rows, keep)
                                  if kept), encoding="utf-8")
    if vectors_file.exists():
        vectors = np.load(vectors_file)
        if len(vectors) == len(rows):
            np.save(vectors_file, vectors[np.array(keep)])
        else:
            vectors_file.unlink()  # out of step: embed() rebuilds it
    return keep.count(False)


def embed(out_dir):
    """Encode the articles not yet in embeddings.npy (items.jsonl only grows between prunes)."""
    from src.embedding_store import encode_texts
    rows = [json.loads(line) for line in (out_dir / "items.jsonl").read_text(encoding="utf-8").splitlines()]
    vectors_file = out_dir / "embeddings.npy"
    done = np.load(vectors_file) if vectors_file.exists() else np.empty((0, 0), np.float32)
    if len(done) > len(rows):
        done = np.empty((0, 0), np.float32)
    texts = [row["title"] + "\n" + " ".join(row["text"].split()[:EMBED_WORDS]) for row in rows[len(done):]]
    if texts:
        new = encode_texts(texts, cache=False, batch_size=16, show_progress=True).astype(np.float32)
        np.save(vectors_file, np.vstack([done, new]) if len(done) else new)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=None, help="Default: data/eval/content_pool/<UTC date>.")
    parser.add_argument("--domains", default="", help="Comma-separated publisher domains (trial run).")
    parser.add_argument("--skip-publishers", action="store_true")
    parser.add_argument("--skip-search", action="store_true")
    parser.add_argument("--skip-embed", action="store_true")
    parser.add_argument("--movie-topics-only", action="store_true",
                        help="Topic search only over the movie review / interview / streaming topics.")
    args = parser.parse_args()

    out_dir = args.out or OUT_DIR / datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    settings_file = out_dir / "settings.json"
    if settings_file.exists():
        settings = json.loads(settings_file.read_text(encoding="utf-8"))
    else:
        settings = {"built_at": datetime.now(timezone.utc).isoformat(), "window_days": WINDOW_DAYS,
                    "hobby_window_days": HOBBY_WINDOW_DAYS, "caps": CAPS, "topic_results": TOPIC_RESULTS}
        settings_file.write_text(json.dumps(settings, indent=1), encoding="utf-8")
    built = utc(settings["built_at"])
    pool = Pool(out_dir, built, settings["window_days"], settings["hobby_window_days"])
    fetcher = Fetcher()
    print(f"pool {out_dir}; resuming with {len(pool.done)} URLs already handled", flush=True)
    removed = prune(out_dir)
    if removed:
        print(f"  removed {removed} promo / deal articles", flush=True)
        pool = Pool(out_dir, built, settings["window_days"], settings["hobby_window_days"])

    sources = json.loads(SOURCES_FILE.read_text(encoding="utf-8"))["sources"]
    publishers = publisher_groups(sources)
    if args.domains:
        wanted = {d.strip() for d in args.domains.split(",")}
        publishers = [p for p in publishers if p["domain"] in wanted]
    started = time.time()
    if args.movie_topics_only:
        args.skip_publishers = True
    if not args.skip_publishers:
        with ThreadPoolExecutor(WORKERS) as executor:
            jobs = [executor.submit(fill_publisher, pool, fetcher, p, settings["caps"][p["group"]]) for p in publishers]
            for publisher, job in zip(publishers, jobs):
                try:
                    key, kept, note = job.result()
                except Exception as exc:  # one broken site must not stop the pool
                    print(f"  {publisher['domain']}: failed ({exc!r})", flush=True)
                    continue
                print(f"  {key}: {kept} kept ({note}) [{time.time() - started:.0f}s]", flush=True)
    if not args.skip_search and not args.domains:
        kept, searched = search_topics(pool, fetcher, movie_topics() if args.movie_topics_only else topics(),
                                       out_dir)
        print(f"  topic search: {searched} topics searched, {kept} articles kept [{time.time() - started:.0f}s]",
              flush=True)

    rows = [json.loads(line) for line in pool.items_file.read_text(encoding="utf-8").splitlines()] \
        if pool.items_file.exists() else []
    if rows and not args.skip_embed:
        embed(out_dir)
    rejected = [json.loads(line) for line in pool.rejected_file.read_text(encoding="utf-8").splitlines()] \
        if pool.rejected_file.exists() else []
    manifest = {**settings, "items": len(rows), "by_group": Counter(r["group"] for r in rows),
                "by_via": Counter(r["via"] for r in rows), "by_source": Counter(r["source"] for r in rows),
                "rejected": Counter(r["reason"] for r in rejected),
                "published_range": [min((r["published"] for r in rows), default=None),
                                    max((r["published"] for r in rows), default=None)],
                "median_words": int(np.median([r["words"] for r in rows])) if rows else 0}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("items", "by_group", "by_via", "rejected", "median_words")},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
