"""Fetch articles from topic publishers (Soompi, PC Gamer, Vogue, ...) chosen for this listener.

Each interest in the profile is matched to the closest publishers in site_catalog.json by
embedding similarity, then searched there with a SearXNG site: query. An interest no publisher
covers is skipped; the general news search still reaches it. Articles pass the same text and
relevance checks as the news, and interests' sources (news / movies / reviews, when the profile
records them) share the budget evenly. Only freely readable pages are kept.

Genres are shared by many listeners, so movie interests search the films the listener liked and
fall back to the genre when none of those films has an article.
"""
import argparse
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import requests

from src import paths
from src.user_profile.fetch_breaking_news import (
    HEADERS, article_text, clean, filter_relevant, is_article, load_profile, normalize_url, search)
from src.user_profile.local_translation import save_json

CATALOG_FILE = Path(__file__).with_name("site_catalog.json")
OUTPUT_FILE = paths.DAILY_DIR / "site_articles.json"
PROFILE_FILE = paths.OUTPUT_DIR / "user_profile.json"
# With the latest news (at most 15 kept) a listener gets at most 65 articles a day, for about
# 200 trunks of 20-30 seconds.
MAX_ARTICLES = 50
SITES_PER_INTEREST = 2
MAX_PER_INTEREST = 3  # Without a cap, the interests with the busiest publishers crowd out the rest.
ARTICLES_PER_SEARCH = 2
# Below this an interest has no real match in the catalog; checked on the 50 merged personas.
MIN_SIMILARITY = 0.55
# Search engines suspend a SearXNG that queries in bursts; a run of empty searches means they did.
SEARCH_DELAY = 8
MAX_EMPTY_SEARCHES = 6
CANARY_QUERY = "site:espn.com NFL"
# Listing pages rank well in site: searches but are not articles.
LISTING_PATH = re.compile(r"/(tag|tags|topic|topics|category|categories|hub|author|authors|section|collection|gallery)/", re.I)
GENERIC_TITLES = {"Common Sense Media Review"}
# Films are old, so their searches have no time range; only Bing News serves those reliably.
FILM_ENGINES = "bing news"
FILM_CANDIDATES = 5
FILM_TITLE = re.compile(r"^(.+?)(?: \(.*\))? \((\d{4})\)$")
TRAILING_ARTICLE = re.compile(r"^(.+), (The|A|An)$")


def article_title(title, url):
    """The search title, or the URL slug when the publisher gives every page the same one."""
    if title not in GENERIC_TITLES:
        return title
    return urlsplit(url).path.rstrip("/").rsplit("/", 1)[-1].replace("-", " ").title()


def balanced(items, count, source):
    """Up to count items taken in turn from each source, keeping each source's order."""
    groups = {}
    for item in items:
        groups.setdefault(source(item), []).append(item)
    picked = []
    while len(picked) < count and any(groups.values()):
        for group in groups.values():
            if group and len(picked) < count:
                picked.append(group.pop(0))
    return picked


def load_catalog():
    return json.loads(CATALOG_FILE.read_text(encoding="utf-8"))


def route(profile, catalog):
    """(interest, source, site, similarity) for the closest publishers of each interest."""
    from src.embedding_store import encode_texts
    interests = profile.get("interests") or []
    if not interests:
        return []
    sources = {e["interest"]: e["source"] for e in profile.get("inference", {}).get("interest_evidence", [])}
    vectors = encode_texts([f"{site['name']}: {site['topics']}" for site in catalog] + interests)
    sites, wanted = vectors[:len(catalog)], vectors[len(catalog):]
    routes = []
    for interest, vector in zip(interests, wanted):
        scores = sites @ vector
        for index in scores.argsort()[::-1][:SITES_PER_INTEREST]:
            if scores[index] >= MIN_SIMILARITY:
                routes.append((interest, sources.get(interest), catalog[index], float(scores[index])))
    return routes


def film_name(title):
    """('The Game', '1997') from a MovieLens title such as 'Game, The (1997)', or None."""
    match = FILM_TITLE.match(title)
    if not match:
        return None
    return TRAILING_ARTICLE.sub(r"\2 \1", match.group(1)), match.group(2)


def liked_films(profile):
    """{movie interest: [(name, year), ...]} from the films that evidence each interest."""
    films = {}
    for evidence in profile.get("inference", {}).get("interest_evidence", []):
        if evidence.get("source") == "movies":
            names = [film_name(title) for title in evidence.get("examples") or []]
            films[evidence["interest"]] = [name for name in names if name]
    return films


def names_film(name, text):
    pattern = rf"(?<!\w){re.escape(name.replace(chr(8217), chr(39)))}(?!\w)"
    return re.search(pattern, text.replace(chr(8217), chr(39)), re.I) is not None


def about_film(name, year, title, text):
    """The article discusses this film: a one-word name alone ('Heat') matches too much."""
    return (names_film(name, text) and year in text) or (len(name.split()) > 1 and names_film(name, title))


def on_site(url, domain):
    """A page on the publisher that is not its home page."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    return ((host == domain or host.endswith(f".{domain}")) and len(parts.path.strip("/")) > 10
            and not LISTING_PATH.search(parts.path))


class SiteSearch:
    """Paced searches that stop once SearXNG's engines appear to be suspended."""

    def __init__(self, session):
        self.session = session
        self.empty = 0

    def __call__(self, query, time_range, engines=None):
        print(f"Searching '{query}' ({time_range or 'any time'})...", flush=True)
        results = search(self.session, query, 1, time_range, engines)
        time.sleep(SEARCH_DELAY)
        self.empty = 0 if results else self.empty + 1
        if self.empty >= MAX_EMPTY_SEARCHES:
            # Long hobby names ('Listening to Memphis blues and jazz') find nothing on one site
            # even when the engines work; a query that always has results tells the two apart.
            alive = search(self.session, CANARY_QUERY, 1, "week", engines)
            time.sleep(SEARCH_DELAY)
            if not alive:
                raise RuntimeError("Searches keep returning nothing; SearXNG's engines may be suspended.")
            self.empty = 0
        return results


def site_item(session, result, url, title, key, site, interest, source, query, score):
    """The article at this search result, or None when the page has no readable article."""
    text = clean(article_text(session, url))
    if not is_article(title, text):
        print(f"  Skipped: no article text — {title}", flush=True)
        return None
    return {"item_id": "site_article_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16],
            "source": "site_article", "title": title, "text": text[:12000], "url": url,
            "creator": site["name"], "published_at": result.get("publishedDate"),
            "interest": interest, "interest_source": source, "query": query,
            "site_similarity": round(score, 3)}


def collect_films(session, searcher, routes, films, seen):
    """One article per liked film, from the first of the interest's publishers that has one."""
    items, done = [], set()
    for interest, names in films.items():
        sites = [r for r in routes if r[0] == interest]
        for name, year in names:
            found = name in done  # A film can evidence two genres.
            done.add(name)
            for _, source, site, score in sites:
                if found:
                    break
                query = f'site:{site["domain"]} "{name}"'
                results = searcher(query, None, FILM_ENGINES)
                # Downloading is slow, so only results that name the film are opened.
                candidates = [r for r in results if names_film(name, f"{r.get('title')} {r.get('content')}")]
                for result in candidates[:FILM_CANDIDATES]:
                    url = clean(result.get("url"))
                    title, key = article_title(clean(result.get("title")), url), normalize_url(url)
                    if not title or not on_site(url, site["domain"]) or key in seen:
                        continue
                    seen.add(key)
                    item = site_item(session, result, url, title, key, site, interest, source, query, score)
                    if item and about_film(name, year, title, item["text"]):
                        items.append(dict(item, film=name, matched_interest=interest))
                        found = True
                        print(f"[film] ({site['name']}) {name}: {title}", flush=True)
                        break
    return items


def collect(session, searcher, routes, seen):
    items = []
    for interest, source, site, score in routes:
        query, found = f"site:{site['domain']} {interest}", 0
        # Reviews and interviews stay current longer than news, so a quiet week widens to a month.
        for time_range in ("week", "month"):
            if found:
                break
            for result in searcher(query, time_range):
                if found >= ARTICLES_PER_SEARCH:
                    break
                url = clean(result.get("url"))
                title, key = article_title(clean(result.get("title")), url), normalize_url(url)
                if not title or not on_site(url, site["domain"]) or key in seen:
                    continue
                seen.add(key)
                item = site_item(session, result, url, title, key, site, interest, source, query, score)
                if not item:
                    continue
                items.append(item)
                found += 1
                print(f"[{len(items)}] ({site['name']}) {title}", flush=True)
    return items


def limit(items, max_articles):
    """Items under the per-interest cap, sharing the budget across interest sources."""
    per_interest = Counter()
    capped = [item for item in items
              if per_interest.update([item["interest"]]) or per_interest[item["interest"]] <= MAX_PER_INTEREST]
    return balanced(capped, max_articles, lambda item: item["interest_source"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=PROFILE_FILE)
    parser.add_argument("--max-articles", type=int, default=MAX_ARTICLES)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    args = parser.parse_args(argv)
    profile = load_profile(args.profile)
    routes = route(profile, load_catalog())
    print(f"{len(routes)} interest-site searches across {len({r[2]['domain'] for r in routes})} sites.", flush=True)
    with requests.Session() as session:
        session.headers.update(HEADERS)
        searcher, seen = SiteSearch(session), set()
        film_items = collect_films(session, searcher, routes, liked_films(profile), seen)
        covered = {item["interest"] for item in film_items}
        items = collect(session, searcher, [r for r in routes if r[0] not in covered], seen)
    # A film the listener liked needs no relevance check; the LLM rejects them on old titles alone.
    relevant = filter_relevant(profile, items, lambda item: f"{item['title']}. {item['text'][:300]}")
    items = limit(film_items + relevant, args.max_articles)
    save_json(items, args.output)
    print(f"Saved {len(items)} site articles to {args.output}", flush=True)


if __name__ == "__main__":
    main()
