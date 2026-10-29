"""Collect the Instagram home feed and YouTube homepage into the daily *_manual.json files.

Runs the same fetch_*.js scripts used manually in the DevTools console, in your installed
Chrome with a dedicated, persistent login profile (data/.browser_profile, never committed).
Chrome is started normally and Playwright only attaches over the DevTools port for fetching,
because Google and Instagram refuse logins in Playwright-launched browsers.

First time (log in to Instagram and YouTube in the window that opens, close it, press Enter):
  python -m src.fetch_code.auto_fetch --login
Daily:
  python -m src.fetch_code.auto_fetch
"""
import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DAILY_DIR = PROJECT_ROOT / "data" / "daily"
SCRIPTS = Path(__file__).resolve().parent
PROFILE_DIR = PROJECT_ROOT / "data" / ".browser_profile"
SOURCES = {
    "instagram": {"url": "https://www.instagram.com/", "script": SCRIPTS / "fetch_instagram.js",
                  "output": DAILY_DIR / "instagram_feed_manual.json"},
    "youtube": {"url": "https://www.youtube.com/", "script": SCRIPTS / "fetch_youtube.js",
                "output": DAILY_DIR / "youtube_homepage_manual.json"},
}


class NotLoggedIn(RuntimeError):
    pass


def logged_in(name, page):
    if name == "instagram":
        return "/accounts/login" not in page.url and not page.locator('input[name="password"]').count()
    # YouTube shows a "Sign in" link to accounts.google.com when logged out.
    return not page.locator('a[href^="https://accounts.google.com/ServiceLogin"]').count()


def save_atomic(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(rows, file, ensure_ascii=False, indent=2)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def fetch(name, page, count, settle_ms):
    source = SOURCES[name]
    page.goto(source["url"], wait_until="domcontentloaded")
    page.wait_for_timeout(settle_ms)
    if not logged_in(name, page):
        raise NotLoggedIn(f"{name}: not logged in; run python -m src.fetch_code.auto_fetch --login")
    script = source["script"].read_text(encoding="utf-8").strip().rstrip(";")
    page.evaluate("count => { window.__PODCAST_FETCH__ = {count}; }", count)
    rows = page.evaluate(script)
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"{name}: collected no items; the page layout may have changed.")
    save_atomic(rows, source["output"])
    print(f"{name}: saved {len(rows)} items to {source['output']}", flush=True)
    return rows


def find_chrome(path=None):
    candidates = [path, shutil.which("chrome"), shutil.which("chrome.exe"),
                  *(Path(os.environ.get(v, "")) / "Google/Chrome/Application/chrome.exe"
                    for v in ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA")),
                  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                  shutil.which("google-chrome")]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    raise FileNotFoundError("Google Chrome not found; pass --chrome-path.")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_chrome(chrome, profile_dir, *extra):
    # Plain Chrome, no automation flags: Google/Instagram reject logins in Playwright-launched browsers.
    profile_dir.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen([chrome, f"--user-data-dir={profile_dir}", "--no-first-run",
                             "--no-default-browser-check", *extra])


def wait_for_debugger(port, process, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Chrome exited at startup; close other windows using the fetch profile.")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=1):
                return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError("Chrome did not open its debugging port.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=list(SOURCES), action="append",
                        help="Fetch only this source (repeatable). Default: all.")
    parser.add_argument("--login", action="store_true",
                        help="Open the sites in plain Chrome to log in once; kept in --profile-dir.")
    parser.add_argument("--count", type=int, default=100, help="Items per source.")
    parser.add_argument("--profile-dir", type=Path, default=PROFILE_DIR)
    parser.add_argument("--chrome-path", help="Chrome executable; auto-detected by default.")
    parser.add_argument("--settle-ms", type=int, default=4000,
                        help="Wait after page load before checking login and scrolling.")
    args = parser.parse_args(argv)
    if args.count < 1:
        parser.error("--count must be positive.")
    names = args.only or list(SOURCES)
    profile_dir = args.profile_dir.resolve()
    try:
        chrome = find_chrome(args.chrome_path)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    if args.login:
        process = start_chrome(chrome, profile_dir, *(SOURCES[n]["url"] for n in names))
        input("Log in to every opened site in that Chrome window, CLOSE the window, then press Enter... ")
        if process.poll() is None:
            process.terminate()
        return 0

    from playwright.sync_api import Error as PlaywrightError, sync_playwright
    port = free_port()
    process = start_chrome(chrome, profile_dir, f"--remote-debugging-port={port}", "about:blank")
    try:
        wait_for_debugger(port, process)
        with sync_playwright() as playwright:
            browser = playwright.chromium.connect_over_cdp(f"http://127.0.0.1:{port}")
            context = browser.contexts[0]
            page = context.new_page()
            failed = []
            for name in names:
                try:
                    fetch(name, page, args.count, args.settle_ms)
                except (NotLoggedIn, RuntimeError, PlaywrightError) as exc:
                    failed.append(name)
                    print(f"{name}: failed: {exc}", file=sys.stderr, flush=True)
            browser.close()
        if failed:
            print(f"Previous files kept for: {', '.join(failed)}", file=sys.stderr)
            return 1
        return 0
    except (RuntimeError, PlaywrightError) as exc:
        print(f"Could not drive Chrome: {exc}", file=sys.stderr)
        return 1
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    raise SystemExit(main())
