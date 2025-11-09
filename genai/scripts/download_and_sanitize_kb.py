#!/usr/bin/env python3
"""
download_and_sanitize_kb.py

Fetches a list of article URLs, extracts main article text using readability,
sanitizes content, and writes plain .txt files into kb/ with a small metadata header.

Updated: adds retries and a safer default URL list (NIMH, CDC facts index, NHS, Wikipedia, CHADD).
"""
import os
import re
import sys
import json
import time
from datetime import datetime
from urllib.parse import urlparse
import requests
from readability import Document
from bs4 import BeautifulSoup
from dateutil import parser as dateparser
import tldextract

# === Updated default URLs (stable, public pages)
DEFAULT_URLS = [
    "https://www.nimh.nih.gov/health/topics/attention-deficit-hyperactivity-disorder-adhd",
    "https://www.cdc.gov/adhd/facts/index.html",
    "https://www.nhs.uk/conditions/attention-deficit-hyperactivity-disorder-adhd/",
    "https://en.wikipedia.org/wiki/Attention_deficit_hyperactivity_disorder",
    "https://chadd.org/about-adhd/"  # CHADD main overview (may sometimes be rate-limited)
]

KB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__) or ".", "..", "kb"))
HEADERS = {
    "User-Agent": "Sakshitha-A KB Crawler (+https://yourproject.example) - for research/edu use"
}

# Sanitization helpers
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(\+?\d[\d\-\s\(\)]{6,}\d)")
MULTI_SPACE = re.compile(r"\s{2,}")
HTML_ENTITY_RE = re.compile(r"&[a-z]+;")
MAX_CHARS = 120_000

def safe_filename_from_url(url):
    parsed = urlparse(url)
    ext = tldextract.extract(url)
    domain = ext.domain or parsed.netloc
    path = parsed.path.strip("/").replace("/", "_")
    if not path:
        path = "home"
    fname = f"{domain}__{path}"
    fname = re.sub(r"[^A-Za-z0-9_\-\.]", "_", fname)
    return (fname[:160] + ".txt") if len(fname) > 160 else (fname + ".txt")

def fetch_url_with_retries(url, timeout=25, retries=3, backoff=1.5):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            return r.status_code, r.text
        except requests.RequestException as e:
            last_exc = e
            wait = backoff ** attempt
            print(f"Attempt {attempt} failed for {url}: {e}. Retrying in {wait:.1f}s...")
            time.sleep(wait)
    raise last_exc

def extract_main_text(html, url=None):
    doc = Document(html)
    title = doc.short_title() or ""
    content_html = doc.summary()
    soup = BeautifulSoup(html, "html.parser")
    publish_date = None
    for tag in ("meta[name='pubdate']", "meta[name='publishdate']", "meta[property='article:published_time']"):
        el = soup.select_one(tag)
        if el and el.get("content"):
            try:
                publish_date = dateparser.parse(el["content"])
                break
            except Exception:
                pass
    # Extract readable paragraphs
    content_soup = BeautifulSoup(content_html, "html.parser")
    paragraphs = []
    for p in content_soup.find_all(["p", "li"]):
        text = p.get_text(separator=" ", strip=True)
        if text:
            paragraphs.append(text)
    text = "\n\n".join(paragraphs).strip()
    if not text:
        ps = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
        text = "\n\n".join([p for p in ps if p]).strip()
    return {"title": title.strip(), "content_html": content_html, "text": text, "publish_date": publish_date}

def sanitize_text(text):
    if not text:
        return ""
    t = HTML_ENTITY_RE.sub(" ", text)
    t = EMAIL_RE.sub("[email removed]", t)
    t = PHONE_RE.sub("[phone removed]", t)
    t = MULTI_SPACE.sub(" ", t)
    t = t.replace("\r", "")
    t = "\n\n".join([para.strip() for para in t.split("\n\n") if para.strip()])
    if len(t) > MAX_CHARS:
        t = t[:MAX_CHARS]
        last_n = t.rfind("\n")
        if last_n > int(MAX_CHARS * 0.6):
            t = t[:last_n]
    return t.strip()

def save_kb_file(fname, url, title, publish_date, text):
    os.makedirs(KB_DIR, exist_ok=True)
    fullpath = os.path.join(KB_DIR, fname)
    header_lines = [
        f"Source-URL: {url}",
        f"Title: {title or 'Untitled'}",
        f"Fetched-At: {datetime.utcnow().isoformat()}Z",
        f"Publish-Date: {publish_date.isoformat() if publish_date else 'unknown'}",
        "-" * 40,
    ]
    body = "\n\n".join(header_lines) + "\n\n" + text + "\n"
    with open(fullpath, "w", encoding="utf-8") as f:
        f.write(body)
    return fullpath

def main(urls=None, delay=1.0, only_if_missing=True):
    if urls is None:
        urls = DEFAULT_URLS
    saved = []
    for url in urls:
        try:
            fname = safe_filename_from_url(url)
            target_path = os.path.join(KB_DIR, fname)
            if only_if_missing and os.path.exists(target_path):
                print(f"Skipping (already exists): {target_path}")
                saved.append({"url": url, "path": target_path, "skipped": True})
                continue

            print(f"Fetching: {url}")
            status, html = fetch_url_with_retries(url, timeout=25, retries=3, backoff=1.5)
            print(f"  status: {status}; extracting...")
            info = extract_main_text(html, url=url)
            text = sanitize_text(info["text"])
            if not text or len(text) < 180:
                soup = BeautifulSoup(html, "html.parser")
                maybe = "\n\n".join([p.get_text(strip=True) for p in soup.find_all("p")])[:MAX_CHARS]
                text = sanitize_text(maybe)
            if not text:
                print(f"  Warning: no extractable text for {url}; skipping.")
                saved.append({"url": url, "path": None, "skipped": True, "reason": "no_text"})
                continue
            path = save_kb_file(fname, url, info.get("title", ""), info.get("publish_date", None), text)
            print(f"  Saved: {path} ({len(text)} chars)")
            saved.append({"url": url, "path": path, "skipped": False})
            time.sleep(delay)
        except Exception as e:
            print(f"  Error fetching {url}: {e}")
            saved.append({"url": url, "path": None, "skipped": True, "error": str(e)})
    print("\nSummary:")
    for s in saved:
        print(" -", s)
    return saved

if __name__ == "__main__":
    urls_to_use = None
    cfg_file = os.path.join(os.path.dirname(__file__), "kb_urls.json")
    if os.path.exists(cfg_file):
        try:
            with open(cfg_file, "r", encoding="utf-8") as fh:
                js = json.load(fh)
            if isinstance(js, list):
                urls_to_use = js
        except Exception:
            pass
    if len(sys.argv) > 1:
        urls_to_use = sys.argv[1:]
    main(urls=urls_to_use)
