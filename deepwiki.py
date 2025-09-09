#!/usr/bin/env python3
# hiielloddsad
# pip install playwright
# python3 -m playwright install chromium
# (optional) python3 -m playwright install-deps
"""
DeepWiki RSC listener: capture 'text/x-component' / '?rsc=' responses,
extract Markdown-ish 'content' fields, and save them.

Usage:
  python scrapt.py https://deepwiki.com/juspay/hyperswitch ./out2 --verbose --keep-raw
"""

import argparse
import asyncio
import hashlib
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

from playwright.async_api import async_playwright

# ------------------------- helpers -------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def slugify(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9._-]+", "-", s)
    s = re.sub(r"^-+|-+$", "", s)
    return s[:200] or "file"

def filename_from_url(u: str) -> str:
    try:
        parsed = urlparse(u)
        parts = [p for p in parsed.path.split("/") if p]
        last = parts[-1] if parts else "index"
        last = re.sub(r"\.(json|txt|js|jsx|ts|tsx)$", "", last, flags=re.I)
        return slugify(last)
    except Exception:
        return "response"

def json_unescape(s: str) -> str:
    try:
        return json.loads(f'"{s.replace("\\", "\\\\").replace("\"", "\\\"")}"')
    except Exception:
        return s

def looks_like_markdown(text: str) -> bool:
    return bool(
        re.search(r"^\s*#{1,6}\s+\S", text, re.M) or
        re.search(r"^\s*[-*]\s+\S", text, re.M) or
        re.search(r"^\s*\d+\.\s+\S", text, re.M) or
        "```" in text or
        re.search(r"^\s*>\s+\S", text, re.M) or
        re.search(r"\[[^\]]+\]\([^)]+\)", text) or
        "\n\n" in text
    )

def try_parse_json(txt: str):
    t = txt.lstrip()
    if not (t.startswith("{") or t.startswith("[")):
        return None
    try:
        return json.loads(t)
    except Exception:
        return None

def extract_markdown_candidates(body: str):
    """
    Return list of {name, content, looksLikeMD}.
    Heuristics in order:
      0) Whole body already *looks like Markdown* -> take as-is.
      1) Body is JSON -> walk for "content": <str>.
      2) RSC-ish: parse any {...} chunks -> "content": <str>.
      3) Loose pairing: "name": "...", (anything), "content": "..."
    """
    results = []

    def add(name, content):
        if not content:
            return
        results.append({
            "name": name or None,
            "content": content,
            "looksLikeMD": looks_like_markdown(content),
        })

    # 0) whole body MD
    if looks_like_markdown(body):
        add(None, body)

    # 1) whole JSON
    parsed = try_parse_json(body)
    if parsed is not None:
        def walk(node):
            if isinstance(node, dict):
                if isinstance(node.get("content"), str):
                    nm = node.get("name") if isinstance(node.get("name"), str) else None
                    if not nm and isinstance(node.get("title"), str):
                        nm = node["title"]
                    add(nm, node["content"])
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)
        walk(parsed)

    # 2) brace chunks (very loose)
    # NOTE: This is heuristic and intentionally permissive.
    for obj_str in re.findall(r"\{(?:[^{}]|(?:\{[^{}]*\}))*\}", body, flags=re.DOTALL):
        try:
            o = json.loads(obj_str)
            if isinstance(o, dict) and isinstance(o.get("content"), str):
                nm = o.get("name") or o.get("title")
                add(nm if isinstance(nm, str) else None, o["content"])
        except Exception:
            pass

    # 3) loose pairs — FIXED: use '.*?' with DOTALL (no JS-style [^]*?)
    rx = re.compile(r'"name"\s*:\s*"(.*?)".*?"content"\s*:\s*"(.*?)"', re.DOTALL)
    for m in rx.finditer(body):
        add(json_unescape(m.group(1)), json_unescape(m.group(2)))

    # Deduplicate by content hash
    dedup = []
    seen = set()
    for r in results:
        h = hashlib.sha256(r["content"].encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            dedup.append(r)
    return dedup

# ------------------------- scraper -------------------------

async def scrape(url: str, out_dir: Path, verbose: bool = False, keep_raw: bool = False) -> None:
    ensure_dir(out_dir)
    seen_by_resp = set()     # (url, size)
    seen_by_content = set()  # content hash

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(bypass_csp=True)
        page = await context.new_page()

        async def on_response(resp):
            try:
                r_url = resp.url
                headers = resp.headers
                ct = (headers.get("content-type") or "").lower()

                is_rsc = ("text/x-component" in ct) or (re.search(r"[?&]rsc=", r_url) is not None)
                if not is_rsc:
                    return

                body_bytes = await resp.body()
                key = (r_url, len(body_bytes))
                if key in seen_by_resp:
                    if verbose:
                        print(f"[skip-dup] {r_url} ({len(body_bytes)} bytes)")
                    return
                seen_by_resp.add(key)

                body = body_bytes.decode("utf-8", errors="replace")
                slug = filename_from_url(r_url)
                base = out_dir / slug

                cands = extract_markdown_candidates(body)

                wrote_any = False
                for idx, c in enumerate(cands):
                    h = hashlib.sha256(c["content"].encode("utf-8")).hexdigest()
                    if h in seen_by_content:
                        continue
                    seen_by_content.add(h)

                    name_slug = slugify(c["name"] or (f"{slug}-{idx}" if idx else slug))
                    md_path = out_dir / f"{name_slug}.md"
                    md_path.write_text(c["content"], encoding="utf-8")
                    wrote_any = True
                    print(f"[ok]  {md_path.name} ({len(c['content'])} chars)  <- {r_url}")

                if (keep_raw or not wrote_any):
                    raw_path = base.with_suffix(".raw.txt")
                    raw_path.write_text(body, encoding="utf-8")
                    if not wrote_any:
                        print(f"[raw] {raw_path.name} (no markdown found)  <- {r_url}")
                    elif verbose:
                        print(f"[raw] kept {raw_path.name}")

            except Exception as e:
                # Never crash the listener; report and continue
                print("response handler error:", repr(e), file=sys.stderr)

        page.on("response", on_response)

        if verbose:
            page.on("request", lambda r: print("[req]", r.method, r.url))

        print("Navigating:", url)
        await page.goto(url, wait_until="networkidle")

        # Scroll to trigger lazy loads / prefetch
        async def auto_scroll(ms: int = 8000):
            t0 = asyncio.get_event_loop().time()
            y = 0
            while (asyncio.get_event_loop().time() - t0) * 1000 < ms:
                y += 700
                await page.evaluate("(yy) => window.scrollTo(0, yy)", y)
                await page.wait_for_timeout(250)
        await auto_scroll(8000)

        await page.wait_for_timeout(2000)
        await browser.close()
        print("Done. Files saved to:", str(out_dir.resolve()))

# ------------------------- CLI -------------------------

def main():
    ap = argparse.ArgumentParser(description="Capture DeepWiki RSC responses and save Markdown.")
    ap.add_argument("url", help="DeepWiki page (e.g., https://deepwiki.com/juspay/hyperswitch)")
    ap.add_argument("out", nargs="?", default="./out2", help="Output directory")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging")
    ap.add_argument("--keep-raw", action="store_true", help="Always save .raw.txt alongside extracted Markdown")
    args = ap.parse_args()

    asyncio.run(scrape(args.url, Path(args.out), verbose=args.verbose, keep_raw=args.keep_raw))

if __name__ == "__main__":
    main()
