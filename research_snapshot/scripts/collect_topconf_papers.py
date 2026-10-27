"""Collect verified public paper sources and extract page-indexed text locally."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import time
from urllib.parse import urljoin

import fitz
import requests


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links, self.current = [], None

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.current = [dict(attrs).get("href", ""), ""]

    def handle_data(self, value):
        if self.current is not None:
            self.current[1] += value

    def handle_endtag(self, tag):
        if tag == "a" and self.current is not None:
            self.links.append(self.current)
            self.current = None


def fetch(url, limit=40 * 1024 * 1024):
    errors = []
    for attempt in range(3):
        try:
            with requests.get(url, timeout=(15, 60), stream=True,
                              headers={"User-Agent": "PHM-public-literature-research/1.0"}) as response:
                response.raise_for_status()
                chunks, size = [], 0
                for chunk in response.iter_content(256 * 1024):
                    size += len(chunk)
                    if size > limit:
                        raise ValueError("Public source exceeds bounded download size")
                    chunks.append(chunk)
                content = b"".join(chunks)
                return content, {"requested_url": url, "final_url": response.url,
                                 "http_status": response.status_code,
                                 "content_type": response.headers.get("Content-Type"),
                                 "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                                 "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
                                 "earlier_attempt_errors": errors}
        except Exception as error:
            errors.append({"attempt": attempt + 1, "error": str(error)})
            if attempt < 2:
                time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(json.dumps(errors))


def collect(paper, run):
    directory = run / "papers" / (paper["id"] + "_" + paper["slug"])
    directory.mkdir(parents=True, exist_ok=True)
    result = dict(paper)
    try:
        landing, landing_record = fetch(paper["landing_url"])
        (directory / "venue_page.html").write_bytes(landing)
        result["landing_retrieval"] = landing_record
        pdf_url = paper.get("pdf_url")
        if not pdf_url:
            parser = Links()
            parser.feed(landing.decode("utf-8", errors="replace"))
            candidates = [href for href, label in parser.links
                          if label.strip().lower() in ("paper", "pdf", "view pdf", "download pdf")]
            if len(candidates) != 1:
                raise ValueError("Expected one official paper link: " + repr(candidates))
            pdf_url = urljoin(landing_record["final_url"], candidates[0])
        content, result["pdf_retrieval"] = fetch(pdf_url)
        if not content.startswith(b"%PDF-"):
            raise ValueError("Source did not return PDF bytes")
        (directory / "paper.pdf").write_bytes(content)
        with fitz.open(stream=content, filetype="pdf") as document:
            pages = [page.get_text(sort=True) for page in document]
            result["pdf_pages"] = len(pages)
            result["pdf_metadata"] = document.metadata
        compact = lambda text: re.sub(r"[^a-z0-9]", "", text.lower())
        result["title_in_first_two_pages"] = compact(paper["title"]) in compact(" ".join(pages[:2]))
        if not result["title_in_first_two_pages"]:
            raise ValueError("Downloaded first pages do not match the frozen title")
        (directory / "pages.json").write_text(json.dumps(
            [{"pdf_page": i + 1, "text": value} for i, value in enumerate(pages)],
            ensure_ascii=False, indent=2) + "\n")
        (directory / "full_text.txt").write_text("\f".join(pages))
        result["total_text_characters"] = sum(map(len, pages))
        result["status"] = "downloaded_and_title_verified"
    except Exception as error:
        result["status"], result["error"] = "failed", str(error)
    (directory / "retrieval.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--selection", default="selection_frozen61.json")
    parser.add_argument("--summary", default="retrieval_summary61.json")
    args = parser.parse_args()
    selected = json.loads((args.run / args.selection).read_text())["selected"]
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(collect, paper, args.run): paper for paper in selected}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps({key: result.get(key) for key in
                              ("id", "status", "pdf_pages", "error")}), flush=True)
            (args.run / args.summary).write_text(json.dumps(
                sorted(results, key=lambda x: x["id"]), ensure_ascii=False, indent=2) + "\n")
    if any(result["status"] == "failed" for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
