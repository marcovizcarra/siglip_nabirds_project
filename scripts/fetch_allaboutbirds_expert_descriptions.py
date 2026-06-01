"""
Populate compact expert visual descriptions from All About Birds ID pages.

Usage:
  pip install requests beautifulsoup4 lxml tqdm
  python fetch_allaboutbirds_expert_descriptions.py \
      --input bird_class_expert_description_coverage_with_expert_descriptions.csv \
      --output bird_class_expert_description_coverage_fetched.csv

Notes:
- This script is for an educational class project.
- Follow Cornell/All About Birds terms of use.
- Prefer storing compact normalized visual traits + source URL, not long raw page text.
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

HEADERS = {
    "User-Agent": (
        "CS231N-student-project/0.1 "
        "(educational data enrichment; store compact visual traits only)"
    )
}

STOP_HEADINGS = {
    "relative size", "measurements", "behavior", "habitat", "regional differences",
    "species in this family", "compare with similar species", "looking for id help?",
    "photo gallery", "overview", "life history", "maps", "sounds"
}

TARGET_SECTIONS = ["size & shape", "color pattern"]


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"© .*?Macaulay Library", "", text)
    return text.strip()


def fetch_html(url: str, cache_path: Path, delay: float) -> Optional[str]:
    if not url or not url.startswith("http"):
        return None

    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8", errors="ignore")

    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        time.sleep(delay)
        if response.status_code != 200:
            return None
        cache_path.write_text(response.text, encoding="utf-8")
        return response.text
    except requests.RequestException:
        return None


def html_to_lines(html: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    lines = [clean_text(x) for x in soup.get_text("\n").splitlines()]
    return [x for x in lines if x]


def extract_section(lines: list[str], section_name: str, max_chars: int = 450) -> str:
    section = section_name.lower()
    start = None

    for i, line in enumerate(lines):
        if line.lower().strip() == section:
            start = i + 1
            break

    if start is None:
        return ""

    chunks = []
    for line in lines[start:]:
        low = line.lower().strip()
        if low in STOP_HEADINGS or low in TARGET_SECTIONS:
            break
        if len(line) < 12:
            continue
        if line.startswith("Image:"):
            continue
        if "Macaulay Library" in line:
            continue
        chunks.append(line)
        if len(" ".join(chunks)) >= max_chars:
            break

    return clean_text(" ".join(chunks))[:max_chars].rstrip()


def extract_visual_description(html: str, max_chars: int = 800) -> tuple[str, str]:
    lines = html_to_lines(html)
    parts = []
    used_sections = []

    for section in TARGET_SECTIONS:
        text = extract_section(lines, section)
        if text:
            used_sections.append(section.title())
            parts.append(text)

    visual = clean_text(" ".join(parts))[:max_chars].rstrip()
    return visual, "; ".join(used_sections)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default="cache/allaboutbirds_pages")
    parser.add_argument("--delay", type=float, default=1.5)
    parser.add_argument("--overwrite-verified", action="store_true")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    with open(args.input, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fields = list(rows[0].keys())

    needed = [
        "expert_description_status", "expert_visual_description", "expert_description_source",
        "source_page_verified_in_this_pass", "expert_description_sections", "description_notes",
        "safe_use_note"
    ]
    for col in needed:
        if col not in fields:
            fields.append(col)
            for row in rows:
                row[col] = ""

    # Fetch each unique source once, then apply to all class labels that map to it.
    url_to_desc: dict[str, tuple[str, str]] = {}
    urls = sorted({r.get("recommended_url", "") for r in rows if r.get("recommended_url", "").endswith("/id")})

    for url in tqdm(urls, desc="Fetching All About Birds pages"):
        safe_name = re.sub(r"[^A-Za-z0-9]+", "_", url).strip("_")[:180]
        html = fetch_html(url, cache_dir / f"{safe_name}.html", args.delay)
        if html:
            url_to_desc[url] = extract_visual_description(html)

    for row in rows:
        label_type = row.get("label_type", "")
        if label_type == "group_or_family_label":
            row["expert_description_status"] = "not_applicable_group_or_family_label"
            continue
        if row.get("expert_description_status") == "found_verified_compact_description_added" and not args.overwrite_verified:
            continue

        url = row.get("recommended_url", "")
        visual, sections = url_to_desc.get(url, ("", ""))
        if visual:
            row["expert_visual_description"] = visual
            row["expert_description_source"] = url
            row["source_page_verified_in_this_pass"] = "yes_local_fetch"
            row["expert_description_sections"] = sections
            row["expert_description_status"] = "found_raw_visual_sections_needs_normalization"
            row["description_notes"] = "Extracted Size & Shape / Color Pattern text; consider normalizing into compact attributes before model release."
            row["safe_use_note"] = "For class project use only; store compact normalized traits + source URL rather than long raw text."
        elif row.get("coverage_status") == "needs_manual_mapping":
            row["expert_description_status"] = "needs_manual_mapping_before_description_fetch"
        elif label_type != "group_or_family_label":
            row["expert_description_status"] = "fetch_failed_or_page_missing"

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
