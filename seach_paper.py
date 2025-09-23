# main.py
import argparse
import os
import config

# Parse command line arguments
parser = argparse.ArgumentParser(description='Search and analyze spatial genomics papers')
parser.add_argument('--since_days', type=int, default=10, help='Number of days to search back')
parser.add_argument('--print_details', type=str, default='True', help='Whether to print details (True/False)')
parser.add_argument('--model', type=str, default='gpt-5', help='OpenAI model to use')
parser.add_argument('--search_query', type=str, default='spatial transcriptomics', help='Search query', nargs='+')

args = parser.parse_args()

# Convert string to boolean for print_details
# print_details_input = args.print_details.lower() in ['true', '1', 'yes', 'on']
since_days_input = args.since_days
# model_input = args.model
search_query_input = ' '.join(args.search_query)
# print_details_input = True
# since_days_input = 10
# model_input = 'gpt-5'
# search_query_input = 'spatial transcriptomics'

# Search query and settings
# SEARCH_QUERY = '("spatial transcriptomics" OR Visium OR MERFISH OR seqFISH OR CosMX OR Xenium)'
SEARCH_QUERY = search_query_input
# FIELDS_OF_STUDY = "Biology"  # Restrict search to biology-related papers
FIELDS_OF_STUDY = None

# search_papers_bulk.py
import requests, datetime, time
from tqdm import tqdm
# fetch_pdf_pipeline.py
import re, io, time, urllib.parse, requests, fitz
from pathlib import Path
from typing import Optional, Union, Tuple
import re
import json
import fitz  # PyMuPDF for PDF parsing
from openai import OpenAI
import sys
import pandas as pd

sess = requests.Session()


def _get_with_backoff(url, params, headers, max_retries=5, timeout=30, session=None):
    delay = 1.0
    for _ in range(max_retries):
        s = session or requests.Session()
        r = s.get(url, params=params, headers=headers, timeout=timeout)
        if r.status_code in (429,) or 500 <= r.status_code < 600:
            time.sleep(delay); delay = min(delay * 2, 30); continue
        return r
    return r

def search_new_papers_bulk(since_days=365):
    since_date = datetime.date.today() - datetime.timedelta(days=since_days)
    url = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"

    headers = {}
    if config.SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = config.SEMANTIC_SCHOLAR_API_KEY

    params = {
        "query": SEARCH_QUERY,
        # "query": 'spatial transcriptomics',                       # simpler query string
        "fields": "title,year,venue,paperId,externalIds,openAccessPdf,publicationDate,abstract",
        "publicationDateOrYear": f"{since_date.isoformat()}:{datetime.date.today().isoformat()}",
        "sort": "publicationDate:desc",
        "limit": 1000,
        "fieldsOfStudy": FIELDS_OF_STUDY,
    }

    results, token = [], None
    while True:
        p = params.copy()
        if token: p["token"] = token
        resp = _get_with_backoff(url, p, headers, session=sess)
        if resp.status_code != 200:
            print("API", resp.status_code, resp.text[:500]); break

        data = resp.json()
        for paper in data.get("data", []):
            pub = paper.get("publicationDate")
            doi = (paper.get("externalIds") or {}).get("DOI")
            oa  = (paper.get("openAccessPdf") or {}).get("url")
            abstract = paper.get("abstract")
            # tldr = paper.get("tldr")
            if doi and type(doi) == str and len(doi) > 0:
                results.append({
                    "title": paper.get("title",""),
                    "year": paper.get("year"),
                    "venue": paper.get("venue",""),
                    "paper_id": paper.get("paperId"),
                    "doi": doi,
                    "publication_date": pub,
                    "oa_pdf_url": oa,
                    "abstract": abstract,
                    # "tldr": tldr,
                })
        token = data.get("token")
        if not token: break
    return results

def get_references(paper_id, fields="citedPaper.paperId,citedPaper.externalIds", max_per_page=1000):
    """
    Robustly fetch references for a paper, returning a list of citedPaper dicts.
    Handles 'data': null, pagination via 'next', and non-200 responses.
    """
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references"
    headers = {"x-api-key": config.SEMANTIC_SCHOLAR_API_KEY} if config.SEMANTIC_SCHOLAR_API_KEY else {}
    params = {"fields": fields, "limit": max_per_page, "offset": 0}
    out = []
    while True:
        r = _get_with_backoff(url, params, headers, session=sess)
        if r.status_code != 200:
            # Surface API response text to help debugging auth/rate-limit/etc.
            raise RuntimeError(f"S2 references error {r.status_code}: {r.text[:300]}")
        try:
            data = r.json()
        except ValueError:
            raise RuntimeError("S2 references returned non-JSON response")

        items = data.get("data") or []  # <- key fix: 'null' -> []
        # Each item normally has {'citedPaper': {...}}; fall back defensively.
        for row in items:
            cp = (row or {}).get("citedPaper") or row or {}
            out.append(cp)

        nxt = data.get("next")
        if nxt is None:
            break
        params["offset"] = nxt
    return out

def get_cited_papers(paper_id: str) -> list[dict]:
    """
    Return a list of {"paper_id": <str>, "doi": <str|None>} for all papers
    that the given paper_id cites.
    """
    refs = get_references(
        paper_id,
        fields="citedPaper.paperId,citedPaper.externalIds",
        max_per_page=1000
    )

    out, seen = [], set()
    for cp in refs:  # cp is the cited paper dict
        pid = cp.get("paperId")
        doi = (cp.get("externalIds") or {}).get("DOI")
        key = (pid, (doi or "").lower())
        if pid and key not in seen:
            out.append({"paper_id": pid, "doi": (doi.lower() if doi else None)})
            seen.add(key)
    return out

result = search_new_papers_bulk(since_days = since_days_input)
print(f'Found {len(result)} new papers in the last {since_days_input} days since today: {datetime.date.today()}')

for i, one_paper in enumerate(result):
    # print('\n--------------------------------------------------------------------')
    # print(f"Processing paper {i+1} of {len(result)}, doi: {one_paper['doi']}")

    # sess = requests.Session()

    items = get_cited_papers(one_paper['paper_id'])
    time.sleep(5)
    result[i]['reference'] = items

for paper in result:
    paper['datasets_info'] = None
result_df = pd.DataFrame(result)

result_df['index_paper'] = [str(i) for i in range(len(result_df))]

result_df = result_df.drop_duplicates(subset=['paper_id'], keep='first')
csv_path = "/resnick/groups/mthomson/yunruilu/Github_repo/spatial-genomics-llm-collection/temp/temp.csv"
result_df.to_csv(csv_path, index=False)

# Store the number of rows in result_df as a text file
temp_dir = "/resnick/groups/mthomson/yunruilu/Github_repo/spatial-genomics-llm-collection/temp"
row_count_file = os.path.join(temp_dir, "tempcsv_row_number.txt")

with open(row_count_file, "w") as f:
    f.write(str(len(result_df)))

print(f"Saved row count ({len(result_df)}) to {row_count_file}")



# csv_path = "/resnick/groups/mthomson/yunruilu/Github_repo/spatial-genomics-llm-collection/temp.csv"
# # df = pd.read_csv(csv_path)
# new_papers_df = pd.DataFrame(result)
# new_papers_df['']
# df_updated = pd.concat([df, new_papers_df], ignore_index=True)
# df_updated = df_updated.drop_duplicates(subset=['paper_id'], keep='first')

# df_updated.to_csv(csv_path, index=False)

# print(f"Added {len(new_papers_df)} new papers to the CSV file")
# print(f"Total papers in CSV after deduplication: {len(df_updated)}")
# # print(display(df_updated.head()))