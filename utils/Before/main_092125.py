import argparse
import os

# Parse command line arguments
parser = argparse.ArgumentParser(description='Search and analyze spatial genomics papers')
parser.add_argument('--since_days', type=int, default=10, help='Number of days to search back')
parser.add_argument('--print_details', type=str, default='True', help='Whether to print details (True/False)')
parser.add_argument('--model', type=str, default='gpt-5', help='OpenAI model to use')
parser.add_argument('--search_query', type=str, default='spatial transcriptomics', help='Search query', nargs='+')

args = parser.parse_args()

# Convert string to boolean for print_details
print_details_input = args.print_details.lower() in ['true', '1', 'yes', 'on']
since_days_input = args.since_days
model_input = args.model
search_query_input = ' '.join(args.search_query)

SEMANTIC_SCHOLAR_API_KEY = "19tQFoyv7w5xBQNMsUA7C5lwNqEni5g3GKkP8Pkj"  # e.g., 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx' | https://www.semanticscholar.org/product/api/tutorial?utm_campaign=API%20transaction&utm_medium=email&_hsenc=p2ANqtz--KbD5dVfVRom22kVjKkL-55Ikb73h1Nze5JYW6_8OGfj15Pf_Z7OjRXzHnO2BntuA89mE6jdPyHOEzQnaYDLInKFPGxw&_hsmi=329822401&utm_content=329822401&utm_source=hs_automation
OPENAI_API_KEY = "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"  # e.g., 'sk-...'
UNPAYWALL_EMAIL = "yunruilu@caltech.edu"  # Email for Unpaywall API
NCBI_EMAIL = "yunruilu@caltech.edu"  # Email for NCBI Entrez (required by NCBI)
NCBI_API_KEY = '903b8602ced7c96ae73650c3ff78350a100'  # Optional: NCBI API Key for higher rate limits, or None | https://support.nlm.nih.gov/kbArticle/?pn=KA-05317
USE_OPENATHENS = bool(int(os.getenv("USE_OPENATHENS", "1")))
OPENATHENS_PREFIX = "https://go.openathens.net/redirector/caltech.edu?url="


# Model and other options
# OPENAI_MODEL = "gpt-5"  # Use 'gpt-4' for best results; 'gpt-3.5-turbo' if lower cost is desired
INSTITUTIONAL_ACCESS = True  # True if running on a network with institutional access to paywalled PDFs

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

from http.cookiejar import LWPCookieJar

sess = requests.Session()
sess.headers.update({"User-Agent": "Mozilla/5.0"})
cookie_path = os.path.expanduser("~/.oa_cookies.lwp")
sess.cookies = LWPCookieJar(cookie_path)
try: sess.cookies.load(ignore_discard=True, ignore_expires=True)
except FileNotFoundError: pass

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
    if SEMANTIC_SCHOLAR_API_KEY:
        headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY

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
    headers = {"x-api-key": SEMANTIC_SCHOLAR_API_KEY} if SEMANTIC_SCHOLAR_API_KEY else {}
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

# ---------- Small helpers ----------

def _safe_filename_from_doi(doi: str) -> str:
    # Make a safe filename from the DOI
    # e.g., 10.1038/s41586-019-1049-y -> 10.1038_s41586-019-1049-y.pdf
    return re.sub(r'[^A-Za-z0-9._-]+', '_', doi) + ".pdf"

def _extract_text_from_pdf_bytes(pdf_bytes: bytes) -> Optional[str]:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        return None
    try:
        parts = []
        for page in doc:
            parts.append(page.get_text())  # plain text extraction
        return "".join(parts)
    finally:
        doc.close()

def _download_ok(resp: requests.Response) -> bool:
    ctype = (resp.headers.get("Content-Type") or "").lower()
    return resp.status_code == 200 and (resp.content and ("pdf" in ctype or resp.content[:4] == b"%PDF"))

# ---------- Main function ----------

def fetch_pdf_and_text_by_doi(
    doi: str,
    save_dir: str = "/resnick/groups/mthomson/yunruilu/Github_repo/spatial-genomics-llm-collection/PDF",
    oa_pdf_url: Optional[str] = None,
    openathens_prefix: str = "https://go.openathens.net/redirector/caltech.edu?url=",
    session: Optional[requests.Session] = None,
    timeout: int = 30,
) -> Tuple[Optional[str], Optional[str]]:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    pdf_bytes = None
    s = session or requests.Session()

    # 1) Direct OA URL
    if oa_pdf_url:
        print(f"Trying method 1: Direct OA URL for DOI {doi}")
        try:
            r = s.get(oa_pdf_url, timeout=timeout)
            if _download_ok(r):
                pdf_bytes = r.content
                print(f"Success at method 1: Direct OA URL for DOI {doi}")
            else:
                print(f"Failed method 1: Direct OA URL for DOI {doi}")
        except Exception:
            print(f"Failed method 1: Direct OA URL for DOI {doi}")

    # 2) Unpaywall
    if pdf_bytes is None and UNPAYWALL_EMAIL and "your_email" not in UNPAYWALL_EMAIL:
        print(f"Trying method 2: Unpaywall for DOI {doi}")
        try:
            upw_url = f"https://api.unpaywall.org/v2/{doi}?email={UNPAYWALL_EMAIL}"
            ur = s.get(upw_url, timeout=timeout)
            if ur.status_code == 200:
                j = ur.json()
                pdf_url = (j.get("best_oa_location") or {}).get("url_for_pdf") or (j.get("best_oa_location") or {}).get("url")
                if not pdf_url:
                    for loc in j.get("oa_locations") or []:
                        pdf_url = loc.get("url_for_pdf") or loc.get("url")
                        if pdf_url: break
                if pdf_url:
                    pr = s.get(pdf_url, timeout=timeout)
                    if _download_ok(pr):
                        pdf_bytes = pr.content
                        print(f"Success at method 2: Unpaywall for DOI {doi}")
                    else:
                        print(f"Failed method 2: Unpaywall for DOI {doi}")
                else:
                    print(f"Failed method 2: Unpaywall for DOI {doi} (no PDF URL found)")
            else:
                print(f"Failed method 2: Unpaywall for DOI {doi} (API error)")
        except Exception:
            print(f"Failed method 2: Unpaywall for DOI {doi}")

    # 3) DOI content negotiation
    if pdf_bytes is None:
        print(f"Trying method 3: DOI content negotiation for DOI {doi}")
        try:
            r = s.get(f"https://doi.org/{doi}", headers={"Accept": "application/pdf"}, timeout=timeout, allow_redirects=True)
            if _download_ok(r):
                pdf_bytes = r.content
                print(f"Success at method 3: DOI content negotiation for DOI {doi}")
            else:
                print(f"Failed method 3: DOI content negotiation for DOI {doi}")
        except Exception:
            print(f"Failed method 3: DOI content negotiation for DOI {doi}")

    # 4) OpenAthens redirector
    if pdf_bytes is None and openathens_prefix:
        print(f"Trying method 4: OpenAthens redirector for DOI {doi}")
        try:
            # s = session # or requests.Session()
            proxied_url = f"{openathens_prefix}{urllib.parse.quote('https://doi.org/' + doi, safe='')}"
            resp = s.get(proxied_url, allow_redirects=True, timeout=timeout)
            ctype = resp.headers.get("Content-Type", "")

            if ctype.startswith("application/pdf"):
                # Got the PDF directly
                pdf_bytes = resp.content
                print(f"Success at method 4: OpenAthens redirector for DOI {doi}")
            elif "html" in ctype:
                html = resp.text
                # Check if this is a login page or the article page by looking for clues
                if "openathens.net" in html.lower() or "login" in resp.url:
                    print(f"Failed method 4: OpenAthens redirector for DOI {doi} (authentication required)")
                else:
                    # Assume this is the article page HTML, try to find a PDF link
                    match = re.search(r'href="([^"]+\.pdf[^"]*)"', html)
                    if match:
                        pdf_link = match.group(1)
                        # Complete relative link if needed
                        if pdf_link.startswith("/"):
                            from urllib.parse import urljoin
                            pdf_link = urljoin(resp.url, pdf_link)
                        pdf_resp = s.get(pdf_link, timeout=timeout)
                        if pdf_resp.headers.get("Content-Type","").startswith("application/pdf"):
                            pdf_bytes = pdf_resp.content
                            print(f"Success at method 4: OpenAthens redirector for DOI {doi}")
                        else:
                            print(f"Failed method 4: OpenAthens redirector for DOI {doi} (PDF link didn't return PDF)")
                    else:
                        print(f"Failed method 4: OpenAthens redirector for DOI {doi} (no PDF link found on page)")
            else:
                print(f"Failed method 4: OpenAthens redirector for DOI {doi} (unexpected content type)")
        except Exception:
            print(f"Failed method 4: OpenAthens redirector for DOI {doi}")

    if pdf_bytes is None:
        print(f"All methods failed for DOI {doi}")
        return (None, None)

    # Save + extract
    pdf_name = _safe_filename_from_doi(doi)
    dest = os.path.join(save_dir, pdf_name)
    with open(dest, "wb") as f:
        f.write(pdf_bytes)
    text = _extract_text_from_pdf_bytes(pdf_bytes) or ""
    print(f"Successfully saved PDF and extracted text for DOI {doi}")
    return (dest, text)

# Initialize OpenAI client with API key
client = OpenAI(api_key=OPENAI_API_KEY)

def extract_datasets_from_text(full_text: str = None, pdf_path: str = None, paper_title: str = None):
    """
    Extract detailed dataset information from a research paper given its text or PDF file path.
    
    Either `full_text` or `pdf_path` must be provided. If both are provided, `pdf_path` is prioritized.
    
    Returns:
        A list of dictionaries, each containing details about a dataset used in the paper:
        [
            {
                "data link": str,        # Direct URL or DOI link to the dataset if available
                "repository": str,       # Repository name (e.g., GEO, SRA, Zenodo) or 'Not available'
                "accession": str,        # Accession ID or DOI of the dataset (if applicable)
                "platform": str,         # Technology platform (e.g., Visium, Xenium, MERFISH, scRNA-seq, CODEX, etc.)
                "species": str,          # Organism species (if mentioned)
                "tissue": str,           # Tissue or sample type (if mentioned)
                "raw_data_available": bool, # True if raw data files are available, False otherwise
                "available": bool,       # True if the dataset is publicly available, False if restricted/not available
                "description": str       # Description of the dataset, including platform resolution and origin (generated by this study or from another source)
            },
            ...
        ]
    """
    # Validate input
    if pdf_path is None and full_text is None:
        print("Error: No input text or PDF path provided.")
        return []
    
    # Extract text from PDF if a path is provided
    text_content = ""
    if pdf_path is not None:
        try:
            # Open the PDF and extract all text
            doc = fitz.open(pdf_path)
            for page in doc:
                text_content += page.get_text()
            doc.close()
        except Exception as e:
            print(f"Failed to read PDF file {pdf_path}: {e}")
            return []
    else:
        text_content = full_text
    
    if not text_content:
        # If text extraction failed or resulted in empty content
        print("Error: No text content could be extracted from the input.")
        return []
    
    # Normalize whitespace and remove hyphenation line breaks for better parsing
    text_clean = text_content.replace("-\n", "").replace("\n", " ")
    # Exclude references section to avoid confusion with data DOIs or accessions in references
    text_upper = text_clean.upper()
    if "REFERENCES" in text_upper:
        text_body = text_clean[: text_upper.index("REFERENCES")]
    else:
        text_body = text_clean
    
    # Prepare the system and user messages for the GPT model
    system_msg = (
        "You are an expert assistant extracting dataset information from scientific papers. "
        "Identify all datasets mentioned in the paper and extract relevant details. "
        "Include the repository (or source) and accession/ID or DOI for each dataset, the data platform/technology used "
        "(e.g., 10x Genomics Visium, 10x Xenium, NanoString CosMX, MERFISH, seqFISH, CODEX, or single-cell RNA-seq if applicable), "
        "the species and tissue, whether raw data is available, whether the dataset is publicly available, and a brief description. "
        "whether the authors generated the data or reused the data from another source, and a brief description. "
        "In the description, mention the platform and its resolution (for example, if it's spatial transcriptomics with spot-based or single-cell resolution, or if it's non-spatial single-cell RNA-seq), "
        "and state whether the dataset was generated in this study or obtained from another source (citing the source or reference if mentioned)."
    )
    user_msg = (
        "Extract all datasets (particularly spatially-resolved omics datasets) mentioned in the following text. "
        "If the paper includes a single-cell RNA-seq dataset (which is non-spatial) for analysis, include it as well and denote it appropriately. "
        "Return ONLY a valid JSON array of objects, where each object has the keys: "
        "data link, repository, accession, platform, species, tissue, raw_data_available, available, original_data, description. "
        "If a dataset is not publicly available (e.g., available upon request or not provided), set repository to \"Not available\" and available to false. "
        "Provide no extra commentary or explanation, only the JSON.\n\n"
        f"Text:\n\"\"\"\n{text_body}\n\"\"\""
    )
    
    # Call the OpenAI API to get the dataset details in JSON format
    try:
        response = client.chat.completions.create(
            model=model_input,
            # model="gpt-4.1",
            # model=OPENAI_MODEL,  # e.g., 'gpt-4' or 'gpt-3.5-turbo'
            messages=[{"role": "system", "content": system_msg},
                      {"role": "user",  "content": user_msg}],
            # temperature=0
        )
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return []
    
    # The model's answer (should be JSON or contain JSON)
    content = response.choices[0].message.content.strip()
    
    # Helper to parse the JSON from the model's response
    def _extract_json(text: str):
        # Attempt direct JSON parse
        try:
            return json.loads(text), None
        except Exception:
            pass
        # Check for JSON in a markdown code block
        match = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
        if match:
            try:
                return json.loads(match.group(1)), None
            except Exception as e:
                last_err = e
        else:
            last_err = None
        # Fallback: find first JSON object/array in the text
        match = re.search(r"(\{.*?\}|\[.*?\])", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1)), None
            except Exception as e:
                last_err = e
        return None, last_err
    
    # Parse the JSON content from the model's output
    datasets, err = _extract_json(content)
    if err or datasets is None:
        print(f"Failed to parse JSON from model output: {err or 'No JSON found'}")
        return []
    
    # Ensure the result is a list of dicts
    if isinstance(datasets, dict):
        datasets = [datasets]
    
    return datasets



result = search_new_papers_bulk(since_days = since_days_input)
print(f'Found {len(result)} new papers in the last {since_days_input} days since today: {datetime.date.today()}')

for i, one_paper in tqdm(enumerate(result)):
    print('\n--------------------------------')
    print(f"Processing paper {i+1} of {len(result)}")

    # sess = requests.Session()

    items = get_cited_papers(one_paper['paper_id'])
    result[i]['reference'] = items
    # sess = requests.Session()
    path, full_text = fetch_pdf_and_text_by_doi(doi = one_paper['doi'], 
                                                session = sess,
                                                )
    # sess.close()
    if full_text:
        data_info = extract_datasets_from_text(full_text = full_text)
        if data_info:
            if print_details_input:
                print(f"Successfully extracted datasets information")
                for ds in data_info:
                    print(json.dumps(ds, indent=2))
            result[i]['Datasets_info'] = data_info
        else:
            result[i]['Datasets_info'] = None
    else:
        result[i]['Datasets_info'] = None

csv_path = "/resnick/groups/mthomson/yunruilu/Github_repo/spatial-genomics-llm-collection/Papers.csv"
df = pd.read_csv(csv_path)
new_papers_df = pd.DataFrame(result)
df_updated = pd.concat([df, new_papers_df], ignore_index=True)
df_updated = df_updated.drop_duplicates(subset=['paper_id'], keep='first')

df_updated.to_csv(csv_path, index=False)

print(f"Added {len(new_papers_df)} new papers to the CSV file")
print(f"Total papers in CSV after deduplication: {len(df_updated)}")
# print(display(df_updated.head()))