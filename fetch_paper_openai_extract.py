# fetch_pdf_pipeline.py
import os, re, io, time, urllib.parse, requests, fitz
from pathlib import Path
from typing import Optional, Union, Tuple
import sys
sys.path.append('/resnick/groups/mthomson/yunruilu/Github_repo/spatial-genomics-llm-collection')
import config
import argparse
import pandas as pd
from openai import OpenAI
import json

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

    # 1) Direct OA URL
    if oa_pdf_url:
        print(f"Trying method 1: Direct OA URL for DOI {doi}")
        try:
            r = requests.get(oa_pdf_url, timeout=timeout)
            if _download_ok(r):
                pdf_bytes = r.content
                print(f"Success at method 1: Direct OA URL for DOI {doi}")
            else:
                print(f"Failed method 1: Direct OA URL for DOI {doi}")
        except Exception:
            print(f"Failed method 1: Direct OA URL for DOI {doi}")

    # 2) Unpaywall
    if pdf_bytes is None and config.UNPAYWALL_EMAIL and "your_email" not in config.UNPAYWALL_EMAIL:
        print(f"Trying method 2: Unpaywall for DOI {doi}")
        try:
            upw_url = f"https://api.unpaywall.org/v2/{doi}?email={config.UNPAYWALL_EMAIL}"
            ur = requests.get(upw_url, timeout=timeout)
            if ur.status_code == 200:
                j = ur.json()
                pdf_url = (j.get("best_oa_location") or {}).get("url_for_pdf") or (j.get("best_oa_location") or {}).get("url")
                if not pdf_url:
                    for loc in j.get("oa_locations") or []:
                        pdf_url = loc.get("url_for_pdf") or loc.get("url")
                        if pdf_url: break
                if pdf_url:
                    pr = requests.get(pdf_url, timeout=timeout)
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
            r = requests.get(f"https://doi.org/{doi}", headers={"Accept": "application/pdf"}, timeout=timeout, allow_redirects=True)
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
            sess = session or requests.Session()
            proxied_url = f"{openathens_prefix}{urllib.parse.quote('https://doi.org/' + doi, safe='')}"
            resp = sess.get(proxied_url, allow_redirects=True, timeout=timeout)
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
                        pdf_resp = sess.get(pdf_link, timeout=timeout)
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
client = OpenAI(api_key=config.OPENAI_API_KEY)

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



# def main():
parser = argparse.ArgumentParser(description="Fetch PDF and extract full text by DOI")
parser.add_argument("--paper_index", required=True, help="Index of the paper to fetch")
# parser.add_argument("--oa-pdf-url", help="Direct open access PDF URL (optional)")
parser.add_argument('--print_details', type=str, default='True', help='Whether to print details (True/False)')
parser.add_argument('--model', type=str, default='gpt-5', help='OpenAI model to use')

args = parser.parse_args()
paper_index_input = args.paper_index
paper_index_input = int(paper_index_input)
print_details_input_ = args.print_details
print_details_input = int(print_details_input_) == 1
model_input = args.model

# sess = requests.Session()
# cookie_path = os.path.expanduser(os.getenv("OPENATHENS_COOKIES", "/resnick/groups/mthomson/yunruilu/Github_repo/spatial-genomics-llm-collection/utils/openanthens_cookies.txt"))
# sess.cookies = MozillaCookieJar(cookie_path)
# try:
#     sess.cookies.load(ignore_discard=True, ignore_expires=True)
# except FileNotFoundError:
#     pass
# sess.headers.update({"User-Agent": "Mozilla/5.0"})
# cookie_path = os.path.expanduser("~/.oa_cookies.lwp")
# sess.cookies = LWPCookieJar(cookie_path)
# try:
#     sess.cookies.load(ignore_discard=True, ignore_expires=True)
# except FileNotFoundError:
#     pass
csv_path = "/resnick/groups/mthomson/yunruilu/Github_repo/spatial-genomics-llm-collection/temp/temp.csv"
df = pd.read_csv(csv_path)
# doi_input = df[df['index_paper'] == paper_index_input]['doi']
doi_input = df.loc[df['index_paper'] == paper_index_input, 'doi'].iloc[0]
try:
    pdf_path, full_text = fetch_pdf_and_text_by_doi(doi_input)
    # try:
    #     sess.cookies.save(ignore_discard=True, ignore_expires=True)
    # except Exception as e:
    #     print(f"Warning: failed to save cookies: {e}", file=sys.stderr)
    if pdf_path and full_text:
        print(f"Successfully fetched PDF and extracted text for DOI {doi_input}")
        # print(f"PDF path: {pdf_path}")
        # print(f"Full text: {full_text[:2000]}")
        # Save success status to openanthens.txt
        with open("/resnick/groups/mthomson/yunruilu/Github_repo/spatial-genomics-llm-collection/temp/openanthens.txt", "w") as f:
            f.write("Success")
        # Success - path already printed in function
        # pass
    else:
        with open("/resnick/groups/mthomson/yunruilu/Github_repo/spatial-genomics-llm-collection/temp/openanthens.txt", "w") as f:
            f.write("Failed")
        print("Failed to fetch PDF or extract text")
        sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    with open("/resnick/groups/mthomson/yunruilu/Github_repo/spatial-genomics-llm-collection/temp/openanthens.txt", "w") as f:
            f.write("Failed")
    sys.exit(1)

# if __name__ == "__main__":
# main()

if full_text:
    data_info = extract_datasets_from_text(full_text = full_text)
    # print(type(data_info))
    # print(data_info)
    if data_info:
        if print_details_input:
            print(f"Successfully extracted datasets information")
            for ds in data_info:
                print(json.dumps(ds, indent=2))
        df.loc[df['index_paper'] == paper_index_input, 'Datasets_info'] = json.dumps(data_info)
    else:
        print(f"Failed to extract datasets information for DOI {doi_input}")
        df.loc[df['index_paper'] == paper_index_input, 'Datasets_info'] = None
else:
    print(f"No full text found for DOI {doi_input}")
    df.loc[df['index_paper'] == paper_index_input, 'Datasets_info'] = None

df.to_csv(csv_path, index=False)

print(f"Successfully updated temp.csv with dataset information for paper index {paper_index_input}")

# # Read the main Papers.csv file and append the updated df to it
# papers_csv_path = "/resnick/groups/mthomson/yunruilu/Github_repo/spatial-genomics-llm-collection/Papers.csv"
# try:
#     papers_df = pd.read_csv(papers_csv_path)
#     # Update the corresponding row in papers_df with the new dataset info
#     papers_df.loc[papers_df['index_paper'] == paper_index_input, 'Datasets_info'] = df.loc[df['index_paper'] == paper_index_input, 'Datasets_info'].iloc[0]
#     # Save the updated dataframe back to Papers.csv
#     papers_df.to_csv(papers_csv_path, index=False)
#     print(f"Successfully updated Papers.csv with dataset information for paper index {paper_index_input}")
# except Exception as e:
#     print(f"Error updating Papers.csv: {e}")
