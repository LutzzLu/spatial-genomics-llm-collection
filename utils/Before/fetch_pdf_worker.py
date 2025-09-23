#!/usr/bin/env python3
"""
Standalone worker to fetch a paper PDF + text given a DOI.

Usage:
  python fetch_pdf_worker.py --doi 10.1038/s41586-019-1049-y \
      --out-dir ./data \
      --unpaywall-email you@example.edu \
      --openathens-prefix "https://go.openathens.net/redirector/yourcampus.edu?url=" \
      --cookie-jar "~/.oa_cookies.lwp"

Behavior:
- Creates its own requests.Session() and cookie jar each run.
- Tries, in order:
    1) Direct OA URL if provided
    2) Unpaywall
    3) DOI content negotiation (Accept: application/pdf)
    4) OpenAthens redirector (detects login pages; will not bypass auth)
- On success, writes:
    data/pdf/<safe_doi>.pdf
    data/txt/<safe_doi>.txt
    data/meta/<safe_doi>.json
- Prints a one-line JSON summary to STDOUT and exits:
    0 = success
    2 = auth required
    1 = other failure
"""

import argparse
import datetime
import io
import json
import os
import re
import sys
import urllib.parse
from pathlib import Path
from typing import Optional, Tuple

import requests
from http.cookiejar import LWPCookieJar

try:
    import fitz  # PyMuPDF
except Exception:
    print("PyMuPDF (fitz) is required. pip install pymupdf", file=sys.stderr)
    sys.exit(1)

USER_AGENT = "Mozilla/5.0 (worker) Python requests for scholarly PDF retrieval"
DEFAULT_OUT_DIR = "./data"


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _now_iso() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def safe_name(s: str) -> str:
    # DOI -> safe filename root
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def _download_ok(resp: requests.Response) -> bool:
    ctype = (resp.headers.get("Content-Type") or "").lower()
    return (
        resp.status_code == 200
        and resp.content
        and ("pdf" in ctype or resp.content[:4] == b"%PDF")
    )


def _extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        return ""
    try:
        parts = []
        for page in doc:
            parts.append(page.get_text())
        return "".join(parts)
    finally:
        doc.close()


def try_direct_oa(session: requests.Session, oa_pdf_url: Optional[str], timeout: int) -> Tuple[Optional[bytes], dict]:
    if not oa_pdf_url:
        return None, {"method": "direct_oa", "detail": "no_url"}
    _stderr(f"Trying method 1: direct OA URL -> {oa_pdf_url}")
    try:
        r = session.get(oa_pdf_url, timeout=timeout, allow_redirects=True)
        if _download_ok(r):
            return r.content, {"method": "direct_oa", "final_url": r.url}
        return None, {"method": "direct_oa", "detail": f"bad_content_type_or_status_{r.status_code}"}
    except Exception as e:
        return None, {"method": "direct_oa", "detail": f"exception_{type(e).__name__}"}


def try_unpaywall(session: requests.Session, doi: str, email: Optional[str], timeout: int) -> Tuple[Optional[bytes], dict]:
    if not email or "@" not in email:
        return None, {"method": "unpaywall", "detail": "no_email"}
    api = f"https://api.unpaywall.org/v2/{doi}?email={email}"
    _stderr(f"Trying method 2: Unpaywall -> {api}")
    try:
        ur = session.get(api, timeout=timeout)
        if ur.status_code != 200:
            return None, {"method": "unpaywall", "detail": f"api_status_{ur.status_code}"}
        j = ur.json()
        pdf_url = (j.get("best_oa_location") or {}).get("url_for_pdf") or (j.get("best_oa_location") or {}).get("url")
        if not pdf_url:
            for loc in j.get("oa_locations") or []:
                pdf_url = loc.get("url_for_pdf") or loc.get("url")
                if pdf_url:
                    break
        if not pdf_url:
            return None, {"method": "unpaywall", "detail": "no_pdf_url"}
        pr = session.get(pdf_url, timeout=timeout, allow_redirects=True)
        if _download_ok(pr):
            return pr.content, {"method": "unpaywall", "final_url": pr.url}
        return None, {"method": "unpaywall", "detail": f"bad_content_type_or_status_{pr.status_code}"}
    except Exception as e:
        return None, {"method": "unpaywall", "detail": f"exception_{type(e).__name__}"}


def try_doi_content_negotiation(session: requests.Session, doi: str, timeout: int) -> Tuple[Optional[bytes], dict]:
    url = f"https://doi.org/{doi}"
    _stderr(f"Trying method 3: DOI content negotiation -> {url}")
    try:
        r = session.get(url, headers={"Accept": "application/pdf"}, timeout=timeout, allow_redirects=True)
        if _download_ok(r):
            return r.content, {"method": "doi_accept_pdf", "final_url": r.url}
        return None, {"method": "doi_accept_pdf", "detail": f"bad_content_type_or_status_{r.status_code}"}
    except Exception as e:
        return None, {"method": "doi_accept_pdf", "detail": f"exception_{type(e).__name__}"}


def try_openathens(session: requests.Session, doi: str, openathens_prefix: Optional[str], timeout: int) -> Tuple[Optional[bytes], dict]:
    if not openathens_prefix:
        return None, {"method": "openathens", "detail": "no_prefix"}
    proxied = f"{openathens_prefix}{urllib.parse.quote('https://doi.org/' + doi, safe='')}"
    _stderr(f"Trying method 4: OpenAthens redirector -> {proxied}")
    try:
        resp = session.get(proxied, allow_redirects=True, timeout=timeout)
        ctype = (resp.headers.get("Content-Type") or "").lower()
        # Direct PDF?
        if ctype.startswith("application/pdf") or resp.content[:4] == b"%PDF":
            return resp.content, {"method": "openathens", "final_url": resp.url, "login_required": False}

        login_required = False
        html = ""
        if "html" in ctype:
            html = resp.text
            probe = (resp.url.lower() + " " + html.lower())
            if any(t in probe for t in ["openathens", "login", "samlrequest", "shibboleth"]):
                login_required = True

        # If not obvious login page, look for a PDF link on the article page
        if not login_required and html:
            m = re.search(r'href="([^"]+\.pdf[^"]*)"', html, re.IGNORECASE)
            if m:
                pdf_link = urllib.parse.urljoin(resp.url, m.group(1))
                pr = session.get(pdf_link, timeout=timeout)
                if (pr.headers.get("Content-Type") or "").lower().startswith("application/pdf") or pr.content[:4] == b"%PDF":
                    return pr.content, {"method": "openathens", "final_url": pr.url, "login_required": False}

        # No dice—likely needs auth
        return None, {"method": "openathens", "final_url": resp.url, "login_required": login_required or "html" in ctype}
    except Exception as e:
        return None, {"method": "openathens", "detail": f"exception_{type(e).__name__}"}


def main():
    ap = argparse.ArgumentParser(description="Fetch PDF+text for a DOI in an isolated worker process.")
    ap.add_argument("--doi", required=True, help="DOI string (e.g., 10.1038/s41586-019-1049-y)")
    ap.add_argument("--oa-pdf-url", default=None, help="Optional direct OA PDF URL if already known")
    ap.add_argument("--unpaywall-email", default=os.getenv("UNPAYWALL_EMAIL"), help="Email for Unpaywall API")
    ap.add_argument("--openathens-prefix", default=os.getenv("OPENATHENS_PREFIX"), help="OpenAthens redirector prefix")
    ap.add_argument("--cookie-jar", default=os.path.expanduser("~/.oa_cookies.lwp"), help="Path to LWPCookieJar file")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Base output directory (default: ./data)")
    ap.add_argument("--timeout", type=int, default=40, help="HTTP timeout seconds (default: 40)")
    ap.add_argument("--force", action="store_true", help="Re-download even if outputs already exist")
    args = ap.parse_args()

    doi = args.doi.strip()
    safe = safe_name(doi)

    # Prepare output dirs
    out_base = Path(args.out_dir)
    pdf_dir = out_base / "pdf"
    txt_dir = out_base / "txt"
    meta_dir = out_base / "meta"
    for d in (pdf_dir, txt_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    pdf_path = pdf_dir / f"{safe}.pdf"
    txt_path = txt_dir / f"{safe}.txt"
    meta_path = meta_dir / f"{safe}.json"

    # If exists and not forcing, short-circuit
    if pdf_path.exists() and txt_path.exists() and not args.force:
        result = {
            "doi": doi,
            "status": "success",
            "method": "cached",
            "pdf_path": str(pdf_path.resolve()),
            "txt_path": str(txt_path.resolve()),
            "timestamp": _now_iso(),
        }
        print(json.dumps(result), flush=True)
        sys.exit(0)

    # Fresh session + cookie jar per run
    session = requests.Session()
    # session.headers.update({"User-Agent": USER_AGENT})
    # # Attach LWPCookieJar so cookies persist across runs (and across different publishers)
    # jar_path = os.path.expanduser(args.cookie_jar)
    # session.cookies = LWPCookieJar(jar_path)
    # try:
    #     session.cookies.load(ignore_discard=True, ignore_expires=True)
    #     _stderr(f"Loaded cookies from {jar_path}")
    # except FileNotFoundError:
    #     _stderr(f"No cookie jar found at {jar_path} (will create)")

    attempted = []
    pdf_bytes = None
    meta = {}

    # 1) Direct OA URL
    if pdf_bytes is None:
        pdf_bytes, info = try_direct_oa(session, args.oa_pdf_url, args.timeout)
        attempted.append(info)

    # 2) Unpaywall
    if pdf_bytes is None:
        b, info = try_unpaywall(session, doi, args.unpaywall_email, args.timeout)
        attempted.append(info)
        if b:
            pdf_bytes = b

    # 3) DOI content negotiation
    if pdf_bytes is None:
        b, info = try_doi_content_negotiation(session, doi, args.timeout)
        attempted.append(info)
        if b:
            pdf_bytes = b

    # 4) OpenAthens
    login_required = False
    if pdf_bytes is None:
        b, info = try_openathens(session, doi, args.openathens_prefix, args.timeout)
        attempted.append(info)
        if b:
            pdf_bytes = b
        else:
            login_required = bool(info.get("login_required", False))

    # Save cookies back (if any new set during redirects)
    try:
        session.cookies.save(ignore_discard=True, ignore_expires=True)
        # _stderr(f"Saved cookies to {jar_path}")
    except Exception:
        pass

    # Prepare final result
    if pdf_bytes:
        # Write PDF
        pdf_path.write_bytes(pdf_bytes)
        # Extract text
        text = _extract_text_from_pdf_bytes(pdf_bytes)
        txt_path.write_text(text, encoding="utf-8")

        result = {
            "doi": doi,
            "status": "success",
            "method": next((a.get("method") for a in attempted if a.get("method") in ("direct_oa", "unpaywall", "doi_accept_pdf", "openathens") and "final_url" in a), "unknown"),
            "pdf_path": str(pdf_path.resolve()),
            "txt_path": str(txt_path.resolve()),
            "bytes": len(pdf_bytes),
            "timestamp": _now_iso(),
            "attempted": attempted,
        }
        meta_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result), flush=True)
        sys.exit(0)
    else:
        status = "auth_required" if login_required else "failed"
        code = 2 if login_required else 1
        result = {
            "doi": doi,
            "status": status,
            "timestamp": _now_iso(),
            "attempted": attempted,
        }
        meta_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result), flush=True)
        sys.exit(code)


if __name__ == "__main__":
    main()
