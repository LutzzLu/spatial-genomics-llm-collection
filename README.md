## spatial-genomics-llm-collection

Automated pipeline to:
- Search recent spatial genomics/transcriptomics papers via Semantic Scholar
- Download PDFs (Open Access, Unpaywall, DOI negotiation, or via OpenAthens redirect)
- Extract full text and use an LLM to parse dataset metadata
- Aggregate results into `Papers.csv` with deduplication

### Repository layout
- `seach_paper.py`: Searches recent papers (bulk API), gathers references, writes `temp/temp.csv` and `temp/tempcsv_row_number.txt`.
- `fetch_paper_openai_extract.py`: For a given index in `temp/temp.csv`, fetches the PDF, extracts text, calls OpenAI to extract dataset info, and writes to `temp/temp.csv` (`Datasets_info` column).
- `combine_csv.py`: Appends `temp/temp.csv` into `Papers.csv` and deduplicates by `paper_id`, prioritizing rows with non-null `Datasets_info`.
- `config.py`: API keys and configuration (Semantic Scholar, OpenAI, Unpaywall, NCBI, model, defaults).
- `main.sh`, `main_test.sh`: Slurm job scripts orchestrating the full pipeline (search → loop through papers with retries → combine).
- `utils/Initiate_database.py`: Creates an empty `Papers.csv` with expected columns.
- `Notebooks/`: Example notebooks (`0_test.ipynb`, `0_1_test.ipynb`, `1_pipeline.ipynb`, `2_pipeline.ipynb`) for interactive exploration.
- `PDF/`: Saved PDFs named by DOI-derived filenames.
- `temp/`: Intermediate CSV and status files (`temp.csv`, `openanthens.txt`, `tempcsv_row_number.txt`).
- `utils/Before/`: Archived earlier versions of scripts.

### Requirements
- Python 3.10+ (Conda recommended)
- Slurm cluster (optional but recommended for batch processing)
- Accounts/keys:
  - Semantic Scholar API key (recommended for higher limits)
  - OpenAI API key
  - Unpaywall contact email
  - Optional NCBI API key

### Installation
```bash
# Create environment
conda create -n Paper_Collection python=3.10 -y
conda activate Paper_Collection

# Install packages
pip install pandas requests tqdm openai PyMuPDF
```

### Configure credentials and defaults
Edit `config.py` and set:
- `SEMANTIC_SCHOLAR_API_KEY = "<your_s2_api_key>"`
- `OPENAI_API_KEY = "<your_openai_api_key>"`
- `UNPAYWALL_EMAIL = "<your_email_for_unpaywall>"`
- `NCBI_EMAIL = "<your_email_for_ncbi>"`
- `NCBI_API_KEY = "<optional_ncbi_key>"  # or None`
- `OPENAI_MODEL = "gpt-4.1"` (or your preferred model)
- Optional search defaults: `SEARCH_QUERY`, `FIELDS_OF_STUDY`, `SEARCH_LIMIT`

Note: Do not commit real keys to version control.

### Initialize the database (first time)
```bash
python utils/Initiate_database.py
```
This creates `Papers.csv` at the repo root.

### Quickstart (Slurm)
1) Adjust `main.sh`:
- Conda init line: `source /path/to/miniconda3/etc/profile.d/conda.sh`
- Environment name: `conda activate Paper_Collection`
- CLI knobs (inside the script): `--since_days`, `--search_query`, and the model used by `fetch_paper_openai_extract.py`.

2) Submit:
```bash
sbatch main.sh
```
- Logs: `Paper_Collection_%A.out`
- Default resources in `main.sh`: 1 node, 10 CPUs, 50 GB RAM, 1 day.

What the job does:
- Runs `seach_paper.py` to produce `temp/temp.csv` and `temp/tempcsv_row_number.txt`
- For each paper index, runs `fetch_paper_openai_extract.py` up to 3 attempts, waiting 20s between retries. Success/failure recorded in `temp/openanthens.txt`.
- After the loop, runs `combine_csv.py` to update `Papers.csv` with deduplication.

### Manual run (no Slurm)
1) Search papers:
```bash
python seach_paper.py --since_days 10 --search_query "spatial transcriptomics"
```
Outputs:
- `temp/temp.csv`: rows with `paper_id`, `doi`, `title`, `abstract`, `oa_pdf_url`, `reference`, etc.
- `temp/tempcsv_row_number.txt`: number of rows to process.

2) Process a specific paper by index (0-based):
```bash
python fetch_paper_openai_extract.py --paper_index 0 --print_details 1 --model gpt-5
```
- Attempts multiple PDF fetch strategies and extracts full text with PyMuPDF.
- Calls OpenAI to generate dataset metadata JSON.
- Writes JSON string to `Datasets_info` column in `temp/temp.csv`.
- Sets `temp/openanthens.txt` to `Success` or `Failed`.

3) Combine into master CSV:
```bash
python combine_csv.py
```
- Merges `temp/temp.csv` into `Papers.csv`, deduping by `paper_id`, keeping entries with non-null `Datasets_info` first.

### Outputs
- `Papers.csv`: Master table of all processed papers.
  - Key columns: `title, year, venue, paper_id, doi, publication_date, oa_pdf_url, abstract, reference, Datasets_info`
  - `Datasets_info` is a JSON string of a list of objects with keys:
    - `data link, repository, accession, platform, species, tissue, raw_data_available, available, original_data, description`
- `PDF/`: Saved PDFs named from DOI (e.g., `10.1038_s41592-025-02773-5.pdf`).
- `temp/temp.csv`: Latest search batch with `Datasets_info` as it’s filled.
- `temp/openanthens.txt`: Last status (`Success`/`Failed`) for fetch/extract.
- `Paper_Collection_%A.out`: Slurm job logs.

### Customization
- Change search window and query:
  - CLI: `--since_days <int>`, `--search_query "..."` (supports multi-word strings)
  - Edit `main.sh` to change defaults for scheduled runs.
- Model selection:
  - CLI: `--model <openai_model>` (e.g., `gpt-4.1`, `gpt-4o`, etc.)
  - Default in `config.py` via `OPENAI_MODEL`.
- Institutional access:
  - Pipeline attempts OA and Unpaywall first, then DOI negotiation, then OpenAthens redirect.
  - Ensure your network/session has appropriate institutional access if relying on OpenAthens.

### Notes and limitations
- Absolute paths: Some scripts use absolute paths under this repository folder. If you move the repo, update paths in:
  - `seach_paper.py`, `fetch_paper_openai_extract.py`, `combine_csv.py`, `utils/Initiate_database.py`, `main.sh`, `main_test.sh`.
- Column name caveat:
  - `utils/Initiate_database.py` creates `datasets_info` (lowercase).
  - The pipeline writes to `Datasets_info` (uppercase D). After combining, `Papers.csv` may include both. Prefer using `Datasets_info`.
- API limits: The Semantic Scholar bulk API uses basic backoff; still subject to rate limits.
- PDF/text quality: Some PDFs may yield poor text extraction; dataset parsing depends on text quality and paper structure.
- Compliance: Respect publisher terms and institutional access policies.

### Examples
- Search last 30 days for Xenium:
```bash
python seach_paper.py --since_days 30 --search_query "Xenium"
```
- Process first 5 papers with visible details:
```bash
for i in 0 1 2 3 4; do
  python fetch_paper_openai_extract.py --paper_index "$i" --print_details 1 --model gpt-4.1
done
python combine_csv.py
```