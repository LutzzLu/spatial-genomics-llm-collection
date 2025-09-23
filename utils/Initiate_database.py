import pandas as pd

# Create an empty DataFrame with the specified columns
columns = [
    'title',
    'year', 
    'venue',
    'paper_id',
    'doi',
    'publication_date',
    'oa_pdf_url',
    'abstract',
    'tldr',
    'reference',
    'datasets_info',
]

df = pd.DataFrame(columns=columns)

# Save to the specified path
output_path = '/resnick/groups/mthomson/yunruilu/Github_repo/spatial-genomics-llm-collection/Papers.csv'
df.to_csv(output_path, index=False)

print(f"Created empty DataFrame with columns: {list(df.columns)}")
print(f"Saved to: {output_path}")

