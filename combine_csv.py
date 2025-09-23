import pandas as pd

csv_path = "/resnick/groups/mthomson/yunruilu/Github_repo/spatial-genomics-llm-collection/temp/temp.csv"
temp_df = pd.read_csv(csv_path)
paper_df = pd.read_csv("/resnick/groups/mthomson/yunruilu/Github_repo/spatial-genomics-llm-collection/Papers.csv")

# Drop the index_paper column from temp_df
temp_df = temp_df.drop(columns=['index_paper'], errors='ignore')

# Append temp_df to paper_df, ensuring columns match
paper_df = pd.concat([paper_df, temp_df], ignore_index=True, sort=False)

# # Display the combined dataframe
# print("Combined dataframe:")
# display(paper_df.tail())
# Remove duplicates based on paper_id, keeping the first row where Datasets_info is not None
# First, sort by paper_id and put non-null Datasets_info rows first
paper_df_sorted = paper_df.sort_values(
    by=['paper_id', 'Datasets_info'], 
    key=lambda x: x.isnull() if x.name == 'Datasets_info' else x,
    na_position='last'
)

# Drop duplicates keeping the first occurrence (which will have non-null Datasets_info if available)
paper_df = paper_df_sorted.drop_duplicates(subset=['paper_id'], keep='first')

# Reset index
paper_df = paper_df.reset_index(drop=True)

# print("After removing duplicates:")
# display(paper_df.tail())

paper_df.to_csv("/resnick/groups/mthomson/yunruilu/Github_repo/spatial-genomics-llm-collection/Papers.csv", index=False)
print('Papers.csv updated')
