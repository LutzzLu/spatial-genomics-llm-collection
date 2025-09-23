#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=50G
#SBATCH --output=Paper_Collection_%A.out
#SBATCH -J "Paper_Collection"
#SBATCH --mail-user=yunruilu@caltech.edu
#SBATCH --mail-type=END

source /resnick/groups/mthomson/yunruilu/miniconda3/etc/profile.d/conda.sh
conda activate Paper_Collection

python seach_paper.py --since_days 10 --search_query "spatial transcriptomics"

# Read CSV file and process papers by index
# Read the row count from the text file
row_count_file="/resnick/groups/mthomson/yunruilu/Github_repo/spatial-genomics-llm-collection/temp/tempcsv_row_number.txt"
total_papers=$(cat "$row_count_file")
echo "Found $total_papers papers to process"

# Loop through each paper index and run fetch_paper_full_text.py
for ((paper_index=0; paper_index<total_papers; paper_index++)); do
    echo "Processing paper $((paper_index+1))/$total_papers:"
    
    # Give each paper 3 opportunities
    for ((attempt=1; attempt<=3; attempt++)); do
        echo "Attempt $attempt/3 for paper $((paper_index+1))"
        python fetch_paper_openai_extract.py --paper_index "$paper_index" --print_details 0 --model gpt-5
        
        # Check the result file
        if [ -f "/resnick/groups/mthomson/yunruilu/Github_repo/spatial-genomics-llm-collection/temp/openanthens.txt" ]; then
            result=$(cat /resnick/groups/mthomson/yunruilu/Github_repo/spatial-genomics-llm-collection/temp/openanthens.txt)
            if [ "$result" = "Success" ]; then
                echo "Successfully processed paper $((paper_index+1)) on attempt $attempt"
                break
            else
                echo "Failed attempt $attempt for paper $((paper_index+1))"
                if [ $attempt -lt 3 ]; then
                    echo "Waiting 20 seconds before retry..."
                    sleep 20
                fi
            fi
        else
            echo "Result file not found for paper $((paper_index+1)) on attempt $attempt"
            if [ $attempt -lt 3 ]; then
                echo "Waiting 20 seconds before retry..."
                sleep 20
            fi
        fi
    done
    
    # Final check after all attempts
    if [ -f "/resnick/groups/mthomson/yunruilu/Github_repo/spatial-genomics-llm-collection/temp/openanthens.txt" ]; then
        result=$(cat /resnick/groups/mthomson/yunruilu/Github_repo/spatial-genomics-llm-collection/temp/openanthens.txt)
        if [ "$result" != "Success" ]; then
            echo "All 3 attempts failed for paper $((paper_index+1))"
        fi
    else
        echo "All 3 attempts failed for paper $((paper_index+1)) (no result file)"
    fi
    
    sleep 20
done

echo "Finished processing all papers"

python combine_csv.py



# python main.py