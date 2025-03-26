import pandas as pd
import os
from pathlib import Path

# Function to split the input file into multiple output files with consistent headers
def split_file(input_file: str, output_dir: str, num_files: int = 4) -> None:
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Read the input file - first check if it has headers
    # Try reading first row to see if it looks like a header
    sample = pd.read_csv(input_file, sep='\t', nrows=1)
    has_header = all(isinstance(col, str) for col in sample.columns)
    
    # Read the file with appropriate header setting
    if has_header:
        df = pd.read_csv(input_file, sep='\t')
    else:
        df = pd.read_csv(input_file, sep='\t', header=None)
    
    # Calculate the number of rows in each output file
    num_rows = len(df)
    rows_per_file = num_rows // num_files
    
    # Split the dataframe into chunks and save to output files
    for i in range(num_files):
        start_row = i * rows_per_file
        end_row = (i + 1) * rows_per_file if i < num_files - 1 else num_rows
        output_file = os.path.join(output_dir, f'batch3-{start_row + 1}-{end_row}.tsv')
        
        # Always include header in each output file
        df.iloc[start_row:end_row].to_csv(output_file, sep='\t', index=False)
        print(f"Saved {output_file} with rows {start_row + 1} to {end_row}")
        
    
def main():
    input_file = 'batch3_4001-6000-FULL.tsv'  # Default input file
    output_dir = 'split_files'  # Default output directory
    num_files = 4  # Default number of output files
    
    # Call the split_file function
    split_file(input_file, output_dir, num_files)
    
if __name__ == '__main__':
    main()

    