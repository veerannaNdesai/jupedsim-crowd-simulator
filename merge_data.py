import pandas as pd
import glob
import os

def merge_master_csvs():
    input_dir = 'master_data'
    output_file = os.path.join(input_dir, 'master_combined_dataset.csv')
    
    # Get all master CSV files in the directory
    csv_files = [os.path.join(input_dir, f) for f in ['master_mall.csv', 'master_airport.csv', 'master_stadium.csv']]
    
    # Check if files exist
    valid_files = [f for f in csv_files if os.path.exists(f)]
    
    if not valid_files:
        print("No master CSV files found to merge.")
        return
        
    print(f"Merging {len(valid_files)} files: {[os.path.basename(f) for f in valid_files]}")
    
    # Merge them
    dataframes = []
    for file in valid_files:
        df = pd.read_csv(file)
        # Ensure venue_type is present as it will distinguish the rows globally
        if 'venue_type' not in df.columns:
            venue_type = os.path.basename(file).split('_')[1].split('.')[0] # Extracts 'mall', 'airport', 'stadium'
            df['venue_type'] = venue_type
            
        dataframes.append(df)
        
    merged_df = pd.concat(dataframes, ignore_index=True)
    
    # Save the combined dataset
    merged_df.to_csv(output_file, index=False)
    
    print(f"Merge successful! Total shape: {merged_df.shape}")
    print(f"File saved to: {output_file}")

if __name__ == "__main__":
    merge_master_csvs()
