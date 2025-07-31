import os
import glob
import pandas as pd

input_dir = 'resnet_val_data'
output_file = 'concatenated_data.csv'

csv_files = glob.glob(os.path.join(input_dir, '*.csv'))

dfs = []
for file in csv_files:
    df = pd.read_csv(file, skiprows=1, header=None)
    dfs.append(df)

if dfs:
    concatenated_df = pd.concat(dfs, ignore_index=True)
    concatenated_df.to_csv(output_file, index=False, header=False)
else:
    print("No CSV files found in", input_dir)