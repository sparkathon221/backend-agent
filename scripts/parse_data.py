import os
import pandas as pd

def merge_data():
    folder_path = "./amazon-products-dataset"
    merged_df = pd.DataFrame()

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            try:
                df = pd.read_csv(file_path)
                if df.shape[0] > 0:
                    sample_df = df.head(100)
                    merged_df = pd.concat([merged_df, sample_df], ignore_index=True)
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    merged_df = merged_df.drop(columns=["Unnamed: 0"])

    merged_df.to_csv("merged_dataset.csv", index=False)
    print(f"Saved merged dataset with {merged_df.shape[0]} rows.")

import pandas as pd

def add_sequential_product_id(
    csv_in: str = "merged_dataset.csv",
    csv_out: str = "merged_dataset_with_id.csv"
) -> pd.DataFrame:
    df = pd.read_csv(csv_in)
    total = len(df)
    pad_len = len(str(total))

    df.insert(0, "product_id", ["P" + str(i).zfill(pad_len) for i in range(1, total + 1)])
    df.to_csv(csv_out, index=False)
    return df

if __name__ == "__main__":
    merge_data()
    add_sequential_product_id()