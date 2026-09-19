import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(input_file="data.csv"):
    """
    Randomly split data.csv into 4 files:
    - external_test.csv  : 20% of total data
    - train1.csv         : 60% of total data
    - test1.csv          : 20% of total data (remaining after first two splits)
    - train_test.csv     : union of train1.csv and test1.csv (i.e., 80% of total data)
    """

    # Load the original dataset
    df = pd.read_csv(input_file)

    # =========================================================
    # Step 1: Split off external_test (20% of total)
    #         Remaining 80% goes into a temporary pool
    # =========================================================
    # test_size=0.2 means 20% of data goes to external_test
    df_temp, external_test = train_test_split(
        df, test_size=0.2, random_state=42
    )

    # =========================================================
    # Step 2: Split the remaining 80% into train1 (60%) and test1 (20%)
    #         From the 80% pool, we need 60/80 = 75% for train1
    #         and 20/80 = 25% for test1
    # =========================================================
    # test_size=0.25 of the 80% pool gives 0.25 * 80% = 20% of total
    train1, test1 = train_test_split(
        df_temp, test_size=0.25, random_state=42
    )

    # =========================================================
    # Step 3: Create train_test by combining train1 and test1
    #         This is equivalent to the remaining 80% (df_temp)
    # =========================================================
    train_test = pd.concat([train1, test1], axis=0)

    # =========================================================
    # Step 4: Save all 4 files to CSV
    # =========================================================
    external_test.to_csv("external_test.csv", index=False)
    train1.to_csv("train1.csv", index=False)
    test1.to_csv("test1.csv", index=False)
    train_test.to_csv("train_test.csv", index=False)

    # Print summary statistics for verification
    total = len(df)
    print(f"Total samples:          {total}  (100%)")
    print(f"external_test.csv:      {len(external_test)}  ({len(external_test)/total*100:.1f}%)")
    print(f"train1.csv:             {len(train1)}  ({len(train1)/total*100:.1f}%)")
    print(f"test1.csv:              {len(test1)}  ({len(test1)/total*100:.1f}%)")
    print(f"train_test.csv:         {len(train_test)}  ({len(train_test)/total*100:.1f}%)")


if __name__ == "__main__":
    split_dataset()