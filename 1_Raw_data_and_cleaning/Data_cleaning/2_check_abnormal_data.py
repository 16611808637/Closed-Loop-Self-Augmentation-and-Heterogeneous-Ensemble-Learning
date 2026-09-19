import pandas as pd

def preprocess_data(df):
    df_compare = df.copy()
    for col in range(15):
        df_compare[col] = pd.to_numeric(df_compare[col], errors='coerce')
    return df_compare

def find_groups_to_remove(df_compare, original_df):
    all_remove_indexes = set()
    group_log = []
    total_rows = len(df_compare)
    used_row = set()

    for i in range(total_rows):
        if i in used_row:
            continue
        group_members = [i]
        for j in range(i + 1, total_rows):
            if j in used_row:
                continue
            col_diff = abs(df_compare.iloc[i, 0:15] - df_compare.iloc[j, 0:15])
            if all(col_diff <= 2):
                col17_same = (original_df.iloc[i, 16] == original_df.iloc[j, 16])
                col16_diff = (original_df.iloc[i, 15] != original_df.iloc[j, 15])
                if col17_same and col16_diff:
                    group_members.append(j)
        if len(group_members) > 1:
            for idx in group_members:
                all_remove_indexes.add(idx)
                used_row.add(idx)
            group_info = [(num, original_df.iloc[num].tolist()) for num in group_members]
            group_log.append(group_info)
    return all_remove_indexes, group_log

def main():
    input_file_path = r"D:\Papers_related_materials\second_paper\Raw_data_and_cleaning\Data_cleaning\Data_after_deleting the_same_data.csv"
    output_file_name = "Data_after_cleaning.csv"
    log_file_name = "1.txt"

    df_original = pd.read_csv(input_file_path, header=0, encoding="utf-8-sig")
    df_original.columns = list(range(df_original.shape[1]))
    df_for_compare = preprocess_data(df_original)

    delete_index_set, group_log = find_groups_to_remove(df_for_compare, df_original)

    with open(log_file_name, 'w', encoding='utf-8') as f:
        group_num = 1
        for single_group in group_log:
            f.write(f"===== Matching Group {group_num} =====\n")
            for row_id, row_data in single_group:
                f.write(f"Row {row_id}: {', '.join(map(str, row_data))}\n")
            f.write("\n")
            group_num += 1
        f.write(f"Total matching groups removed: {len(group_log)}\n")
        f.write(f"Total removed data rows: {len(delete_index_set)}\n")

    df_cleaned = df_original.drop(index=delete_index_set)
    df_cleaned.to_csv(output_file_name, index=False, encoding="utf-8-sig")

    print("Processing finished successfully!")
    print(f"Total deleted rows: {len(delete_index_set)}")
    print(f"Remaining valid rows: {len(df_cleaned)}")
    print(f"Group information saved to {log_file_name}")
    print(f"Cleaned dataset saved to {output_file_name}")

if __name__ == "__main__":
    main()