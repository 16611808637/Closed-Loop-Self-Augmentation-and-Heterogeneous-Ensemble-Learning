import csv


def read_csv(file_name):
    data = []
    with open(file_name, 'r', encoding='utf-8-sig') as file:
        reader = csv.reader(file)
        header = next(reader)  # Read and store header line separately
        data.append(header)    # Preserve header in final output file
        for row in reader:
            processed_row = []
            for idx, val in enumerate(row):
                # Do not fill 0 for any column
                if idx < 15:
                    # Convert first 15 columns to float value
                    if val.strip():
                        processed_row.append(float(val))
                    else:
                        processed_row.append(None)
                else:
                    # Column 1516 keep original string type (original 1617 column)
                    processed_row.append(val)
            data.append(processed_row)
    return data


def are_group_matched(row1, row2, threshold=2):
    # Compare first 15 numerical columns
    for col_idx in range(15):
        val1 = row1[col_idx]
        val2 = row2[col_idx]
        # Skip comparison if value is empty
        if val1 is None or val2 is None:
            return False
        if abs(val1 - val2) > threshold:
            return False
    # Original 16th column (index15) AND original 17th column(index16) both need exact match
    if row1[15] != row2[15] or row1[16] != row2[16]:
        return False
    return True


def process_data(input_file, output_file):
    raw_data = read_csv(input_file)
    remove_index_set = set()
    total_rows = len(raw_data)

    # Start traversal from index 1, skip header(index=0)
    for main_idx in range(1, total_rows):
        # Skip current row if already marked to be removed
        if main_idx in remove_index_set:
            continue
        main_row = raw_data[main_idx]
        # Match with subsequent rows
        for compare_idx in range(main_idx + 1, total_rows):
            if compare_idx in remove_index_set:
                continue
            compare_row = raw_data[compare_idx]
            if are_group_matched(main_row, compare_row, threshold=0.1):
                remove_index_set.add(compare_idx)

    # Reserve header + valid data rows
    filtered_data = [row for idx, row in enumerate(raw_data) if idx not in remove_index_set]

    # Write filtered dataset to target csv file
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as f_out:
        csv_writer = csv.writer(f_out)
        csv_writer.writerows(filtered_data)


def main():
    # Full absolute input file path
    input_path = r"D:\Papers_related_materials\second_paper\Raw_data_and_cleaning\data_reference.csv"
    # Output file name
    output_path = "Data_after_deleting the_same_data.csv"
    process_data(input_path, output_path)


if __name__ == "__main__":
    main()