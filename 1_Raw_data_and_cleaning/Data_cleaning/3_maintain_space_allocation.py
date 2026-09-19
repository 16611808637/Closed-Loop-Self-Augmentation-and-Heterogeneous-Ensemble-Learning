import pandas as pd
import numpy as np


def process_data(input_file='Data_after_cleaning.csv', output_file='train_and_test.csv'):
    # ========== 1. Read data ==========
    df = pd.read_csv(input_file)
    print(f"Original number of rows: {len(df)}")

    # Extract the first 15 composition columns (index 0~14)
    comp_cols = df.columns[:15].tolist()
    data = df[comp_cols].values.astype(float)  # shape: (N, 15)

    # ========== 2. Define conflict detection function ==========
    def is_conflict(row_a, row_b):
        """
        Determine whether two rows are in conflict:
        For the first 15 columns, consider only columns where the difference is non-zero.
        If abs(val1 - val2) < 2 for ALL such non-zero-difference columns, they conflict.
        (i.e., there is no column with a difference >= 2)
        """
        diff = np.abs(row_a - row_b)
        # Only consider columns with non-zero differences
        nonzero_mask = diff > 0
        if not np.any(nonzero_mask):
            # All column differences are zero, i.e., two identical rows -> conflict
            return True
        # Check if all non-zero differences are < 2
        return np.all(diff[nonzero_mask] < 2)

    # ========== 3. Build conflict graph (adjacency list) ==========
    N = len(data)
    print(f"Computing conflict relationships among {N} rows...")

    # Use sets to store conflicting row indices for each row
    conflicts = {i: set() for i in range(N)}

    # Vectorized acceleration: compare row by row
    for i in range(N):
        row_i = data[i]  # shape: (15,)
        # Compare with all subsequent rows
        diff_matrix = np.abs(data[i+1:] - row_i)  # shape: (N-i-1, 15)

        for j_offset in range(diff_matrix.shape[0]):
            diff = diff_matrix[j_offset]
            nonzero_mask = diff > 0
            if not np.any(nonzero_mask):
                # Completely identical rows
                conflicts[i].add(i + 1 + j_offset)
                conflicts[i + 1 + j_offset].add(i)
            elif np.all(diff[nonzero_mask] < 2):
                j = i + 1 + j_offset
                conflicts[i].add(j)
                conflicts[j].add(i)

    total_conflicts = sum(len(v) for v in conflicts.values()) // 2
    print(f"Initial number of conflict pairs: {total_conflicts}")

    # ========== 4. Greedy removal of the most conflicting row ==========
    removed = set()
    iteration = 0

    while True:
        # Find the row with the most current conflicts (among non-removed rows)
        max_conflict_count = 0
        max_conflict_row = -1

        for i in range(N):
            if i in removed:
                continue
            # Count currently active conflicts
            current_conflicts = len(conflicts[i] - removed)
            if current_conflicts > max_conflict_count:
                max_conflict_count = current_conflicts
                max_conflict_row = i

        # Exit loop if no conflicts remain
        if max_conflict_count == 0:
            break

        # Remove the row with the most conflicts
        removed.add(max_conflict_row)
        iteration += 1

        if iteration % 50 == 0:
            print(f"  Iteration {iteration}: Removed row {max_conflict_row} "
                  f"(conflicts={max_conflict_count}), total removed: {len(removed)}")

    print(f"Total rows removed: {len(removed)}")

    # ========== 5. Save results ==========
    # Keep rows that were not removed (preserve original header)
    keep_mask = [i for i in range(N) if i not in removed]
    result_df = df.iloc[keep_mask].reset_index(drop=True)

    result_df.to_csv(output_file, index=False)
    print(f"Saved to: {output_file}")
    print(f"Final number of rows: {len(result_df)} "
          f"(original: {N}, removed: {len(removed)})")

    return result_df


if __name__ == '__main__':
    result = process_data()