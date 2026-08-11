"""
Script to prepare and clean the stance detection dataset.
"""
import pandas as pd


# ==========================================
# LOAD TRAINING DATA
# ==========================================

train_path = "../data/trainingdata-all-annotations.txt"

train_df = pd.read_csv(
    train_path,
    sep="\t",
    encoding="latin1"
)


# ==========================================
# LOAD TEST DATA
# ==========================================

test_path = "../data/testdata-taskA-all-annotations.txt"

test_df = pd.read_csv(
    test_path,
    sep="\t",
    encoding="latin1"
)


# ==========================================
# DISPLAY INFORMATION
# ==========================================

print("TRAINING DATA")
print("==============================")
print("Shape:", train_df.shape)
print("Columns:", train_df.columns.tolist())


print("\nTEST DATA")
print("==============================")
print("Shape:", test_df.shape)
print("Columns:", test_df.columns.tolist())


print("\nFirst 5 test rows:")
print(test_df.head())
