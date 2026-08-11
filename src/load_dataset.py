""" Functions to load stance detection datasets. """
import pandas as pd

# Path to training dataset
path = "../data/trainingdata-all-annotations.txt"

# Load dataset
df = pd.read_csv(
    path,
    sep="\t",
    encoding="latin1"
)

# Display basic information
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nStance distribution:")
print(df["Stance"].value_counts())

print("\nTargets:")
print(df["Target"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())

# Keep the columns needed for our stance detection project
data = df[["ID", "Target", "Tweet", "Stance"]].copy()

print("\nWorking dataset:")
print(data.head())

print("\nShape:", data.shape)

print("\nDuplicate tweets:", data["Tweet"].duplicated().sum())
print(
    "Duplicate Tweet + Target:",
    data.duplicated(subset=["Tweet", "Target"]).sum()
)

print("\nStance distribution by target:")
print(
    pd.crosstab(
        data["Target"],
        data["Stance"]
    )
)

for stance in ["FAVOR", "AGAINST", "NONE"]:
    print("\n-----------------")
    print("STANCE:", stance)
    print("-------------------")

    samples = data[data["Stance"] == stance].head(3)

    for _, row in samples.iterrows():
        print("\nTarget:", row["Target"])
        print("Tweet:", str(row["Tweet"]).encode('ascii', 'ignore').decode('ascii'))
