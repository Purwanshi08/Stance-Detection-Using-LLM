"""
Script to test retrieval using the built FAISS index.
"""
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


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
# LOAD FAISS INDEX
# ==========================================

index = faiss.read_index(
    "../results/stance_index.faiss"
)

print("FAISS vectors:", index.ntotal)


# ==========================================
# LOAD SENTENCE TRANSFORMER
# ==========================================

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
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

test_row = test_df.iloc[0]

target = test_row["Target"]
tweet = test_row["Tweet"]


query_text = (
    "Target: "
    + target
    + " Tweet: "
    + tweet
)

print("\nQuery:")
print(query_text)


# ==========================================
# GENERATE QUERY EMBEDDING
# ==========================================

query_embedding = model.encode(
    [query_text]
)

query_embedding = np.asarray(
    query_embedding,
    dtype="float32"
)


# Normalize because our FAISS index
# contains normalized vectors
faiss.normalize_L2(query_embedding)


# ==========================================
# SEARCH FAISS
# ==========================================

k = 10

scores, indices = index.search(
    query_embedding,
    k
)


# ==========================================
# DISPLAY RESULTS
# ==========================================

print("\nTop", k, "retrieved tweets:")
print("=" * 70)

for rank, (score, idx) in enumerate(
    zip(scores[0], indices[0]),
    start=1
):

    row = train_df.iloc[idx]

    print("\nRank:", rank)
    print("Similarity:", round(float(score), 4))
    print("Target:", row["Target"])
    print("Stance:", row["Stance"])
    print("Tweet:", str(row["Tweet"]).encode('ascii', 'ignore').decode('ascii'))
