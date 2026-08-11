"""
Script to build a FAISS index from generated embeddings.
"""
import pandas as pd
import numpy as np
import faiss
import os


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
# LOAD EMBEDDINGS
# ==========================================

embeddings = np.load(
    "../results/train_embeddings.npy"
)

print("Embedding shape:", embeddings.shape)


# ==========================================
# NORMALIZE EMBEDDINGS
# ==========================================

faiss.normalize_L2(embeddings)


# ==========================================
# CREATE FAISS INDEX
# ==========================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)


# ==========================================
# ADD EMBEDDINGS
# ==========================================

index.add(embeddings)


print("\nFAISS index created.")

print("Number of vectors in index:")
print(index.ntotal)


# ==========================================
# SAVE INDEX
# ==========================================

os.makedirs("../results", exist_ok=True)

faiss.write_index(
    index,
    "../results/stance_index.faiss"
)

print("\nFAISS index saved successfully.")
