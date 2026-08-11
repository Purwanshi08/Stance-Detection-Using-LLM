"""
Script to generate text embeddings using sentence-transformers.
"""
import pandas as pd
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
# CREATE TEXT FOR EMBEDDING
# ==========================================

train_df["embedding_text"] = (
    "Target: "
    + train_df["Target"]
    + " Tweet: "
    + train_df["Tweet"]
)

print("Example text:")
print(train_df["embedding_text"].iloc[0])


# ==========================================
# LOAD SENTENCE TRANSFORMER
# ==========================================

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==========================================
# GENERATE EMBEDDINGS
# ==========================================

embeddings = model.encode(
    train_df["embedding_text"].tolist(),
    show_progress_bar=True
)


# ==========================================
# CHECK EMBEDDINGS
# ==========================================

print("\nEmbedding shape:")
print(embeddings.shape)

print("\nFirst embedding:")
print(embeddings[0])

import numpy as np
import os

# Convert embeddings to float32
embeddings = np.asarray(
    embeddings,
    dtype="float32"
)

# Create results directory if it doesn't exist
os.makedirs("../results", exist_ok=True)

# Save embeddings
np.save(
    "../results/train_embeddings.npy",
    embeddings
)

print("\nEmbeddings saved successfully.")
