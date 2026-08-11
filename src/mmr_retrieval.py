"""
Script to test Maximum Marginal Relevance (MMR) retrieval.
"""
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


# ==========================================
# LOAD DATA
# ==========================================

train_df = pd.read_csv(
    "../data/trainingdata-all-annotations.txt",
    sep="\t",
    encoding="latin1"
)

embeddings = np.load(
    "../results/train_embeddings.npy"
).astype("float32")

index = faiss.read_index(
    "../results/stance_index.faiss"
)

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==========================================
# MMR
# ==========================================

def mmr(
    query_embedding,
    candidate_embeddings,
    candidate_indices,
    k=3,
    lambda_param=0.7
):

    selected = []

    query_scores = np.dot(
        candidate_embeddings,
        query_embedding
    )

    while len(selected) < k:

        mmr_scores = []

        for i in range(len(candidate_indices)):

            if i in selected:
                mmr_scores.append(-np.inf)
                continue

            relevance = query_scores[i]

            if len(selected) == 0:

                diversity_penalty = 0

            else:

                selected_embeddings = (
                    candidate_embeddings[selected]
                )

                similarities = np.dot(
                    selected_embeddings,
                    candidate_embeddings[i]
                )

                diversity_penalty = np.max(
                    similarities
                )

            score = (
                lambda_param * relevance
                -
                (1 - lambda_param)
                * diversity_penalty
            )

            mmr_scores.append(score)

        best = np.argmax(mmr_scores)

        selected.append(best)

    return [
        candidate_indices[i]
        for i in selected
    ]


# ==========================================
# GENERIC RETRIEVAL FUNCTION
# ==========================================

def retrieve_mmr_examples(
    tweet,
    target,
    top_k=20,
    final_k=3,
    lambda_param=0.7
):

    query_text = (
        "Target: "
        + target
        + " Tweet: "
        + tweet
    )

    # Generate query embedding
    query_embedding = model.encode(
        [query_text]
    )[0]

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    # Normalize
    query_embedding /= np.linalg.norm(
        query_embedding
    )

    # --------------------------------------
    # FAISS: retrieve candidate pool
    # --------------------------------------

    scores, indices = index.search(
        query_embedding.reshape(1, -1),
        top_k
    )

    candidate_indices = indices[0]

    # --------------------------------------
    # Keep same target
    # --------------------------------------

    candidate_indices = [
        idx
        for idx in candidate_indices
        if train_df.iloc[idx]["Target"] == target
    ]

    # --------------------------------------
    # MMR
    # --------------------------------------

    candidate_embeddings = embeddings[
        candidate_indices
    ]

    selected_indices = mmr(
        query_embedding,
        candidate_embeddings,
        candidate_indices,
        k=min(final_k, len(candidate_indices)),
        lambda_param=lambda_param
    )

    return train_df.iloc[
        selected_indices
    ]

# ==========================================
# TEST
# ==========================================

examples = retrieve_mmr_examples(
    tweet="Scientists have provided enough evidence.",
    target="Climate Change is a Real Concern"
)

print("\nSelected examples:")
print("=" * 70)

for _, row in examples.iterrows():

    print("\nTarget:", row["Target"])
    print("Tweet:", str(row["Tweet"]).encode('ascii', 'ignore').decode('ascii'))
    print("Stance:", row["Stance"])
    
