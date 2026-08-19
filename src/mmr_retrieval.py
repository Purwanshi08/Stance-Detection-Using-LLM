"""
MMR retrieval for stance detection.
Shows:
1. Top 3 FAISS examples before MMR
2. 3 examples selected by MMR
3. Pairwise similarity between MMR examples
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

# Normalize embeddings
faiss.normalize_L2(embeddings)

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
    lambda_param=0.5
):

    selected = []

    # Similarity of every candidate to query
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

            # First example:
            # no diversity penalty
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

                diversity_penalty = np.max(similarities)

            score = (
                lambda_param * relevance
                -
                (1 - lambda_param) * diversity_penalty
            )

            mmr_scores.append(score)

        best = np.argmax(mmr_scores)

        selected.append(best)

    return [
        candidate_indices[i]
        for i in selected
    ]


# ==========================================
# RETRIEVAL + MMR
# ==========================================

def retrieve_mmr_examples(
    tweet,
    target,
    faiss_k=100,
    mmr_candidate_k=50,
    final_k=3,
    lambda_param=0.5
):

    query_text = (
        "Target: "
        + target
        + " Tweet: "
        + tweet
    )

    # --------------------------------------
    # QUERY EMBEDDING
    # --------------------------------------

    query_embedding = model.encode(
        [query_text]
    )[0]

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    query_embedding /= np.linalg.norm(
        query_embedding
    )


    # ======================================
    # STEP 1: FAISS RETRIEVAL
    # ======================================

    scores, indices = index.search(
        query_embedding.reshape(1, -1),
        faiss_k
    )

    faiss_indices = indices[0]


    # ======================================
    # STEP 2: SAME TARGET FILTER
    # ======================================

    candidate_indices = [
        idx
        for idx in faiss_indices
        if idx >= 0
        and train_df.iloc[idx]["Target"] == target
    ]


    # Remove duplicates just in case
    candidate_indices = list(
        dict.fromkeys(candidate_indices)
    )


    if len(candidate_indices) == 0:

        print("\nNo training examples found for target:")
        print(target)

        return train_df.iloc[[]]


    # ======================================
    # STEP 3: TOP 3 FAISS EXAMPLES
    # BEFORE MMR
    # ======================================

    print("\n")
    print("#" * 80)
    print("TOP 3 FAISS EXAMPLES — BEFORE MMR")
    print("#" * 80)

    top_faiss_indices = candidate_indices[:3]

    for rank, idx in enumerate(
        top_faiss_indices,
        start=1
    ):

        similarity = np.dot(
            embeddings[idx],
            query_embedding
        )

        row = train_df.iloc[idx]

        print(f"\nFAISS Example {rank}")
        print("-" * 60)
        print("Similarity:", round(float(similarity), 4))
        print(
            "Tweet:",
            str(row["Tweet"])
        )
        print(
            "Stance:",
            row["Stance"]
        )


    # ======================================
    # STEP 4: MMR CANDIDATE POOL
    # ======================================

    mmr_candidates = candidate_indices[
        :mmr_candidate_k
    ]

    candidate_embeddings = embeddings[
        mmr_candidates
    ]


    # ======================================
    # STEP 5: MMR
    # ======================================

    selected_indices = mmr(
        query_embedding=query_embedding,
        candidate_embeddings=candidate_embeddings,
        candidate_indices=mmr_candidates,
        k=min(
            final_k,
            len(mmr_candidates)
        ),
        lambda_param=lambda_param
    )


    # ======================================
    # STEP 6: DISPLAY MMR RESULTS
    # ======================================

    print("\n")
    print("#" * 80)
    print("3 MMR EXAMPLES — AFTER MMR")
    print("#" * 80)

    for rank, idx in enumerate(
        selected_indices,
        start=1
    ):

        relevance = np.dot(
            embeddings[idx],
            query_embedding
        )

        row = train_df.iloc[idx]

        print(f"\nMMR Example {rank}")
        print("-" * 60)

        print(
            "Query similarity:",
            round(float(relevance), 4)
        )

        print(
            "Tweet:",
            str(row["Tweet"])
        )

        print(
            "Stance:",
            row["Stance"]
        )


    # ======================================
    # STEP 7: PAIRWISE SIMILARITY
    # ======================================

    print("\n")
    print("#" * 80)
    print("PAIRWISE SIMILARITY BETWEEN MMR EXAMPLES")
    print("#" * 80)

    selected_embeddings = embeddings[
        selected_indices
    ]

    similarity_matrix = np.dot(
        selected_embeddings,
        selected_embeddings.T
    )

    print(
        np.round(
            similarity_matrix,
            4
        )
    )


    # Print individual pairs too

    print("\nIndividual pairwise similarities:")

    for i in range(
        len(selected_indices)
    ):

        for j in range(
            i + 1,
            len(selected_indices)
        ):

            similarity = np.dot(
                selected_embeddings[i],
                selected_embeddings[j]
            )

            print(
                f"MMR Example {i+1} "
                f"<-> "
                f"MMR Example {j+1}: "
                f"{similarity:.4f}"
            )


    return train_df.iloc[
        selected_indices
    ]