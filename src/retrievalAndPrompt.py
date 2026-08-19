"""
Retrieval + prompt generation pipeline.
"""

import pandas as pd

from mmr_retrieval import retrieve_mmr_examples
from prompt import create_prompt


# ==========================================
# LOAD TEST DATA
# ==========================================

test_df = pd.read_csv(
    "../data/testdata-taskA-all-annotations.txt",
    sep="\t",
    encoding="latin1"
)


# ==========================================
# DEVELOPMENT MODE
# ==========================================

test_sample = test_df.head(1)


# ==========================================
# RETRIEVAL + PROMPT PIPELINE
# ==========================================

for _, row in test_sample.iterrows():

    tweet = row["Tweet"]
    target = row["Target"]


    # ======================================
    # MMR RETRIEVAL
    # ======================================

    examples = retrieve_mmr_examples(
        tweet=tweet,
        target=target,

        # Retrieve a large pool from FAISS
        faiss_k=100,

        # Give MMR enough candidates
        mmr_candidate_k=50,

        # Final number of examples
        final_k=3,

        # Balance relevance and diversity
        lambda_param=0.5
    )


    # ======================================
    # CREATE PROMPT
    # ======================================

    prompt = create_prompt(
        tweet=tweet,
        target=target,
        examples=examples
    )


    # ======================================
    # DISPLAY FINAL PROMPT
    # ======================================

    print("\n")
    print("=" * 80)
    print("FINAL PROMPT")
    print("=" * 80)

    print(
        prompt.encode(
            "ascii",
            "ignore"
        ).decode("ascii")
    )