"""
Evaluation script for stance detection.
Runs the RAG pipeline on N test tweets and computes metrics.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from mmr_retrieval import retrieve_mmr_examples
from prompt import create_prompt
from llm import get_stance


# ==========================================
# CONFIG
# ==========================================

SAVE_CSV = True


# ==========================================
# LOAD TEST DATA
# ==========================================

test_df = pd.read_csv(
    "../data/testdata-taskA-all-annotations.txt",
    sep="\t",
    encoding="latin1",
)

test_sample = test_df

print(f"Evaluating on {len(test_sample)} tweets...\n")


# ==========================================
# RUN PIPELINE
# ==========================================

predictions = []

for i, (_, row) in enumerate(
    test_sample.iterrows(), start=1
):

    tweet = row["Tweet"]
    target = row["Target"]
    true_stance = row["Stance"]

    print(f"[{i}/{len(test_sample)}] Target: {target}")
    print(f"  Tweet: {tweet[:80]}...")

    # Retrieve examples
    examples = retrieve_mmr_examples(
        tweet=tweet,
        target=target,
        faiss_k=100,
        mmr_candidate_k=50,
        final_k=3,
        lambda_param=0.5,
    )

    # Build prompt
    prompt = create_prompt(
        tweet=tweet,
        target=target,
        examples=examples,
    )

    # Get LLM prediction
    result = get_stance(prompt)
    predicted_stance = result["stance"]

    correct = predicted_stance == true_stance
    status = "CORRECT" if correct else "WRONG"

    print(f"  True: {true_stance} | Predicted: {predicted_stance} | {status}")
    print(f"  Explanation: {result['explanation'][:100]}")
    print()

    predictions.append(
        {
            "ID": row["ID"],
            "Target": target,
            "Tweet": tweet,
            "True_Stance": true_stance,
            "Predicted_Stance": predicted_stance,
            "Explanation": result["explanation"],
            "Correct": correct,
        }
    )


# ==========================================
# COMPUTE METRICS
# ==========================================

pred_df = pd.DataFrame(predictions)

y_true = pred_df["True_Stance"]
y_pred = pred_df["Predicted_Stance"]

print("\n" + "=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)

# Accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"\nAccuracy: {accuracy:.2%}")
print(f"Correct: {sum(pred_df['Correct'])}/{len(pred_df)}")

# Per-class metrics
labels = ["FAVOR", "AGAINST", "NONE"]

print("\nPer-class metrics:")
report = classification_report(
    y_true,
    y_pred,
    labels=labels,
    zero_division=0,
)
print(report)

# Confusion matrix
print("Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred, labels=labels)

# Header
print(f"\n{'':>12}", end="")
for label in labels:
    print(f"{label:>12}", end="")
print()

# Rows
for i, row_label in enumerate(labels):
    print(f"{row_label:>12}", end="")
    for j in range(len(labels)):
        print(f"{cm[i][j]:>12}", end="")
    print()


# ==========================================
# SAVE PREDICTIONS
# ==========================================

if SAVE_CSV:
    import os

    os.makedirs("../results", exist_ok=True)

    pred_df.to_csv(
        "../results/predictions.csv",
        index=False,
    )

    print(f"\nPredictions saved to results/predictions.csv")
