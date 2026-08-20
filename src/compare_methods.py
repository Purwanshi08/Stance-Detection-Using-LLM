"""
Compare 3 prompt augmentation methods for stance detection:
1. SDPStan-R  — Random examples
2. SDPStan-S  — Similarity-based (top-k FAISS)
3. Our Method — FAISS + MMR (diversity-aware)
"""
import sys
import io
import time
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# Suppress verbose output from retrieval functions
class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = io.StringIO()
        return self
    def __exit__(self, *args):
        sys.stdout = self._original_stdout

from mmr_retrieval import (
    retrieve_random_examples,
    retrieve_topk_examples,
    retrieve_mmr_examples,
)
from prompt import create_prompt
from llm import get_stance


# ==========================================
# CONFIG
# ==========================================

NUM_TWEETS = 50
FINAL_K = 3
LABELS = ["FAVOR", "AGAINST", "NONE"]


# ==========================================
# LOAD TEST DATA
# ==========================================

test_df = pd.read_csv(
    "../data/testdata-taskA-all-annotations.txt",
    sep="\t",
    encoding="latin1",
)

test_sample = test_df.head(NUM_TWEETS)

print(f"Comparing 3 methods on {len(test_sample)} tweets...\n")


# ==========================================
# METHOD FUNCTIONS
# ==========================================

def run_random(tweet, target):
    examples = retrieve_random_examples(
        tweet, target, final_k=FINAL_K
    )
    prompt = create_prompt(tweet, target, examples)
    return get_stance(prompt)


def run_similarity(tweet, target):
    examples = retrieve_topk_examples(
        tweet, target, faiss_k=100, final_k=FINAL_K
    )
    prompt = create_prompt(tweet, target, examples)
    return get_stance(prompt)


def run_mmr(tweet, target):
    examples = retrieve_mmr_examples(
        tweet, target,
        faiss_k=100,
        mmr_candidate_k=50,
        final_k=FINAL_K,
        lambda_param=0.5,
    )
    prompt = create_prompt(tweet, target, examples)
    return get_stance(prompt)


METHODS = {
    "SDPStan-R (Random)": run_random,
    "SDPStan-S (Similarity)": run_similarity,
    "Our Method (MMR)": run_mmr,
}


# ==========================================
# RUN ALL METHODS
# ==========================================

results = {name: [] for name in METHODS}

for i, (_, row) in enumerate(
    test_sample.iterrows(), start=1
):

    tweet = row["Tweet"]
    target = row["Target"]
    true_stance = row["Stance"]

    print(f"[{i}/{len(test_sample)}] {target}")

    for name, method_fn in METHODS.items():

        with SuppressOutput():
            result = method_fn(tweet, target)
        predicted = result["stance"]
        correct = predicted == true_stance

        results[name].append({
            "ID": row["ID"],
            "Target": target,
            "Tweet": tweet,
            "True_Stance": true_stance,
            "Predicted_Stance": predicted,
            "Explanation": result["explanation"],
            "Correct": correct,
        })

        time.sleep(2)

    # Print progress summary for this tweet
    statuses = []
    for name in METHODS:
        pred = results[name][-1]["Predicted_Stance"]
        mark = "CORRECT" if results[name][-1]["Correct"] else "WRONG"
        statuses.append(f"{name}: {pred} {mark}")

    print(f"  True: {true_stance}")
    for s in statuses:
        print(f"    {s}")
    print()


# ==========================================
# COMPUTE METRICS PER METHOD
# ==========================================

print("\n" + "=" * 80)
print("COMPARISON RESULTS")
print("=" * 80)

all_metrics = {}

for name in METHODS:

    pred_df = pd.DataFrame(results[name])
    y_true = pred_df["True_Stance"]
    y_pred = pred_df["Predicted_Stance"]

    accuracy = accuracy_score(y_true, y_pred)

    report_dict = classification_report(
        y_true, y_pred,
        labels=LABELS,
        zero_division=0,
        output_dict=True,
    )

    all_metrics[name] = {
        "accuracy": accuracy,
        "report": report_dict,
        "pred_df": pred_df,
    }


# ==========================================
# PRINT COMPARISON TABLE
# ==========================================

print(f"\n{'Method':<25} {'Accuracy':>10} {'FAV-F1':>10} "
      f"{'AGN-F1':>10} {'NON-F1':>10} {'Macro-F1':>10}")

print("-" * 85)

for name in METHODS:
    m = all_metrics[name]
    acc = m["accuracy"]
    fav_f1 = m["report"]["FAVOR"]["f1-score"]
    agn_f1 = m["report"]["AGAINST"]["f1-score"]
    non_f1 = m["report"]["NONE"]["f1-score"]
    macro_f1 = m["report"]["macro avg"]["f1-score"]

    print(
        f"{name:<25} {acc:>9.2%} {fav_f1:>10.3f} "
        f"{agn_f1:>10.3f} {non_f1:>10.3f} {macro_f1:>10.3f}"
    )


# ==========================================
# PRINT PER-METHOD DETAILED REPORT
# ==========================================

for name in METHODS:
    print(f"\n\n{'=' * 60}")
    print(f"DETAILED REPORT: {name}")
    print(f"{'=' * 60}")

    pred_df = all_metrics[name]["pred_df"]
    y_true = pred_df["True_Stance"]
    y_pred = pred_df["Predicted_Stance"]

    print(classification_report(
        y_true, y_pred,
        labels=LABELS,
        zero_division=0,
    ))

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    print("Confusion Matrix:")
    print(f"{'':>12}", end="")
    for label in LABELS:
        print(f"{label:>12}", end="")
    print()
    for i, row_label in enumerate(LABELS):
        print(f"{row_label:>12}", end="")
        for j in range(len(LABELS)):
            print(f"{cm[i][j]:>12}", end="")
        print()


# ==========================================
# SAVE ALL PREDICTIONS
# ==========================================

import os
os.makedirs("../results", exist_ok=True)

for name in METHODS:
    safe_name = name.split("(")[0].strip().replace(
        " ", "_"
    ).replace("-", "_")
    pred_df = pd.DataFrame(results[name])
    pred_df.to_csv(
        f"../results/predictions_{safe_name}.csv",
        index=False,
    )

print("\nAll predictions saved to results/")
