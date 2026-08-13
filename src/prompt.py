"""
Script for defining prompts for stance detection.
"""
def create_prompt(tweet, target, examples):

    prompt = f"""
You are an expert in stance detection.

Your task is to determine the stance of a tweet
towards a given target.

The possible stance labels are:
FAVOR
AGAINST
NONE

Target:
{target}

Here are some examples:

"""

    # Add the retrieved examples
    for i, (_, row) in enumerate(
        examples.iterrows(),
        start=1
    ):

        prompt += f"""
Example {i}:

Tweet:
{row["Tweet"]}

Stance:
{row["Stance"]}

"""

    # Add the query tweet
    prompt += f"""
Now classify the following tweet with respect to the target.

Tweet:
{tweet}

Determine whether the tweet expresses:

- FAVOR: supports the target
- AGAINST: opposes the target
- NONE: does not clearly express a stance toward the target

Return your answer in exactly this format:

Stance: <FAVOR/AGAINST/NONE>

Explanation: <brief explanation based on the tweet and target>
"""

    return prompt

if __name__ == "__main__":

    import pandas as pd

    # Load training data
    train_df = pd.read_csv(
        "../data/trainingdata-all-annotations.txt",
        sep="\t",
        encoding="latin1"
    )

    # Temporary test examples
    examples = train_df[
        train_df["Target"] ==
        "Climate Change is a Real Concern"
    ].head(3)

    tweet = "Scientists have provided enough evidence."

    target = "Climate Change is a Real Concern"

    prompt = create_prompt(
        tweet,
        target,
        examples
    )

    print(prompt.encode('ascii', 'ignore').decode('ascii'))
