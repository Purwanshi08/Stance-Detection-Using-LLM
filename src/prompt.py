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
    prompt += """
    The following examples were retrieved because they are
    semantically relevant to the NEW TWEET. They are provided
    as in-context examples to help determine the stance.

    """

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

# Add the query/test tweet
    prompt += f"""
    Now classify the following NEW TWEET with respect to the target.

    Target:
    {target}

    NEW TWEET:
    {tweet}

    Determine whether the NEW TWEET expresses:

    - FAVOR: supports the target
    - AGAINST: opposes the target
    - NONE: does not clearly express a stance toward the target

    Use the retrieved examples above as guidance, but classify the NEW TWEET itself.

    You MUST provide both lines below. Do NOT skip the Explanation line.

    Stance: <FAVOR/AGAINST/NONE>
    Explanation: <2-3 sentences explaining why>
    """

    return prompt
