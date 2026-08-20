"""
Module for interacting with OpenCode Go LLM for stance classification.
"""
import os
import re
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        "Set OPENAI_API_KEY in your .env file or environment."
    )

client = OpenAI(
    api_key=api_key,
    base_url="https://opencode.ai/zen/go/v1"
)

MODEL = "deepseek-v4-flash"

SYSTEM_INSTRUCTION = """You are an expert stance detection system.

Your task: determine if a tweet expresses FAVOR, AGAINST, or NONE toward a given target.

Rules:
- FAVOR: the tweet supports/agrees with the target
- AGAINST: the tweet opposes/disagrees with the target
- NONE: the tweet does not clearly express a stance toward the target

You MUST respond with exactly two lines:
Line 1: Stance: <FAVOR or AGAINST or NONE>
Line 2: Explanation: <2-3 sentences citing specific words from the tweet and explaining why the stance was chosen>"""


def get_stance(prompt, max_retries=3):
    """
    Send the prompt to OpenAI and parse stance + explanation.
    Returns: {"stance": "FAVOR|AGAINST|NONE", "explanation": "..."}
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=500,
            )

            text = response.choices[0].message.content.strip()

            stance = "NONE"
            explanation = ""

            stance_match = re.search(
                r"Stance:\s*(FAVOR|AGAINST|NONE)",
                text,
                re.IGNORECASE,
            )

            if stance_match:
                stance = stance_match.group(1).upper()

            lines = text.split("\n")

            for line in lines:
                line = line.strip()
                if re.match(r"Explanation:", line, re.IGNORECASE):
                    explanation = re.sub(
                        r"^Explanation:\s*", "", line, flags=re.IGNORECASE
                    ).strip()
                    break

            if not explanation:
                for line in lines:
                    line = line.strip()
                    if line and not re.match(
                        r"Stance:", line, re.IGNORECASE
                    ):
                        explanation = line
                        break

            return {
                "stance": stance,
                "explanation": explanation,
            }

        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  API error (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  API failed after {max_retries} attempts: {e}")
                return {
                    "stance": "NONE",
                    "explanation": f"API error: {e}",
                }
