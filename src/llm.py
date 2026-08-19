"""
Module for interacting with Google Gemini LLM for stance classification.
"""
import os
import re
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "Set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file or environment."
    )

client = genai.Client(api_key=api_key)

MODEL = "gemini-3.5-flash"

SYSTEM_INSTRUCTION = """You are an expert stance detection system.

Your task: determine if a tweet expresses FAVOR, AGAINST, or NONE toward a given target.

Rules:
- FAVOR: the tweet supports/agrees with the target
- AGAINST: the tweet opposes/disagrees with the target
- NONE: the tweet does not clearly express a stance toward the target

CRITICAL: You MUST respond with exactly TWO lines:
Line 1: Stance: <FAVOR or AGAINST or NONE>
Line 2: Explanation: <one or two sentences explaining why>

Do NOT output anything else. Do NOT skip the Explanation line."""


def get_stance(prompt):
    """
    Send the prompt to Gemini and parse stance + explanation.
    Returns: {"stance": "FAVOR|AGAINST|NONE", "explanation": "..."}
    """
    chat = client.chats.create(
        model=MODEL,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.1,
            max_output_tokens=500,
        ),
    )

    response = chat.send_message(prompt)

    text = response.text.strip()

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
