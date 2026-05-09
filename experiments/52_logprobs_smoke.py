"""Smoke test: does the Azure OpenAI Responses API expose logprobs
on the user's GPT-5.4 deployment? Try 1 query both with and without
the logprobs flag and report what the response looks like.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from factor_leak.env import load_dotenv  # noqa: E402
load_dotenv(REPO / ".env")

from openai import AzureOpenAI  # noqa: E402

api_key = os.environ["AZURE_OPENAI_API_KEY"]
endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]

client = AzureOpenAI(api_key=api_key,
                    azure_endpoint=endpoint,
                    api_version=api_version,
                    timeout=60.0)

prompt = ("What was the monthly return of the Fama-French market "
          "excess return (Mkt-RF) factor in October 1987? Answer with "
          "a signed decimal percentage (e.g., -3.12) and nothing else.")

print(f"deployment: {deployment}")
print(f"api_version: {api_version}\n")

print("=== Attempt 1: with top_logprobs=5 ===")
try:
    resp = client.responses.create(
        model=deployment,
        input=prompt,
        max_output_tokens=24,
        top_logprobs=5,
        include=["message.output_text.logprobs"],
    )
    print("call succeeded")
    print(f"output_text: {resp.output_text!r}")
    # Try to surface logprobs
    for item in resp.output:
        print(f"item type: {type(item).__name__}")
        for piece in getattr(item, "content", []) or []:
            print(f"  piece type: {type(piece).__name__}")
            lp = getattr(piece, "logprobs", None)
            if lp:
                print(f"  logprobs found: {lp[:2] if isinstance(lp, list) else lp}")
            else:
                print(f"  no logprobs on this piece. keys: {dir(piece)}")
except Exception as e:
    print(f"call failed: {type(e).__name__}: {e}")

print("\n=== Attempt 2: bare call (sanity) ===")
try:
    resp = client.responses.create(
        model=deployment,
        input=prompt,
        max_output_tokens=24,
    )
    print(f"output_text: {resp.output_text!r}")
except Exception as e:
    print(f"call failed: {type(e).__name__}: {e}")
