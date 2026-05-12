"""LLM-as-Judge evaluation using GPT-4o or another strong model."""

import json
import os
import time
from typing import Literal

from openai import OpenAI

JUDGE_SYSTEM_PROMPT = """You are a strict but fair evaluator. Your task is to judge whether a model's answer correctly solves the user's question.

## Evaluation Rules
1. Focus on factual correctness, not wording style.
2. The model may have used tools (function calls) to arrive at the answer. Only judge the FINAL answer.
3. If the answer is partially correct but misses key information, mark it as FAIL.
4. If the model says "I don't know" or gives up, mark it as FAIL.
5. Answer format must be exactly: PASS or FAIL, followed by a one-sentence reason.

## Examples

User: What is the capital of France?
Model: The capital of France is Paris.
Judgment: PASS - Correctly identifies Paris as the capital.

User: What is the weather in Tokyo?
Model: The weather in Tokyo is sunny, 22 degrees Celsius.
Ground Truth: Tokyo weather is sunny, 22°C
Judgment: PASS - Weather description matches ground truth.

User: Who wrote the book 1984?
Model: I believe it was Aldous Huxley.
Ground Truth: George Orwell
Judgment: FAIL - Incorrect author; 1984 was written by George Orwell, not Aldous Huxley.
"""

JUDGE_USER_TEMPLATE = """Evaluate this model response:

User Question: {question}

Ground Truth (if available): {ground_truth}

Model's Final Answer: {model_answer}

Judgment (PASS or FAIL + reason):"""


class LLMJudge:
    """GPT-4o based evaluator for ReAct final answers."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.model = model
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "sk-placeholder"),
            base_url=base_url,
        )

    def judge(
        self,
        question: str,
        model_answer: str,
        ground_truth: str = "Not available",
        max_retries: int = 3,
    ) -> tuple[bool, str]:
        """Judge a single answer.

        Returns (pass_or_fail: bool, reason: str).
        """
        user_prompt = JUDGE_USER_TEMPLATE.format(
            question=question,
            ground_truth=ground_truth,
            model_answer=model_answer,
        )

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=200,
                )
                text = response.choices[0].message.content.strip()
                if text.upper().startswith("PASS"):
                    return True, text
                elif text.upper().startswith("FAIL"):
                    return False, text
                else:
                    # Retry on malformed response
                    time.sleep(1)
                    continue
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return False, f"Judge API error: {e}"

        return False, "Judge returned malformed response after retries"

    def judge_batch(
        self,
        samples: list[dict],
        concurrency: int = 5,
    ) -> list[dict]:
        """Judge a batch of samples.

        Each sample: {"question": str, "model_answer": str, "ground_truth": str}

        Returns list with added "pass" (bool) and "reason" (str) keys.
        """
        results = []
        for i in range(0, len(samples), concurrency):
            batch = samples[i : i + concurrency]
            for sample in batch:
                passed, reason = self.judge(
                    sample["question"],
                    sample["model_answer"],
                    sample.get("ground_truth", "Not available"),
                )
                sample["pass"] = passed
                sample["reason"] = reason
                results.append(sample)
        return results
