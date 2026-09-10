#!/usr/bin/env python3
"""
Generate a test RAG prompt with a real integral and MSE answer from the database.
This can be pasted into ChatGPT or Claude to manually test the RAG concept.
"""
import json
import random
import re
from pathlib import Path

# Add IntegralIndx to path
import sys
INTEGRALINDX_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(INTEGRALINDX_ROOT))

from src.database.integral_db import IntegralDatabase


def html_to_text(html):
    """simple HTML to text conversion"""
    # remove script and style tags
    text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
    text = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL)
    # replace common tags with newlines
    text = re.sub(r'<br\s*/?>', '\n', text)
    text = re.sub(r'</p>', '\n\n', text)
    text = re.sub(r'</div>', '\n', text)
    # remove all other tags
    text = re.sub(r'<[^>]+>', '', text)
    # decode HTML entities
    text = text.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
    text = text.replace('&#39;', "'").replace('&quot;', '"')
    return text.strip()


def main():
    # load database
    db = IntegralDatabase(Path('data/integral.db'))

    # get some groups and find one with MSE metadata
    groups = db.get_all_groups(limit=200, exclude_curated=True)

    selected_instance = None
    selected_group = None

    for group in groups:
        # check if any instances have MSE metadata
        for inst in group.definite_instances + group.indefinite_instances:
            if hasattr(inst, 'mse_metadata') and inst.mse_metadata:
                if inst.mse_metadata.mse_answer_id:
                    selected_instance = inst
                    selected_group = group
                    break
        if selected_instance:
            break

    if not selected_instance:
        print("No instances with MSE metadata found!")
        return

    # load the answer blob
    answer_id = selected_instance.mse_metadata.mse_answer_id
    question_id = selected_instance.mse_metadata.mse_question_id
    blob_path = Path(f'data/raw/answer_blobs/answer_{answer_id}.json')

    if not blob_path.exists():
        print(f"Answer blob not found: {blob_path}")
        return

    with open(blob_path, 'r', encoding='utf-8') as f:
        blob_data = json.load(f)

    if not blob_data.get('items'):
        print("No items in blob!")
        return

    answer_item = blob_data['items'][0]

    # extract answer details
    body_html = answer_item.get('body', '')
    score = answer_item.get('score', 0)
    is_accepted = answer_item.get('is_accepted', False)
    author = answer_item.get('owner', {}).get('display_name', 'Unknown')

    # convert HTML to text
    try:
        body_text = html_to_text(body_html)
    except:
        body_text = body_html  # fallback to raw HTML

    # construct the prompt
    prompt = f"""You are a mathematical assistant specializing in integral calculus.
You will be given an integral query and a relevant Math StackExchange answer.

CRITICAL INSTRUCTIONS:
1. EXTRACT, do not GENERATE: Copy formulas and techniques from the provided answer. Never invent new mathematical content.
2. ATTRIBUTE sources: Link all claims to the MSE answer.
3. ADMIT uncertainty: If the answer doesn't contain a closed form, say "No closed form found in this answer" (NOT "no closed form exists").
4. CLASSIFY difficulty: Based on techniques mentioned (not your judgment).

OUTPUT FORMAT (JSON):
{{
  "techniques": [
    "Technique 1 (from this answer)",
    "Technique 2 (from this answer)"
  ],
  "closed_form": "exact formula from answer or null",
  "closed_form_confidence": "high|medium|low|none",
  "difficulty": "elementary|intermediate|advanced",
  "summary": "2-3 sentence synthesis of the solution approach",
  "relevance": "high|medium|low"
}}

DIFFICULTY DEFINITIONS:
- elementary: Basic techniques (u-substitution, trig identities, basic integration)
- intermediate: Multiple techniques, integration by parts, special functions
- advanced: Polylogarithms, contour integration, non-elementary integrals

───────────────────────────────────────────────────────────────────

QUERY INTEGRAL: {selected_instance.sympy_integrand}

INTEGRAL TYPE: {selected_instance.integral_type}

SIMILAR INTEGRAL FROM MATH STACKEXCHANGE:

[Answer - Score: {score}, Accepted: {'Yes' if is_accepted else 'No'}]
Author: {author}
Question ID: {question_id}
Answer ID: {answer_id}
Source: https://math.stackexchange.com/a/{answer_id}

{body_text}

───────────────────────────────────────────────────────────────────

Based on the above MSE answer, please extract the solution approach for the query integral.
Remember: EXTRACT from the answer, do NOT generate new content.
"""

    # print the prompt
    print("="*70)
    print("TEST RAG PROMPT (copy everything below this line)")
    print("="*70)
    print(prompt)
    print()
    print("="*70)
    print("METADATA (for reference, don't paste this)")
    print("="*70)
    print(f"Integrand: {selected_group.integrand_canonical}")
    print(f"Integrand Hash: {selected_group.integrand_hash}")
    print(f"Instance ID: {selected_instance.id}")
    print(f"MSE Question: https://math.stackexchange.com/q/{question_id}")
    print(f"MSE Answer: https://math.stackexchange.com/a/{answer_id}")
    print(f"Answer blob path: {blob_path}")


if __name__ == '__main__':
    main()
