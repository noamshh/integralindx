#!/usr/bin/env python3
"""Generate a test RAG prompt with a real MSE answer."""
import json
import re
from pathlib import Path

def html_to_text(html):
    """simple HTML to text conversion"""
    text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
    text = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL)
    text = re.sub(r'<br\s*/?>', '\n', text)
    text = re.sub(r'</p>', '\n\n', text)
    text = re.sub(r'</div>', '\n', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = text.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
    text = text.replace('&#39;', "'").replace('&quot;', '"')
    return text.strip()

# use answer_102033.json (the one we inspected earlier)
answer_id = 102033
blob_path = Path(f'data/raw/answer_blobs/answer_{answer_id}.json')

with open(blob_path, 'r', encoding='utf-8') as f:
    blob_data = json.load(f)

answer_item = blob_data['items'][0]

body_html = answer_item.get('body', '')
score = answer_item.get('score', 0)
is_accepted = answer_item.get('is_accepted', False)
author = answer_item.get('owner', {}).get('display_name', 'Unknown')
question_id = answer_item.get('question_id', 0)

body_text = html_to_text(body_html)

# construct prompt
query_integral = "1/(1 + exp(-k*x))"

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

QUERY INTEGRAL: ∫ {query_integral} dx

INTEGRAL TYPE: indefinite

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

print("="*70)
print("TEST RAG PROMPT")
print("Copy everything below and paste into ChatGPT/Claude")
print("="*70)
print()
print(prompt)
print()
print("="*70)
