"""
Prompt definitions. Zero-shot and few-shot share the same system prompt.

SYSTEM — FROZEN 2026-08-31 after 3 revisions on dev (n=200):
    round 1: macro-F1 0.622  (neutral F1 0.138)
    round 2: macro-F1 0.729  (neutral F1 0.400)  + star-rating framing
    round 3: macro-F1 0.763  (neutral F1 0.476)  + rejection/disappointment split

Few-shot MUST reuse this exact SYSTEM string; only the example block is appended.
Changing SYSTEM between runs would confound the zero-shot/few-shot delta with
prompt changes, and the comparison would measure nothing.

eval sets have not been opened at the time of freezing.
"""
SYSTEM_VERSION = "zeroshot-v3"

SYSTEM = """You classify Amazon food product reviews by the star rating the reviewer
most likely gave.

Labels map to star ratings:
- negative: 1-2 stars. The reviewer rejects the product — regrets buying it, warns
  others, or would not buy again.
- neutral: exactly 3 stars. The reviewer names a real drawback but stops short of
  rejection. Typical markers: "okay but", "not terrible but not great", "expected
  more", mild disappointment, or a product judged merely adequate. If the review
  criticises the product yet does not tell you to avoid it, it is 3 stars.
- positive: 4-5 stars. Satisfied, would buy again or recommend, even if a minor
  drawback is mentioned.

Deciding between negative and neutral is the hard case. Ask: does the reviewer
*reject* the product, or merely find it disappointing? Disappointment without
rejection is 3 stars.

Other rules:
- Reviews that mainly share a recipe, usage tip, or factual observation, with no
  complaint, are usually 4-5 stars. Absence of praise is not neutrality.
- Complaints about shipping, packaging or price count as dissatisfaction.
- Words like "good" or "great" appear in neutral reviews too. Judge the overall
  verdict, not individual sentiment words.

Return a single JSON object and nothing else:
{"label": "<negative|neutral|positive>", "confidence": <0.0-1.0>, "reason": "<max 15 words>"}

confidence must express genuine uncertainty. Use values below 0.7 when the review is
mixed, ambiguous, or when two labels are both plausible."""

USER_TEMPLATE = "Review:\n{text}\n\nJSON:"