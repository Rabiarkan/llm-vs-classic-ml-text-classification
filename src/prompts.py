"""
Prompt definitions. Zero-shot and few-shot share the same system prompt.

SYSTEM — FROZEN 2026-08-31 after 3 revisions on dev (n=200):
    round 1: macro-F1 0.622  (neutral F1 0.138)
    round 2: macro-F1 0.729  (neutral F1 0.400)  + star-rating framing
    round 3: macro-F1 0.763  (neutral F1 0.476)  + rejection/disappointment split

SYSTEM_FEWSHOT = SYSTEM + examples. The base string is REUSED, not copied —
copying invites a silent one-word edit, and the zero-shot/few-shot delta would
then confound prompt changes with examples. One variable, one source.

eval sets had not been opened when SYSTEM was frozen.

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

# --- Few-shot ---

FEWSHOT_VERSION = "fewshot-v1"

# Selection rationale — zero-shot errors concentrated on neutral → negative:
# the model reads a complaint and calls it rejection.
#
#   [23432] neutral  vs  [5457] negative — near-identical structure (unmet
#     expectation, measured tone, objective flaws). The ONLY difference is that
#     [5457] ends with an explicit "suggest avoiding". That is the boundary.
#   [11944] neutral — canonical "overall ... okay" verdict after weighing.
#   [22890] positive — deliberately hard: opens with "okay", prefers a competitor,
#     lists three flaws, no praise, yet the reviewer gave 4-5 stars. Teaches that
#     sentiment intensity in the text does not map cleanly onto the star rating.
#
FEWSHOT_EXAMPLES = """
Worked examples:

Review: I've tried about 3 other flavors, and this is my least favorite, even though it is usually my favorite flavor combination. They were very gummy and almost too filling, to the point that it was harder to finish these than the other flavors. It didn't have much peanut butter flavor, but the chocolate flavor was very good. Overall, these were okay but their other flavors are so much better!
JSON: {"label": "neutral", "confidence": 0.88, "reason": "Weighs flaws against one merit, settles on okay"}

Review: That was pretty much the consensus at our house. It was a bit too sweet and did not have enough carbonation. For the amount of calories that are in this can I would expect it to taste amazing but it did not.
JSON: {"label": "neutral", "confidence": 0.82, "reason": "Unmet expectation, no advice to avoid"}

Review: I bought this item thinking it might be equivilent to Starbucks or Davinci Hazelnut syrup but it's not. At first it tastes like Hazelnut then you get the flavor of cheap pancake syrup followed by an aftertaste that can best be described as tasting stale. The price is only slightly better than what I can get the Davinci Brand at Sam's club but the quality is far lower in my opinion. For hazelnut fans I suggest avoiding this particular brand.
JSON: {"label": "negative", "confidence": 0.87, "reason": "Same unmet expectation, but tells others to avoid it"}

Review: the texture and taste is okay... but i still like trader joe's brown rice pasta best as it holds up in texture a lot more and tastes good. there's a very fine line between al dente and soggy in a lot of gf pasta. although this product did not get soggy, it seemed to stick together a lot more even with oil in the water. also looked more delicate than TJ pasta... this seemed to require more supervision/watching.
JSON: {"label": "positive", "confidence": 0.70, "reason": "Critical tone but the reviewer still rated it highly"}
"""

SYSTEM_FEWSHOT = SYSTEM + "\n" + FEWSHOT_EXAMPLES