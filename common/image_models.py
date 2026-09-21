"""Reviewed Gemini image pricing (USD per million tokens, 2026-09-18).

https://ai.google.dev/gemini-api/docs/pricing#gemini-3.1-flash-image
Separate from OPENAI_TEXT_RATES: these models cannot use the text transport.
"""

IMAGE_MODEL_RATES = {'gemini-3.1-flash-image': ('0.50', '3.00', '60.00')}
