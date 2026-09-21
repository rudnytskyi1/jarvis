"""Reviewed standard text rates, USD per million tokens (2026-09-18).

Sources: https://developers.openai.com/api/docs/models/gpt-5.4
         https://developers.openai.com/api/docs/models/gpt-5.4-mini
         https://developers.openai.com/api/docs/models/gpt-5.6-luna
         https://developers.openai.com/api/docs/guides/prompt-caching
Cache discounts require explicit usage details. The input byte cap keeps requests below
the 272K-token long-context pricing threshold.
"""

OPENAI_TEXT_RATES = {
    'gpt-5.4-mini': ('0.75', '4.50'),
    'gpt-5.4': ('2.50', '15.00'),
    'gpt-5.6-luna': ('0.20', '1.20'),
}

# GPT-5.6 cache writes cost more than ordinary input. Without usage details,
# reserve/account all input at the write rate; never assume a cache hit.
OPENAI_CACHE_RATES = {'gpt-5.6-luna': ('0.02', '0.25')}

# Reviewed 2026-09-20. GPT-5.4 models have no separate cache-write rate.
# Only API-reported hits receive a discount; reservations assume uncached input.
OPENAI_CACHED_INPUT_RATES = {'gpt-5.4-mini': '0.075', 'gpt-5.4': '0.25', 'gpt-5.6-luna': '0.02'}
