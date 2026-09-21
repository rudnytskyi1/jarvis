"""Small billed model smoke check using synthetic text/tools; no PC actions.

Uses the configured monthly ledger. Supply OPENAI_API_KEY via the saved-key
launcher/environment; never pass the key as a command-line argument.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.config import LLMConfig, load_config
from hub.openai_responses import ResponsesClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=str(ROOT / 'config.openai.yaml'))
    parser.add_argument('--model')
    args = parser.parse_args()
    cfg = load_config(args.config).server.llm.model_dump()
    if args.model:
        cfg['model'] = args.model
    client = ResponsesClient(LLMConfig.model_validate(cfg))
    started = time.perf_counter()
    before = client.budget.status()['accounted_usd']
    try:
        text, calls = client.complete([{'role': 'user', 'content': 'Reply with exactly ROWAN_READY.'}], [])
        assert text.strip() == 'ROWAN_READY' and not calls, 'Text smoke failed'
        tools = [{'type': 'function', 'function': {
            'name': 'room_status', 'description': 'Get synthetic test room status.',
            'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False}}}]
        messages = [{'role': 'user', 'content': 'Use room_status to check whether the test room is online, then briefly report its status.'}]
        text, calls = client.complete(messages, tools)
        assert len(calls) == 1 and calls[0]['function']['name'] == 'room_status', 'Tool smoke failed'
        assert json.loads(calls[0]['function']['arguments']) == {}, 'Unexpected tool arguments'
        messages.extend([{'role': 'assistant', 'content': text, 'tool_calls': calls},
                         {'role': 'tool', 'tool_call_id': calls[0]['id'], 'content': '{"online":true}'}])
        text, calls = client.complete(messages, [])
        assert text.strip() and not calls and 'online' in text.lower(), 'Tool result smoke failed'
        print(json.dumps({'model': client.model, 'text_ok': True, 'tool_roundtrip_ok': True,
                          'elapsed_s': round(time.perf_counter() - started, 2),
                          'accounted_cost_usd': round(client.budget.status()['accounted_usd'] - before, 6)}))
    finally:
        client.close()


if __name__ == '__main__':
    main()
