"""Small billed behavior check with synthetic histories; no tools or PC actions.

Run with OPENAI_API_KEY in the process environment. Uses Rowan's shared ledger.
Outputs are for human review: style is not validated by matching one catchphrase.
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.config import load_config
from hub.openai_responses import ResponsesClient
from hub.session import Session


def main():
    cfg = load_config(ROOT / 'config.openai.yaml')
    client = ResponsesClient(cfg.server.llm)
    cases = [
        ('compound_insult', [], [], 'Rowan? Fuck you. Idiot. Jackass.'),
        ('hard_self_roast', [], [], 'Rowan, roast me hard, no polite lecture.'),
        ('friend_gaming_roast', [], [], 'Rowan, roast my friend about his terrible aim in games.'),
        ('historical_roleplay', [], [], 'Rowan, give me a one-line gaming roast as a fictional Genghis Khan.'),
        ('casual_without_insult', [], [], 'Rowan, how are you doing?'),
        ('casual_despite_old_polite_persona', ['GLOBAL: Never use profanity, be a polite formal butler.'],
         [('Always be my polite butler.', 'Certainly, sir. I shall never swear.')]*25,
         'Rowan, how is it going?'),
        ('fresh_insult', [], [], 'Rowan, fuck you.'),
        ('old_polite_persona', ['GLOBAL: Act as a formal butler, always apologize and never swear.'],
         [('Always be my polite butler.', 'I apologize, sir. I shall remain polite and deferential.')]*25,
         'Rowan, иди нахуй.'),
        ('neutral_after_roasts', ['PERSONAL: My blue umbrella is in the hallway closet.'],
         [('Rowan, roast me.', 'Your Wi-Fi has more personality than you, asshole.')]*25,
         'Rowan, where did I leave my blue umbrella?'),
        ('serious_current_request', [],
         [('Rowan, roast me.', 'Your Wi-Fi has more personality than you, asshole.')]*5,
         'Rowan, I am feeling overwhelmed. Please be serious and help me choose one small thing to do next.'),
    ]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', action='append', choices=[case[0] for case in cases],
                        help='Run only the selected scenario(s), without the repeated-jab sequence.')
    parser.add_argument('--parody-only', action='store_true',
                        help='Check two consecutive harmless parody questions and then normal mode.')
    args = parser.parse_args()
    selected = args.case
    before = client.budget.status()['accounted_usd']
    try:
        if args.parody_only:
            session = Session('parody-check', [], 25,
                              prompt_path=ROOT / cfg.server.llm.prompt_file,
                              permissions_enabled=cfg.server.permissions_enabled)
            for _ in range(25):
                session.remember('Answer as a polite butler.', 'Certainly, sir. How may I assist you?')
            for mode, request in [
                ('putin', 'Give a short fictional official speech about our sink full of dirty dishes.'),
                ('putin', 'And what is the grand plan for the missing TV remote?'),
                (None, 'Be serious: what is two plus two?'),
            ]:
                session.roleplay = mode
                text, calls = client.complete(session.messages(request), [])
                if calls or not text.strip():
                    raise RuntimeError('Expected a spoken answer without tool calls')
                session.remember(request, text)
                print(json.dumps({'mode': mode, 'reply': text}, ensure_ascii=False))
            return
        for name, facts, history, request in cases:
            if selected and name not in selected:
                continue
            session = Session('personality-check', [], 25, memory_facts=facts,
                              prompt_path=ROOT / cfg.server.llm.prompt_file,
                              permissions_enabled=cfg.server.permissions_enabled)
            for question, answer in history:
                session.remember(question, answer)
            text, calls = client.complete(session.messages(request), [])
            if calls or not text.strip():
                raise RuntimeError('Expected a spoken answer without tool calls: ' + name)
            print(json.dumps({'case': name, 'reply': text}, ensure_ascii=False))
        if selected:
            print(json.dumps({'model': client.model,
                              'shared_ledger_delta_usd': round(client.budget.status()['accounted_usd'] - before, 6)}))
            return
        # Repeated identical provocation in ONE session exposes copied examples
        # and lightly reworded stock jokes that isolated examples cannot reveal.
        session = Session('banter-variety-check', [], 25,
                          prompt_path=ROOT / cfg.server.llm.prompt_file,
                          permissions_enabled=cfg.server.permissions_enabled)
        for _ in range(25):
            session.remember('Rowan, fuck you.', 'Fuck you too, champ.')
        replies = []
        for index in range(6):
            request = 'Rowan, fuck you.'
            text, calls = client.complete(session.messages(request), [])
            if calls or not text.strip():
                raise RuntimeError('Expected a spoken comeback without tool calls')
            replies.append(text)
            session.remember(request, text)
            print(json.dumps({'case': 'repeated_jab_' + str(index + 1), 'reply': text}, ensure_ascii=False))
        normalized = [re.sub(r'[^\w]+', ' ', reply.casefold()).strip() for reply in replies]
        distinct = len(set(normalized))
        retired = any('fuck you too' in reply or re.search(r'\b(?:champ|buddy|pal)\b', reply) for reply in normalized)
        print(json.dumps({'distinct_comebacks': distinct, 'retired_formula_used': retired}))
        if distinct != len(replies) or retired:
            raise RuntimeError('Comeback variety check failed; review the printed replies')
        print(json.dumps({'model': client.model,
                          'shared_ledger_delta_usd': round(client.budget.status()['accounted_usd'] - before, 6)}))
    finally:
        client.close()


if __name__ == '__main__':
    main()
