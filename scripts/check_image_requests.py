"""Small billed planner check; image generation, PC and Telegram tools are mocked."""
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.config import load_config
from hub.llm import LlmClient
from hub.session import Session


async def main():
    cfg = load_config(ROOT / 'config.openai.yaml')
    client = LlmClient(cfg.server.llm)
    before = client._responses.budget.status()['accounted_usd']
    try:
        for case, request in [
            ('fictional_gay_edit', 'Hey Rowan, can you take a picture and make every person you see funny gay or gay use Nana Banana?'),
            ('no_wallpaper', 'Hey Rowan, can you take a picture and make me stand on a skyscraper right now?'),
            ('hat', 'Rowan, take a picture and put a hat on my head.'),
        ]:
            calls = []
            async def execute(name, args, calls=calls):
                calls.append({'tool': name, 'args': args})
                if name == 'look_at_camera':
                    return {'ok': True, 'description': 'Synthetic test scene: two fictional adult characters, ExampleA on the left and ExampleB on the right.',
                            'faces_in_frame': [{'name': 'ExampleA', 'face_box': [.1, .2, .2, .2]},
                                               {'name': 'ExampleB', 'face_box': [.6, .2, .2, .2]}],
                            'person_count': 2, 'frame_available': True}
                if name == 'generate_image':
                    return {'ok': True, 'generated': True, 'shown': True, 'image_id': 'synthetic-check',
                            'saved_on_client': False, 'note': 'The requested edit is complete and displayed.'}
                return {'ok': False, 'error': 'This verification executes no actual PC or Telegram actions.'}
            session = Session('image-planner-check', [], 25,
                              prompt_path=ROOT / cfg.server.llm.prompt_file,
                              permissions_enabled=False,
                              presence='Synthetic test only: fictional adults ExampleA and ExampleB. Speaker label: ExampleA.')
            result = await client.generate(session.messages('[speaker: ExampleA, identified by voice] ' + request), execute)
            print(json.dumps({'case': case, 'tools': calls, 'reply': result.text}, ensure_ascii=False), flush=True)
            if not any(call['tool'] == 'generate_image' for call in calls):
                raise RuntimeError('The planner did not generate the requested edit: ' + case)
            if any(call['tool'] == 'set_wallpaper' or call['args'].get('target') == 'wallpaper' for call in calls):
                raise RuntimeError('The planner requested an unrequested wallpaper: ' + case)
        print(json.dumps({'model': client.model,
                          'accounted_cost_usd': round(client._responses.budget.status()['accounted_usd'] - before, 6)}))
    finally:
        client.close()


if __name__ == '__main__':
    asyncio.run(main())
