"""Inspect exactly the uploaded Telegram photo; never substitute a live camera."""
from __future__ import annotations

import asyncio
import math

from hub.segment import draw_boxes


class PhotoInspector:
    def __init__(self, runtime):
        self.runtime = runtime

    async def __call__(self, image, args, facade):
        initial = len(facade._utterance_actions)
        try:
            return await self._inspect(image, args, facade)
        except Exception:
            result = {'ok': False, 'source': 'telegram_attachment', 'current_room_observation': False,
                      'error': 'Could not inspect the uploaded photo. Please try again.'}
            if len(facade._utterance_actions) > initial:
                facade._utterance_actions[-1]['result'] = result
            return result

    async def _inspect(self, image, args, facade):
        runtime = self.runtime()
        target = ' '.join(str(args.get('target') or '').split())[:200]
        query = ' '.join(str(args.get('query') or 'Describe this photo.').split())[:2000]
        record = {'tool': 'inspect_photo', 'args': {'query': query, 'target': target}}
        facade._utterance_actions.append(record)
        def done(result):
            record['result'] = {key: value for key, value in result.items() if key != '_annotation'}
            return result
        if target:
            segment = runtime.get('segment')
            if segment is None or not segment.enabled:
                return done({'ok': False, 'error': 'SAM3 is unavailable.'})
            await facade._make_room_for_segmentation()
            result = await asyncio.to_thread(segment.segment, image.jpeg, target)
            result = dict(result, source='telegram_attachment', current_room_observation=False)
            if result.get('ok'):
                result['_annotation'] = await asyncio.to_thread(draw_boxes, image.jpeg,
                    result.get('boxes') or [], result.get('scores') or [], 85, target)
                result['note'] = 'Detections are on the uploaded photo, not a current camera view.'
            return done(result)
        faces = []
        engine, registry = runtime.get('face'), runtime.get('voices')
        if engine is not None and registry is not None and engine.available:
            profiles = await asyncio.to_thread(registry.face_profiles)
            profiles = await facade._appearance_profiles(profiles)
            for face in await asyncio.to_thread(engine.located_faces, image.jpeg):
                name, score = engine.match(face['embedding'], profiles)
                faces.append({'name': name, 'box': face['box'],
                              'score': float(score) if math.isfinite(float(score)) else 0})
        vision = runtime.get('vision')
        if vision is None:
            return done({'ok': bool(faces), 'faces': faces, 'source': 'telegram_attachment',
                         'error': 'Image description is unavailable; only face matches could be checked.'})
        await facade._make_room_for_vision()
        answer = await vision.describe_screenshot(image.jpeg,
            'This is a user-uploaded photo, not the current room camera or computer screen. '
            'Describe visible content; do not infer anyone\'s identity. Question: ' + query)
        if not answer or answer.startswith('Screen check failed:'):
            return done({'ok': False, 'faces': faces, 'source': 'telegram_attachment',
                         'current_room_observation': False, 'error': 'Image description is unavailable. Please try again.'})
        return done({'ok': True, 'answer': answer, 'faces': faces, 'source': 'telegram_attachment',
                     'current_room_observation': False,
                     'note': 'Names may only come from the saved face matches. Unknown faces stay unknown. '
                             'This uploaded image is not evidence that someone is currently in the room.'})
