"""Bounded request/response handling for camera clips on the existing socket."""
from __future__ import annotations

import asyncio
import logging

from common import protocol as proto
from common.ids import new_ulid
from hub.camera_events import KIND_CLIP, record_event

log = logging.getLogger("jarvis.server.app")


class CameraClipReceiver:
    async def _request_camera_clip(self, identifier, seconds=5, fps=8, names=None,
                                   preroll=None):
        """Ask the room for a short video, with the names of the tracks it holds.

        Владелец 2026-09-24: «можешь чтобы оно с bounding box видео записывало и
        идентификацией человека над ним (label)». The room draws the boxes (only
        it has a box per frame), and the hub tells it which track it already
        recognised, so a name the identity layer confirmed shows up on the
        video. A track nobody was identified in is simply absent from the map:
        the room then draws its box without a name.
        """
        if not getattr(self, '_can_camera_clip', False):
            return 'The room client does not support short video clips.'
        # ТЗ F-702: one video of an alert episode may be up to a minute long.
        if (type(seconds) not in (int, float) or not 3 <= seconds <= 60
                or type(fps) is not int or not 5 <= fps <= 10):
            return 'Invalid clip length or frame rate.'
        if getattr(self, '_clip_future', None) is not None:
            return 'Another clip is already being captured.'
        future = asyncio.get_running_loop().create_future()
        self._clip_future, self._clip_id = future, identifier
        # ТЗ 4.5: a clip is a background camera event and owns an event id.
        self._clip_event_id = new_ulid()
        labels = {}
        for key, value in (names or {}).items():
            if isinstance(key, str) and isinstance(value, str) and key and value:
                labels[key[:100]] = value[:120]
            if len(labels) >= 12:
                break
        try:
            request = {'type': proto.MSG_CAMERA_CLIP_REQUEST, 'id': identifier,
                       'event_id': self._clip_event_id,
                       'seconds': seconds, 'fps': fps}
            if labels:
                request['names'] = labels
            if preroll is not None:
                # Владелец 2026-09-24: the first video of a visit starts in the
                # past (the client's pre-roll); the next parts of the same visit
                # do not need those seconds again, so the hub asks for none.
                request['preroll'] = max(0.0, min(10.0, float(preroll)))
            await self.send_json(request)
            return await asyncio.wait_for(future, timeout=seconds + 25)
        except TimeoutError:
            record_event(self._clip_event_id, kind=KIND_CLIP, home_id=getattr(self, 'home_id', ''),
                         source='camera_clip', frames=0, ok=False, detail='timeout')
            return 'Camera clip capture timed out.'
        finally:
            self._clip_future, self._clip_id = None, None

    def _on_clip_header(self, payload):
        # Always consume the next binary frame, even for a stale/invalid header;
        # a late clip must never become microphone audio after cancellation.
        self._expect_clip = True
        self._clip_header = payload

    def _on_clip_binary(self, data):
        self._expect_clip = False
        header = getattr(self, '_clip_header', {})
        self._clip_header = {}
        future = getattr(self, '_clip_future', None)
        if future is None or future.done() or header.get('id') != getattr(self, '_clip_id', None):
            return
        valid = (header.get('format') == 'mp4' and type(header.get('bytes')) is int
                 and header['bytes'] == len(data) and 12 <= len(data) <= proto.CAMERA_CLIP_MAX_BYTES
                 and data[4:8] == b'ftyp')
        event_id = str(header.get('event_id') or getattr(self, '_clip_event_id', '') or '')[:100]
        if event_id:
            record_event(event_id, kind=KIND_CLIP, home_id=getattr(self, 'home_id', ''),
                         source='camera_clip', ok=valid,
                         detail='' if valid else 'invalid clip payload')
            log.info("Camera clip event %s received (%d bytes)", event_id, len(data),
                     extra={'event_id': event_id})
        future.set_result(bytes(data) if valid else 'The camera returned an invalid video clip.')

    def _on_clip_error(self, payload):
        future = getattr(self, '_clip_future', None)
        if (future is not None and not future.done()
                and payload.get('id') == getattr(self, '_clip_id', None)):
            event_id = str(payload.get('event_id') or getattr(self, '_clip_event_id', '') or '')[:100]
            if event_id:
                record_event(event_id, kind=KIND_CLIP, home_id=getattr(self, 'home_id', ''),
                             source='camera_clip', frames=0, ok=False,
                             detail=str(payload.get('error') or 'capture failed')[:200])
            future.set_result('Camera clip capture failed.')

    def _close_camera_clip(self):
        future = getattr(self, '_clip_future', None)
        if future is not None and not future.done():
            future.set_result('Room client disconnected.')
