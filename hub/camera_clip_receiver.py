"""Bounded request/response handling for camera clips on the existing socket."""
from __future__ import annotations

import asyncio

from common import protocol as proto


class CameraClipReceiver:
    async def _request_camera_clip(self, identifier, seconds=5, fps=8):
        if not getattr(self, '_can_camera_clip', False):
            return 'The room client does not support short video clips.'
        if (type(seconds) not in (int, float) or not 3 <= seconds <= 10
                or type(fps) is not int or not 5 <= fps <= 10):
            return 'Invalid clip length or frame rate.'
        if getattr(self, '_clip_future', None) is not None:
            return 'Another clip is already being captured.'
        future = asyncio.get_running_loop().create_future()
        self._clip_future, self._clip_id = future, identifier
        try:
            await self.send_json({'type': proto.MSG_CAMERA_CLIP_REQUEST, 'id': identifier,
                                  'seconds': seconds, 'fps': fps})
            return await asyncio.wait_for(future, timeout=seconds + 25)
        except TimeoutError:
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
        future.set_result(bytes(data) if valid else 'The camera returned an invalid video clip.')

    def _on_clip_error(self, payload):
        future = getattr(self, '_clip_future', None)
        if (future is not None and not future.done()
                and payload.get('id') == getattr(self, '_clip_id', None)):
            future.set_result('Camera clip capture failed.')

    def _close_camera_clip(self):
        future = getattr(self, '_clip_future', None)
        if future is not None and not future.done():
            future.set_result('Room client disconnected.')
