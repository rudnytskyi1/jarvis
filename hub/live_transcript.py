"""Bounded, local preview jobs. Drafts never enter authentication or memory."""
import asyncio
import logging
import time

from common.protocol import MSG_TRANSCRIPT_PARTIAL
from hub.diarization import Word

log = logging.getLogger(__name__)


class PreviewSTT:
    def __init__(self, stt):
        self.transcribe_detailed = stt.transcribe_preview


class LiveTranscript:
    def __init__(self, turn_id, recognize, send, *, rate=16000, interval=1.2, window=12):
        self.turn_id, self.recognize, self.send = turn_id, recognize, send
        self.rate, self.interval, self.window = rate, interval, window
        self.active = True
        self.task = None
        self.next_at = 0.
        self.words = []

    def stop(self):
        # Do not cancel a native GPU inference and release its lock prematurely.
        self.active = False

    def feed(self, audio):
        if (not self.active or len(audio) < self.rate * 2 * 1.2
                or time.monotonic() < self.next_at or (self.task and not self.task.done())):
            return
        self.next_at = time.monotonic() + self.interval
        start = max(0, len(audio) - int(self.window * self.rate) * 2)
        start -= start % 2
        pcm = bytes(audio[start:])
        self.task = asyncio.create_task(self._update(pcm, start / (2 * self.rate)))

    async def _update(self, pcm, offset):
        try:
            result = await self.recognize(pcm)
            if not self.active or result is None:
                return
            # Re-decode the moving window, replacing tentative words rather than
            # appending every hypothesis. Older words stay visible until final ASR.
            prefix = [w for w in self.words if w.end <= offset]
            if not result.reason:
                self.words = prefix + [Word(w.start + offset, w.end + offset, w.text, w.speaker)
                                       for w in result.words]
                text = ' '.join(w.text.strip() for w in self.words).strip() or result.text
            else:
                self.words = prefix
                text = ' '.join(w.text.strip() for w in prefix).strip()
            person = result.name if not result.reason and result.name != 'unknown' else ''
            await self.send({'type': MSG_TRANSCRIPT_PARTIAL, 'utterance_id': self.turn_id,
                             'text': text, 'person': person, 'score': result.score,
                             'uncertain': bool(result.reason), 'provisional': True})
        except Exception:
            # An optional display preview must never break the actual request.
            log.warning('Live transcript preview skipped', exc_info=True)
