"""Jarvis brain server package.

Runs on the RTX 5090 machine: faster-whisper STT, Ollama LLM with native tool
calling and Silero TTS on CPU, exposed over a WebSocket at ``/ws`` (see SPEC §3/§4).

Entry point: ``python -m server.main`` from the repository root.
"""

__all__ = ["app", "llm", "main", "session", "stt", "tools", "tts"]
