from typing import Optional, Dict
from stt.base import SpeechToTextProvider
from config.logging_config import logger


class MockSTTProvider(SpeechToTextProvider):
    """
    Offline/Development Speech-to-Text provider.
    Used for local testing, benchmarking, and automated unit tests without external API dependencies.
    """

    DEFAULT_TEST_TRANSCRIPT = "what is the capital of France?"

    def __init__(self, default_transcript: Optional[str] = None):
        self.default_transcript = default_transcript or self.DEFAULT_TEST_TRANSCRIPT
        self._preset_audio_map: Dict[int, str] = {}

    @property
    def provider_name(self) -> str:
        return "mock"

    def register_preset(self, audio_size: int, transcript: str) -> None:
        """Register specific transcript for exact audio payload size."""
        self._preset_audio_map[audio_size] = transcript

    def transcribe(
        self, 
        audio_bytes: bytes, 
        language_code: Optional[str] = None,
        filename: str = "audio.wav"
    ) -> str:
        if not audio_bytes or len(audio_bytes) < 10:
            raise ValueError("Audio data is empty or too short.")

        # Check registered preset
        if len(audio_bytes) in self._preset_audio_map:
            return self._preset_audio_map[len(audio_bytes)]

        logger.info(f"Mock STT transcribed {len(audio_bytes)} bytes audio -> '{self.default_transcript}'")
        return self.default_transcript
