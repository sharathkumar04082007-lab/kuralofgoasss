from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class SpeechToTextProvider(ABC):
    """Abstract interface for Speech-to-Text service providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the STT provider."""
        pass

    @abstractmethod
    def transcribe(
        self, 
        audio_bytes: bytes, 
        language_code: Optional[str] = None,
        filename: str = "audio.wav"
    ) -> str:
        """
        Transcribe raw audio bytes to text.
        Raises ValueError or RuntimeError on fatal failures.
        """
        pass
