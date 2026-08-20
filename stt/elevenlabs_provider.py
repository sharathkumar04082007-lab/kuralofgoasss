import os
import httpx
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from config.settings import settings
from config.logging_config import logger
from stt.base import SpeechToTextProvider


class ElevenLabsSTTProvider(SpeechToTextProvider):
    """
    ElevenLabs Speech-to-Text API Provider (Scribe v1).
    """

    API_URL = "https://api.elevenlabs.io/v1/speech-to-text"

    def __init__(self, api_key: Optional[str] = None, timeout: float = None):
        self.api_key = api_key or settings.ELEVENLABS_API_KEY or os.environ.get("ELEVENLABS_API_KEY")
        self.timeout = timeout or settings.STT_TIMEOUT_SECONDS

    @property
    def provider_name(self) -> str:
        return "elevenlabs"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=2.0),
        retry=retry_if_exception_type(httpx.HTTPStatusError),
        reraise=True
    )
    def _execute_transcribe_request(self, audio_bytes: bytes, filename: str) -> str:
        if not self.api_key:
            raise ValueError("ElevenLabs API key is missing. Set ELEVENLABS_API_KEY environment variable.")

        headers = {
            "xi-api-key": self.api_key
        }
        files = {
            "file": (filename, audio_bytes, "audio/wav")
        }
        data = {
            "model_id": "scribe_v1"
        }

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(self.API_URL, headers=headers, files=files, data=data)
            response.raise_for_status()
            result = response.json()
            transcript = result.get("text", "")
            return transcript.strip()

    def transcribe(
        self, 
        audio_bytes: bytes, 
        language_code: Optional[str] = None,
        filename: str = "audio.wav"
    ) -> str:
        if not audio_bytes or len(audio_bytes) < 100:
            raise ValueError("Audio data is empty or too short.")

        try:
            return self._execute_transcribe_request(audio_bytes, filename)
        except Exception as e:
            logger.error(f"ElevenLabs STT error: {e}")
            raise
