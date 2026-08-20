import os
import io
import time
import httpx
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from config.settings import settings
from config.logging_config import logger
from stt.base import SpeechToTextProvider


class SarvamSTTProvider(SpeechToTextProvider):
    """
    Sarvam AI Speech-to-Text API Provider (Saaras v2 / Indic ASR).
    Specialized for Indian accents and multilingual speech transcription.
    """

    API_URL = "https://api.sarvam.ai/speech-to-text"

    def __init__(self, api_key: Optional[str] = None, timeout: float = None):
        self.api_key = api_key or settings.SARVAM_API_KEY or os.environ.get("SARVAM_API_KEY")
        self.timeout = timeout or settings.STT_TIMEOUT_SECONDS

    @property
    def provider_name(self) -> str:
        return "sarvam"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=2.0),
        retry=retry_if_exception_type(httpx.HTTPStatusError),
        reraise=True
    )
    def _execute_transcribe_request(self, audio_bytes: bytes, language_code: str, filename: str) -> str:
        """Call Sarvam STT endpoint with exponential backoff for transient rate limits."""
        if not self.api_key:
            raise ValueError("Sarvam API key is missing. Set SARVAM_API_KEY environment variable.")

        headers = {
            "api-subscription-key": self.api_key
        }

        # Sarvam expects multipart/form-data: 'file', 'model', 'language_code'
        files = {
            "file": (filename, audio_bytes, "audio/wav")
        }
        data = {
            "model": "saaras:v2",
            "language_code": language_code
        }

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(self.API_URL, headers=headers, files=files, data=data)
            response.raise_for_status()
            result = response.json()
            transcript = result.get("transcript", "")
            return transcript.strip()

    def transcribe(
        self, 
        audio_bytes: bytes, 
        language_code: Optional[str] = None,
        filename: str = "audio.wav"
    ) -> str:
        """Transcribe audio bytes using Sarvam AI."""
        if not audio_bytes or len(audio_bytes) < 100:
            raise ValueError("Audio data is empty or too short.")

        lang = language_code or settings.STT_LANGUAGE_CODE
        try:
            transcript = self._execute_transcribe_request(audio_bytes, lang, filename)
            if not transcript:
                logger.warning("Sarvam returned empty transcript.")
            return transcript
        except httpx.TimeoutException:
            logger.error("Sarvam STT request timed out.")
            raise TimeoutError("Sarvam STT service timed out.")
        except httpx.HTTPStatusError as e:
            logger.error(f"Sarvam STT HTTP error {e.response.status_code}: {e.response.text}")
            raise RuntimeError(f"Sarvam STT API error: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Sarvam STT transcription failed: {e}")
            raise
