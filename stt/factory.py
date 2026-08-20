from typing import Optional
from config.settings import settings
from config.logging_config import logger
from stt.base import SpeechToTextProvider
from stt.sarvam_provider import SarvamSTTProvider
from stt.elevenlabs_provider import ElevenLabsSTTProvider
from stt.mock_provider import MockSTTProvider


class STTProviderFactory:
    """Factory to instantiate configured STT provider."""

    @staticmethod
    def get_provider(provider_name: Optional[str] = None) -> SpeechToTextProvider:
        name = (provider_name or settings.STT_PROVIDER).lower()
        
        if name == "sarvam":
            if settings.SARVAM_API_KEY:
                return SarvamSTTProvider()
            else:
                logger.warning("SARVAM_API_KEY is not set. Using MockSTTProvider as fallback.")
                return MockSTTProvider()
        elif name == "elevenlabs":
            if settings.ELEVENLABS_API_KEY:
                return ElevenLabsSTTProvider()
            else:
                logger.warning("ELEVENLABS_API_KEY is not set. Using MockSTTProvider as fallback.")
                return MockSTTProvider()
        elif name == "mock":
            return MockSTTProvider()
        else:
            logger.warning(f"Unknown STT provider '{name}'. Defaulting to MockSTTProvider.")
            return MockSTTProvider()
