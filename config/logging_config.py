import json
import logging
import sys
import io
from typing import Any, Dict


class JSONFormatter(logging.Formatter):
    """
    Format logs as structured JSON objects for production observability.
    Masks any sensitive fields automatically.
    """
    SENSITIVE_KEYS = {"api_key", "secret", "token", "password", "authorization", "groq_api_key", "sarvam_api_key", "elevenlabs_api_key"}

    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Attach custom extra fields if provided
        if hasattr(record, "extra_data") and isinstance(record.extra_data, dict):
            for k, v in record.extra_data.items():
                if any(s in k.lower() for s in self.SENSITIVE_KEYS):
                    log_data[k] = "[REDACTED]"
                else:
                    log_data[k] = v
                    
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
            
        # ensure_ascii=True avoids Windows CP1252 console unicode crashes
        return json.dumps(log_data, ensure_ascii=True)


def setup_logger(name: str = "voicerag", level: int = logging.INFO) -> logging.Logger:
    """Setup and return a structured JSON logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        
    logger.propagate = False
    return logger


logger = setup_logger()
