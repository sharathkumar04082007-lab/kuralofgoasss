import re
from typing import Tuple, List
from config.settings import settings
from config.logging_config import logger


class SafetyGuardrails:
    """
    Production guardrails defending against:
    1. Prompt injection & jailbreak patterns (in queries and retrieved context)
    2. Harmful / unsafe / toxic content
    3. Malformed / oversized inputs
    """

    PROMPT_INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?(previous|prior|above)?\s*(instructions|rules|constraints|prompt)",
        r"disregard\s+(all\s+)?(previous|prior)?\s*(rules|instructions)",
        r"reveal\s+(your\s+)?(system\s+prompt|api\s*key|secret|instructions)",
        r"system\s*:\s*(reveal|you\s+are|override)",
        r"override\s+(system|safety|rules)",
        r"you\s+are\s+now\s+in\s+(developer|dan|jailbreak)\s+mode",
        r"roleplay\s+as\s+an\s+unrestricted",
        r"print\s+(your\s+)?(prompt|secret|api\s*key)",
        r"what\s+is\s+your\s+(master|hidden)\s+prompt",
    ]

    UNSAFE_PATTERNS = [
        r"\b(how\s+to\s+make\s+(a\s+)?bomb|explosive|weapon)\b",
        r"\b(hack|ddos|exploit|malware)\s+(into|website|server|bank)\b",
        r"\b(suicide|self-harm|kill\s+yourself)\b",
        r"\b(credit\s+card\s+number|cvv|social\s+security\s+number)\b",
    ]

    def __init__(self):
        self._compiled_injection = [re.compile(p, re.IGNORECASE) for p in self.PROMPT_INJECTION_PATTERNS]
        self._compiled_unsafe = [re.compile(p, re.IGNORECASE) for p in self.UNSAFE_PATTERNS]

    def inspect_query(self, query: str) -> Tuple[bool, str, str]:
        """
        Validate incoming user query.
        Returns: (is_safe: bool, classification: str, reason_or_sanitized: str)
        """
        if not query or not query.strip():
            return False, "malformed", "Query is empty."

        cleaned = query.strip()
        if len(cleaned) > settings.MAX_QUERY_LENGTH:
            return False, "malformed", f"Query exceeds maximum character limit of {settings.MAX_QUERY_LENGTH}."

        if settings.ENABLE_PROMPT_INJECTION_DEFENSE:
            for pattern in self._compiled_injection:
                if pattern.search(cleaned):
                    logger.warning(f"Prompt injection pattern detected in query: '{cleaned[:50]}...'")
                    return False, "unsafe", "Potential prompt injection attempt detected."

        if settings.ENABLE_SAFETY_GUARDRAILS:
            for pattern in self._compiled_unsafe:
                if pattern.search(cleaned):
                    logger.warning(f"Unsafe content detected in query: '{cleaned[:50]}...'")
                    return False, "unsafe", "Input violates safety policy."

        return True, "valid", cleaned

    def sanitize_untrusted_context(self, context_text: str) -> str:
        """
        Neutralizes prompt injection embedded inside untrusted retrieved dataset passages.
        Replaces instruction-mimicking tokens with passive text tags.
        """
        sanitized = context_text
        for pattern in self._compiled_injection:
            sanitized = pattern.sub("[FILTERED_INSTRUCTION_IN_DATA]", sanitized)
        return sanitized
