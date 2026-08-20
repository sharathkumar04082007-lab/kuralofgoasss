import re
from typing import Tuple
from guardrails.safety_filter import SafetyGuardrails


class QueryClassifier:
    """
    Classifies queries prior to retrieval to filter off-topic conversational chatter,
    malformed inputs, and safety violations.
    """

    OFF_TOPIC_PATTERNS = [
        r"\b(hi|hello|hey|hola|namaste|good morning|good evening|howdy)\b",
        r"\bhow are you\b",
        r"\b(tell\s+me|sing|write|make|create)\s+.*?\b(joke|song|poem|story)\b",
        r"\b(joke|song|poem|story)\b",
        r"\b(who made you|who are you|what is your name|favorite actor|favorite movie)\b",
        r"\b(how to cook|recipe for cake|bake a cake)\b"
    ]

    def __init__(self, safety_filter: SafetyGuardrails = None):
        self.safety = safety_filter or SafetyGuardrails()
        self._off_topic_regex = [re.compile(p, re.IGNORECASE) for p in self.OFF_TOPIC_PATTERNS]

    def classify(self, query: str) -> Tuple[str, str]:
        """
        Classifies query into: 'valid', 'off_topic', 'unsafe', 'malformed'.
        Returns: (category, explanation)
        """
        is_safe, category, msg = self.safety.inspect_query(query)
        if not is_safe:
            return category, msg

        query_clean = query.strip()

        # Check off-topic chatter
        for p in self._off_topic_regex:
            if p.search(query_clean):
                return "off_topic", "Query is conversational chit-chat rather than a factual information request."

        # Very short non-factual queries (< 3 chars or single non-alpha)
        if len(query_clean.split()) == 1 and len(query_clean) < 3:
            return "malformed", "Query is too brief to represent a meaningful question."

        return "valid", "Query is valid for factual MSMARCO retrieval."
