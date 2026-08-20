import os
import re
from typing import List, Optional
from config.settings import settings
from config.logging_config import logger
from pipeline.schemas import SourceMetadata
from generation.base import BaseGenerator
from generation.prompt_builder import PromptBuilder


class LLMAnswerGenerator(BaseGenerator):
    """
    Synthesizes concise, grounded factual answers using configured LLM provider
    (Groq, Gemini, Local Extractive engine, or Mock).
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        prompt_builder: Optional[PromptBuilder] = None
    ):
        self.provider = (provider or settings.LLM_PROVIDER).lower()
        self.model_name = model_name or settings.LLM_MODEL
        self.prompt_builder = prompt_builder or PromptBuilder()
        self._groq_client = None
        self._gemini_client = None

        if self.provider == "groq":
            self._init_groq()
        elif self.provider == "gemini":
            self._init_gemini()

    def _init_groq(self) -> None:
        try:
            from groq import Groq
            api_key = settings.GROQ_API_KEY or os.environ.get("GROQ_API_KEY")
            if api_key:
                self._groq_client = Groq(api_key=api_key)
                logger.info(f"Initialized Groq client for model {self.model_name}")
            else:
                logger.warning("GROQ_API_KEY not found. Fallback to local extractive generator.")
        except Exception as e:
            logger.warning(f"Failed to initialize Groq: {e}")

    def _init_gemini(self) -> None:
        try:
            from google import genai
            api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
            if api_key:
                self._gemini_client = genai.Client(api_key=api_key)
                logger.info(f"Initialized Gemini client.")
            else:
                logger.warning("GEMINI_API_KEY not found. Fallback to local generator.")
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini: {e}")

    def _local_extractive_generation(self, query: str, sources: List[SourceMetadata]) -> str:
        """
        Fast, deterministic local factual answer generator.
        Extracts and composes the most salient factual sentence answering the query from context.
        Latency: < 2 ms.
        """
        if not sources:
            return "I couldn't find enough information in the retrieved data to answer that reliably."

        # Scan retrieved chunks for sentences with highest query overlap
        query_words = set(re.findall(r'\w+', query.lower()))
        best_sentence = ""
        best_overlap = -1

        for src in sources:
            # Clean structural prefixes
            text = re.sub(r'\[.*?\]', '', src.text_excerpt).strip()
            sentences = re.split(r'(?<=[.?!।॥\n])\s+', text)
            for sent in sentences:
                s_clean = sent.strip()
                if len(s_clean) < 15:
                    continue
                s_words = set(re.findall(r'\w+', s_clean.lower()))
                overlap = len(query_words.intersection(s_words))
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_sentence = s_clean

        if best_sentence and best_overlap >= 1:
            return best_sentence
        elif sources:
            # Return first sentence of highest-ranked source
            first_text = re.sub(r'\[.*?\]', '', sources[0].text_excerpt).strip()
            first_sent = re.split(r'(?<=[.?!।॥\n])\s+', first_text)[0].strip()
            return first_sent if len(first_sent) > 10 else first_text[:200]

        return "I couldn't find enough information in the retrieved data to answer that reliably."

    def generate_answer(self, query: str, sources: List[SourceMetadata]) -> str:
        """Synthesize answer using configured LLM or fast local generator."""
        if not sources or len(sources) == 0:
            return "I couldn't find enough information in the retrieved data to answer that reliably."

        user_prompt = self.prompt_builder.build_prompt(query, sources)

        # 1. Try Groq (Ultra-low latency cloud LLM)
        if self.provider == "groq" and self._groq_client is not None:
            try:
                response = self._groq_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": PromptBuilder.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=settings.LLM_TEMPERATURE,
                    max_tokens=settings.LLM_MAX_TOKENS,
                    timeout=settings.LLM_TIMEOUT_SECONDS
                )
                answer = response.choices[0].message.content.strip()
                return answer
            except Exception as e:
                logger.error(f"Groq generation failed: {e}. Falling back to local generation.")

        # 2. Try Gemini
        if self.provider == "gemini" and self._gemini_client is not None:
            try:
                response = self._gemini_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[PromptBuilder.SYSTEM_PROMPT, user_prompt]
                )
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                logger.error(f"Gemini generation failed: {e}. Falling back to local generation.")

        # 3. Local extractive generator
        return self._local_extractive_generation(query, sources)
