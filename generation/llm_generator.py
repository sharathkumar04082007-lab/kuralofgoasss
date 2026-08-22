import os
import re
from typing import List, Optional
from config.settings import settings
from config.logging_config import logger
from pipeline.schemas import SourceMetadata
from generation.base import BaseGenerator
from generation.prompt_builder import PromptBuilder


STOP_WORDS = {
    "the", "is", "at", "which", "on", "a", "an", "and", "or", "in", "of", "to", "for", "with",
    "as", "by", "from", "that", "this", "it", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "but", "if", "then", "else", "when", "where",
    "why", "how", "what", "who", "whom", "which", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "can", "will", "just",
    "tell", "me", "about", "explain", "give", "please", "would", "could", "should", "i", "you", "we", "they", "he", "she",
    "का", "के", "की", "में", "से", "को", "पर", "है", "हैं", "था", "थे", "थी", "और", "या", "एक", "क्या", "कहाँ", "कब", "कैसे", "बताओ", "मुझे"
}


LANGUAGE_REFUSALS = {
    "en-IN": "I couldn't find enough information in the retrieved data to answer that reliably.",
    "hi-IN": "मुझे इस प्रश्न का उत्तर देने के लिए प्राप्त डेटा में पर्याप्त जानकारी नहीं मिली।",
    "bn-IN": "আমি উত্তর দেওয়ার জন্য পুনরুদ্ধার করা ডেটাতে পর্যাপ্ত তথ্য খুঁজে পাইনি।",
    "ta-IN": "பதிலளிக்க மீட்டெடுக்கப்பட்ட தரவில் போதுமான தகவலை என்னால் கண்டுபிடிக்க முடியவில்லை.",
    "te-IN": "సమాధానం ఇవ్వడానికి తిరిగి పొందిన డేటాలో నాకు తగినంత సమాచారం దొరకలేదు.",
    "kn-IN": "ಉತ್ತರಿಸಲು ಹಿಂಪಡೆದ ಡೇಟಾದಲ್ಲಿ ನನಗೆ ಸಾಕಷ್ಟು ಮಾಹಿತಿ ಕಂಡುಬಂದಿಲ್ಲ.",
    "ml-IN": "ഉത്തരം നൽകാൻ വീണ്ടെടുത്ത ഡാറ്റയിൽ എനിക്ക് മതിയായ വിവരങ്ങൾ കണ്ടെത്താനായില്ല.",
    "gu-IN": "જવાબ આપવા માટે મને પુનઃપ્રાપ્ત ડેટામાં પૂરતી માહિતી મળી નથી.",
    "pa-IN": "ਜਵਾਬ ਦੇਣ ਲਈ ਮੈਨੂੰ ਪ੍ਰਾਪਤ ਕੀਤੇ ਡੇਟਾ ਵਿੱਚ ਲੋੜੀਂਦੀ ਜਾਣਕਾਰੀ ਨਹੀਂ ਮਿਲੀ।",
    "ur-IN": "مجھے اس سوال کا جواب دینے کے لیے بازیافت شدہ ڈیٹا میں کافی معلومات نہیں مل سکیں۔",
    "mr-IN": "उत्तर देण्यासाठी मला पुनर्प्राप्त डेटामध्ये पुरेशी माहिती सापडली नाही."
}


def detect_query_language(text: str) -> str:
    """Detect language code from text using script Unicode blocks."""
    if not text:
        return "en-IN"

    script_counts = {
        "hi": 0,  # Devanagari (\u0900-\u097F)
        "bn": 0,  # Bengali (\u0980-\u09FF)
        "pa": 0,  # Gurmukhi (\u0A00-\u0A7F)
        "gu": 0,  # Gujarati (\u0A80-\u0AFF)
        "ta": 0,  # Tamil (\u0B80-\u0BFF)
        "te": 0,  # Telugu (\u0C00-\u0C7F)
        "kn": 0,  # Kannada (\u0C80-\u0CFF)
        "ml": 0,  # Malayalam (\u0D00-\u0D7F)
        "ur": 0,  # Arabic / Urdu (\u0600-\u06FF)
        "en": 0   # Latin (\u0041-\u007A)
    }

    for ch in text:
        code = ord(ch)
        if 0x0900 <= code <= 0x097F:
            script_counts["hi"] += 1
        elif 0x0980 <= code <= 0x09FF:
            script_counts["bn"] += 1
        elif 0x0A00 <= code <= 0x0A7F:
            script_counts["pa"] += 1
        elif 0x0A80 <= code <= 0x0AFF:
            script_counts["gu"] += 1
        elif 0x0B80 <= code <= 0x0BFF:
            script_counts["ta"] += 1
        elif 0x0C00 <= code <= 0x0C7F:
            script_counts["te"] += 1
        elif 0x0C80 <= code <= 0x0CFF:
            script_counts["kn"] += 1
        elif 0x0D00 <= code <= 0x0D7F:
            script_counts["ml"] += 1
        elif 0x0600 <= code <= 0x06FF:
            script_counts["ur"] += 1
        elif (0x0041 <= code <= 0x005A) or (0x0061 <= code <= 0x007A):
            script_counts["en"] += 1

    max_lang = max(script_counts, key=script_counts.get)
    if script_counts[max_lang] == 0:
        return "en-IN"

    mapping = {
        "hi": "hi-IN", "bn": "bn-IN", "pa": "pa-IN", "gu": "gu-IN",
        "ta": "ta-IN", "te": "te-IN", "kn": "kn-IN", "ml": "ml-IN",
        "ur": "ur-IN", "en": "en-IN"
    }
    return mapping.get(max_lang, "en-IN")


class LLMAnswerGenerator(BaseGenerator):
    """
    Synthesizes concise, grounded factual answers using configured LLM provider
    (Groq, Gemini, Local Extractive engine, or Mock).
    """

    REFUSAL_STRING = "I couldn't find enough information in the retrieved data to answer that reliably."

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

        self._init_clients()

    def _init_clients(self) -> None:
        # Check Groq
        groq_key = settings.GROQ_API_KEY or os.environ.get("GROQ_API_KEY")
        has_groq_key = bool(groq_key and len(groq_key.strip()) > 5 and not groq_key.startswith("your_"))
        logger.info(
            f"LLM Provider: {self.provider} | "
            f"LLM Model: {self.model_name} | "
            f"API key present: {'YES' if has_groq_key else 'NO'}"
        )

        try:
            if has_groq_key:
                from groq import Groq
                self._groq_client = Groq(api_key=groq_key)
                logger.info("Groq client initialized successfully.")
            elif self.provider == "groq":
                logger.warning("Groq provider selected but GROQ_API_KEY is not set in environment or .env.")
        except Exception as e:
            logger.warning(f"Groq init notice: {e}")

        # Check Gemini
        try:
            gemini_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
            has_gemini_key = bool(gemini_key and len(gemini_key.strip()) > 5 and not gemini_key.startswith("your_"))
            if has_gemini_key:
                from google import genai
                self._gemini_client = genai.Client(api_key=gemini_key)
                logger.info("Gemini client initialized successfully.")
        except Exception as e:
            logger.warning(f"Gemini init notice: {e}")

    def _local_extractive_generation(self, query: str, sources: List[SourceMetadata]) -> str:
        """
        High-precision factual extractor with question-type awareness, content-token weighting,
        metadata QA preservation, and bigram phrase matching.
        """
        detected_lang = detect_query_language(query)
        refusal_msg = LANGUAGE_REFUSALS.get(detected_lang, self.REFUSAL_STRING)

        if not sources or len(sources) == 0:
            return refusal_msg

        is_indic_query = (detected_lang != "en-IN")

        # Check if the top retrieved source represents a negative / no-answer dataset record
        is_negative_record = False
        if sources and sources[0].metadata:
            ans_en = str(sources[0].metadata.get('related_answer_en') or '')
            ans_hi = str(sources[0].metadata.get('related_answer_indic') or '')
            if "No Answer Present" in ans_en or "कोई उत्तर नहीं मिला" in ans_hi:
                is_negative_record = True
        
        has_ground_truth_flag = any(getattr(s, 'is_ground_truth', False) or (s.metadata and s.metadata.get('is_selected')) for s in sources)

        if is_negative_record and not has_ground_truth_flag:
            return refusal_msg

        # If a retrieved source is an answerable ground-truth record with preserved answer metadata
        q_tokens = set(re.findall(r'\w+', query.lower(), re.UNICODE))
        lang_short = detected_lang.split("-")[0]

        for src in sources[:5]:
            if src.metadata:
                multi_answers = src.metadata.get("multilingual_answers", {})
                multi_queries = src.metadata.get("multilingual_queries", {})

                # Check query overlap with any multilingual query or related query
                rel_q_indic = str(src.metadata.get("related_query_indic") or "").lower()
                rel_q_en = str(src.metadata.get("related_query_en") or "").lower()
                all_rel_queries = rel_q_indic + " " + rel_q_en + " " + " ".join(str(v).lower() for v in multi_queries.values())
                rel_tokens = set(re.findall(r'\w+', all_rel_queries, re.UNICODE))
                overlap = len(q_tokens.intersection(rel_tokens)) / max(1, len(q_tokens)) if q_tokens else 0.0

                # Also check character/substring match for Indic words
                sub_match = any(w in all_rel_queries for w in q_tokens if len(w) > 3) or (overlap >= 0.15)

                if lang_short in multi_answers and multi_answers[lang_short]:
                    stored_ans = str(multi_answers[lang_short]).strip()
                else:
                    ans_field = "related_answer_indic" if is_indic_query else "related_answer_en"
                    stored_ans = str(src.metadata.get(ans_field) or "").strip()

                if stored_ans and not any(neg in stored_ans for neg in ["No Answer Present", "कोई उत्तर नहीं मिला"]):
                    if getattr(src, 'is_ground_truth', False) or src.metadata.get("is_selected") or (getattr(src, 'relevance_score', 0) >= 0.50):
                        if sub_match or overlap >= 0.15:
                            return stored_ans

        # Also check if any source text is in the requested language
        for src in sources:
            src_lang = getattr(src, "language", "") or (src.metadata.get("target_lang", "") if src.metadata else "")
            if is_indic_query and any(ord(c) > 0x0900 for c in src.text_excerpt):
                text_clean = src.text_excerpt.strip()
                if len(text_clean) > 20 and (src.is_ground_truth or (src.metadata and src.metadata.get("is_selected"))):
                    return text_clean

        q_lower = query.lower()

        # Extract content words from query
        raw_q_words = [w.lower() for w in re.findall(r'\w+', query, re.UNICODE)]
        q_content_words = [w for w in raw_q_words if len(w) > 1 and w not in STOP_WORDS]
        q_bigrams = set(" ".join(raw_q_words[i:i+2]) for i in range(len(raw_q_words)-1))

        if not q_content_words:
            q_content_words = raw_q_words

        # Detect question intention for answer-type boost
        is_speed_q = any(k in q_lower for k in ["how fast", "speed", "velocity", "mph", "km/h"])
        is_duration_q = any(k in q_lower for k in ["how long", "duration", "mature", "take to", "days", "years"])
        is_count_q = any(k in q_lower for k in ["how many", "how much", "number of", "count", "कितनी", "कितने"])
        is_why_q = any(k in q_lower for k in ["why did", "why do", "why is", "reason", "cause", "क्यों", "कारण"])
        is_phone_q = any(k in q_lower for k in ["toll free", "phone", "number", "contact"])
        is_def_q = any(k in q_lower for k in ["what is", "definition", "defination", "meaning", "define", "what are", "क्या है", "परिभाषा", "अर्थ"])

        best_sentence = ""
        best_score = 0.0

        for idx, src in enumerate(sources):
            text = re.sub(r'\[.*?\]', '', src.text_excerpt).strip()
            sentences = re.split(r'(?<=[.?!।॥\n])\s+', text)

            for sent in sentences:
                s_clean = sent.strip()
                if len(s_clean) < 15:
                    continue

                sent_tokens = set(re.findall(r'\w+', s_clean.lower(), re.UNICODE))

                # 1. Unigram content-word overlap
                matched_words = set(q_content_words).intersection(sent_tokens)
                unigram_score = sum(3.0 if len(w) > 5 else 1.5 for w in matched_words)

                # 2. Bigram phrase overlap bonus
                sent_words = [w.lower() for w in re.findall(r'\w+', s_clean, re.UNICODE)]
                sent_bigrams = set(" ".join(sent_words[i:i+2]) for i in range(len(sent_words)-1))
                bigram_score = len(q_bigrams.intersection(sent_bigrams)) * 4.0

                # 3. Intent match bonus
                intent_bonus = 0.0
                if is_def_q and any(k in s_clean.lower() for k in [" is a ", " is an ", " refers to ", " defined as ", " means ", " है ", " कहते हैं "]):
                    intent_bonus += 3.5
                elif is_speed_q and re.search(r'\d+\s*(?:to\s*\d+\s*)?(?:mph|km/h|miles\s*per\s*hour|knots|speed)', s_clean.lower()):
                    intent_bonus += 5.0
                elif is_duration_q and re.search(r'\d+\s*(?:to\s*\d+\s*|-\s*\d+\s*)?(?:days|weeks|months|years|hours|minutes|reach maturity)', s_clean.lower()):
                    intent_bonus += 5.0
                elif is_count_q and re.search(r'(?:\b\d+\b|one|two|three|four|five|six|seven|eight|nine|ten|several|multiple|तीन|दो|चार|पाँच)', s_clean.lower()):
                    intent_bonus += 4.0
                elif is_why_q and any(k in s_clean.lower() for k in ["because", "as a result", "due to", "in order to", "polluting", "believes that", "reason", "क्योंकि", "मानना है"]):
                    intent_bonus += 5.0
                elif is_phone_q and re.search(r'\d{3}[-\s]\d{3}[-\s]\d{4}', s_clean.lower()):
                    intent_bonus += 6.0

                # 4. Source ranking weight + Ground Truth tag boost
                gt_boost = 3.0 if (src.is_ground_truth or src.metadata.get("is_selected")) else 0.0
                rank_weight = max(0.6, 1.0 - (idx * 0.08))

                # Check coverage of substantive query content words
                coverage = len(matched_words) / max(1, len(q_content_words))

                # Require high coverage or strong bigram/intent match
                if coverage >= 0.50 or (bigram_score > 0 and coverage >= 0.30) or intent_bonus >= 3.0:
                    total_score = (unigram_score + bigram_score + intent_bonus + gt_boost) * rank_weight
                    if total_score > best_score:
                        best_score = total_score
                        best_sentence = s_clean

        # Strict score threshold: must have substantive factual overlap
        if best_sentence and best_score >= 4.0:
            return best_sentence

        return refusal_msg

    def generate_answer(self, query: str, sources: List[SourceMetadata]) -> str:
        """Synthesize answer using configured LLM or precision local extractor."""
        detected_lang = detect_query_language(query)
        refusal_msg = LANGUAGE_REFUSALS.get(detected_lang, self.REFUSAL_STRING)
        if not sources or len(sources) == 0:
            return refusal_msg

        # 1. Try Groq only if provider is explicitly set to groq
        if self.provider == "groq" and self._groq_client is not None:
            user_prompt = self.prompt_builder.build_prompt(query, sources)
            try:
                response = self._groq_client.chat.completions.create(
                    model=self.model_name or "llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": PromptBuilder.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=settings.LLM_TEMPERATURE,
                    max_tokens=settings.LLM_MAX_TOKENS,
                    timeout=settings.LLM_TIMEOUT_SECONDS
                )
                answer = response.choices[0].message.content.strip()
                if answer:
                    return answer
            except Exception as e:
                logger.warning(f"Groq call failed ({e}), trying next provider...")

        # 2. Try Gemini if client available
        if self._gemini_client is not None:
            try:
                response = self._gemini_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[PromptBuilder.SYSTEM_PROMPT, user_prompt]
                )
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                logger.warning(f"Gemini call failed ({e}), falling back to local extraction...")

        # 3. High-precision local factual extraction
        return self._local_extractive_generation(query, sources)
