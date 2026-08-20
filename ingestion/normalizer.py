import unicodedata
import re
import hashlib
from typing import Optional, Dict, Any, List
from pipeline.schemas import Document


class TextNormalizer:
    """Normalizes raw multilingual text and structures metadata."""

    @staticmethod
    def clean_text(text: Optional[str]) -> str:
        """Clean and normalize unicode text, strip control characters, and normalize whitespace."""
        if not text or not isinstance(text, str):
            return ""
        
        # Unicode normalization (NFKC ensures Indic conjuncts and accents are well-formed)
        normalized = unicodedata.normalize("NFKC", text)
        
        # Remove control characters except standard line breaks and tabs
        cleaned = "".join(ch for ch in normalized if unicodedata.category(ch)[0] != "C" or ch in "\n\r\t")
        
        # Collapse excessive whitespace
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        
        return cleaned.strip()

    @staticmethod
    def generate_deterministic_id(prefix: str, content: str) -> str:
        """Generate a deterministic document / chunk ID from content hash."""
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        return f"{prefix}_{content_hash}"

    @classmethod
    def parse_msmarco_record(
        cls, 
        record: Dict[str, Any], 
        split: str = "validation",
        prefer_english: bool = True
    ) -> List[Document]:
        """
        Parse an MSMARCO-XI record into normalized Document instances.
        Extracts passages and attaches ground truth relevance tags.
        """
        documents: List[Document] = []
        
        query_id = record.get("query_id")
        source_lang = record.get("source_lang", "en")
        target_lang = record.get("target_lang", "hi")
        
        # Query and answers
        eng_query = cls.clean_text(record.get("Eng_Query", ""))
        indic_query = cls.clean_text(record.get("query", ""))
        eng_answer = cls.clean_text(record.get("Eng_Answer", ""))
        indic_answer = cls.clean_text(record.get("Answer", ""))
        
        passages_struct = record.get("passages") or {}
        eng_passages = passages_struct.get("English_passages", []) or []
        indic_passages = passages_struct.get("Translated_passages", []) or []
        is_selected_list = passages_struct.get("is_selected", []) or []
        
        # Max length of passages
        count = max(len(eng_passages), len(indic_passages))
        
        for idx in range(count):
            eng_p = cls.clean_text(eng_passages[idx]) if idx < len(eng_passages) else ""
            indic_p = cls.clean_text(indic_passages[idx]) if idx < len(indic_passages) else ""
            is_selected = (is_selected_list[idx] == 1) if idx < len(is_selected_list) else False
            
            # Create English document entry if available
            if eng_p and len(eng_p) > 10:
                doc_id = f"doc_{query_id}_en_{idx}" if query_id else cls.generate_deterministic_id("doc_en", eng_p)
                documents.append(
                    Document(
                        document_id=doc_id,
                        query_id=query_id,
                        title=f"Query {query_id} Passage {idx}",
                        text=eng_p,
                        language="en",
                        source="ai4bharat/MSMARCO-XI",
                        split=split,
                        metadata={
                            "passage_index": idx,
                            "is_selected": is_selected,
                            "related_query_en": eng_query,
                            "related_answer_en": eng_answer,
                            "source_lang": source_lang,
                            "target_lang": target_lang,
                        }
                    )
                )
                
            # Create Indic document entry if available
            if indic_p and len(indic_p) > 10:
                doc_id = f"doc_{query_id}_{target_lang}_{idx}" if query_id else cls.generate_deterministic_id(f"doc_{target_lang}", indic_p)
                documents.append(
                    Document(
                        document_id=doc_id,
                        query_id=query_id,
                        title=f"Query {query_id} ({target_lang}) Passage {idx}",
                        text=indic_p,
                        language=target_lang,
                        source="ai4bharat/MSMARCO-XI",
                        split=split,
                        metadata={
                            "passage_index": idx,
                            "is_selected": is_selected,
                            "related_query_indic": indic_query,
                            "related_answer_indic": indic_answer,
                            "related_query_en": eng_query,
                            "source_lang": source_lang,
                            "target_lang": target_lang,
                        }
                    )
                )
                
        return documents
