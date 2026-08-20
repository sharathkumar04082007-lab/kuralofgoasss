import os
from typing import Generator, List, Dict, Any, Optional
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
from config.settings import settings
from config.logging_config import logger
from pipeline.schemas import Document
from ingestion.normalizer import TextNormalizer


# High quality seed MSMARCO passages for offline development / deterministic fallback
OFFLINE_SAMPLE_RECORDS = [
    {
        "query_id": 100001,
        "source_lang": "en",
        "target_lang": "hi",
        "Eng_Query": "what is the capital of France?",
        "query": "फ्रांस की राजधानी क्या है?",
        "Eng_Answer": "Paris is the capital and most populous city of France.",
        "Answer": "पेरिस फ्रांस की राजधानी और सबसे अधिक आबादी वाला शहर है।",
        "passages": {
            "English_passages": [
                "Paris is the capital and most populous city of France, with an estimated population of 2,165,423 residents in 2019 in an area of more than 105 km2.",
                "France is a country located in Western Europe. Its largest cities include Paris, Marseille, and Lyon.",
                "Tourism in France is an important economic sector. The Eiffel Tower in Paris attracts millions of global visitors every year."
            ],
            "Translated_passages": [
                "पेरिस फ्रांस की राजधानी और सबसे अधिक आबादी वाला शहर है, जिसकी अनुमानित जनसंख्या 2019 में 2,165,423 निवासी थी।",
                "फ्रांस पश्चिमी यूरोप में स्थित एक देश है। इसके सबसे बड़े शहरों में पेरिस, मार्सिले और ल्यों शामिल हैं।",
                "फ्रांस में पर्यटन एक महत्वपूर्ण आर्थिक क्षेत्र है। पेरिस में एफिल टॉवर हर साल लाखों वैश्विक आगंतुकों को आकर्षित करता है।"
            ],
            "is_selected": [1, 0, 0]
        }
    },
    {
        "query_id": 100002,
        "source_lang": "en",
        "target_lang": "hi",
        "Eng_Query": "what is photosynthesis in plants?",
        "query": "पौधों में प्रकाश संश्लेषण क्या है?",
        "Eng_Answer": "Photosynthesis is the process used by plants and other organisms to convert light energy into chemical energy.",
        "Answer": "प्रकाश संश्लेषण वह प्रक्रिया है जिसका उपयोग पौधे प्रकाश ऊर्जा को रासायनिक ऊर्जा में बदलने के लिए करते हैं।",
        "passages": {
            "English_passages": [
                "Photosynthesis is the biological process by which green plants and certain organisms transform light energy, usually from the sun, into chemical energy stored in glucose bonds. Chlorophyll absorbs solar radiation to drive this reaction.",
                "Plant cells contain chloroplasts, the organelles where photosynthesis takes place using water and carbon dioxide, releasing oxygen as a byproduct.",
                "Cellular respiration is the opposite metabolic pathway where cells break down glucose to generate ATP energy."
            ],
            "Translated_passages": [
                "प्रकाश संश्लेषण वह जैविक प्रक्रिया है जिसके द्वारा हरे पौधे और कुछ जीव प्रकाश ऊर्जा को ग्लूकोज बॉन्ड में संग्रहीत रासायनिक ऊर्जा में बदलते हैं।",
                "पौधों की कोशिकाओं में क्लोरोप्लास्ट होते हैं, वे अंग जहां प्रकाश संश्लेषण पानी और कार्बन डाइऑक्साइड का उपयोग करके होता है, ऑक्सीजन को उप-उत्पाद के रूप में छोड़ता है।",
                "सेलुलर श्वसन विपरीत चयापचय मार्ग है जहां कोशिकाएं एटीपी ऊर्जा उत्पन्न करने के लिए ग्लूकोज को तोड़ती हैं।"
            ],
            "is_selected": [1, 1, 0]
        }
    },
    {
        "query_id": 100003,
        "source_lang": "en",
        "target_lang": "hi",
        "Eng_Query": "what causes high blood pressure?",
        "query": "उच्च रक्तचाप का क्या कारण है?",
        "Eng_Answer": "Common causes of high blood pressure include high salt intake, lack of exercise, obesity, genetics, and chronic stress.",
        "Answer": "उच्च रक्तचाप के सामान्य कारणों में अधिक नमक का सेवन, व्यायाम की कमी, मोटापा, आनुवंशिकी और तनाव शामिल हैं।",
        "passages": {
            "English_passages": [
                "Hypertension, or high blood pressure, develops when the force of blood against artery walls is consistently too high. Primary risk factors include a high-sodium diet, physical inactivity, obesity, aging, and excessive alcohol consumption.",
                "Blood pressure is measured in millimeters of mercury (mmHg) and is recorded as systolic over diastolic pressure.",
                "Normal blood pressure for adults is defined as a systolic pressure of less than 120 mmHg and a diastolic pressure of less than 80 mmHg."
            ],
            "Translated_passages": [
                "उच्च रक्तचाप तब विकसित होता है जब धमनी की दीवारों के खिलाफ रक्त का बल लगातार बहुत अधिक होता है। प्राथमिक जोखिम कारकों में उच्च सोडियम आहार, शारीरिक निष्क्रियता, मोटापा और उम्र बढ़ना शामिल हैं।",
                "रक्तचाप को पारे के मिलीमीटर (mmHg) में मापा जाता है।",
                "वयस्कों के लिए सामान्य रक्तचाप 120/80 mmHg से कम माना जाता है।"
            ],
            "is_selected": [1, 0, 0]
        }
    },
    {
        "query_id": 100004,
        "source_lang": "en",
        "target_lang": "hi",
        "Eng_Query": "how does a vector database work?",
        "query": "वेक्टर डेटाबेस कैसे काम करता है?",
        "Eng_Answer": "Vector databases store high-dimensional embeddings and use approximate nearest neighbor algorithms like HNSW to perform fast semantic similarity search.",
        "Answer": "वेक्टर डेटाबेस उच्च-आयामी एम्बेडिंग संग्रहीत करते हैं और तेज शब्दार्थ समानता खोज करने के लिए HNSW जैसे अनुमानित निकटतम पड़ोसी एल्गोरिदम का उपयोग करते हैं।",
        "passages": {
            "English_passages": [
                "A vector database indexes and stores vector embeddings generated by machine learning models. It uses Approximate Nearest Neighbor (ANN) algorithms such as Hierarchical Navigable Small World (HNSW) and IVF to quickly search multi-dimensional vector spaces using distance metrics like Cosine similarity or Euclidean distance.",
                "Traditional relational databases index scalar data like integers and strings using B-trees, which are not suitable for high-dimensional vector search.",
                "Embeddings capture semantic relationships between words, documents, audio, or images in continuous numerical vectors."
            ],
            "Translated_passages": [
                "वेक्टर डेटाबेस मशीन लर्निंग मॉडल द्वारा उत्पन्न वेक्टर एम्बेडिंग को अनुक्रमित और संग्रहीत करता है। यह कोसाइन समानता जैसी दूरी मेट्रिक्स का उपयोग करके बहु-आयामी वेक्टर रिक्त स्थान को तुरंत खोजने के लिए HNSW जैसे अनुमानित निकटतम पड़ोसी (ANN) एल्गोरिदम का उपयोग करता है।",
                "पारंपरिक संबंधपरक डेटाबेस बी-पेड़ों का उपयोग करके स्केलर डेटा को अनुक्रमित करते हैं, जो उच्च-आयामी वेक्टर खोज के लिए उपयुक्त नहीं हैं।",
                "एम्बेडिंग निरंतर संख्यात्मक वैक्टर में शब्दों, दस्तावेजों, ऑडियो या छवियों के बीच अर्थ संबंधी संबंधों को पकड़ते हैं।"
            ],
            "is_selected": [1, 0, 1]
        }
    },
    {
        "query_id": 100005,
        "source_lang": "en",
        "target_lang": "hi",
        "Eng_Query": "what are speech-to-text models?",
        "query": "स्पीच-टू-टेक्स्ट मॉडल क्या हैं?",
        "Eng_Answer": "Speech-to-text models convert spoken audio waveforms into written text using acoustic and language models.",
        "Answer": "स्पीच-टू-टेक्स्ट मॉडल ध्वनिक और भाषा मॉडल का उपयोग करके बोले गए ऑडियो तरंगों को लिखित पाठ में बदलते हैं।",
        "passages": {
            "English_passages": [
                "Speech-to-text (STT) or Automatic Speech Recognition (ASR) systems process raw acoustic waveforms, extract spectral features like Mel-spectrograms, and map acoustic units to characters or subword tokens using deep neural network architectures such as Conformer, Whisper, or CTC-based models.",
                "Sarvam AI specializes in Indian language speech models optimized for multiple Indic accents, dialects, and code-mixed speech.",
                "ElevenLabs offers state-of-the-art voice synthesis and multilingual speech-to-text transcription APIs."
            ],
            "Translated_passages": [
                "स्पीच-टू-टेक्स्ट (STT) या स्वचालित भाषण पहचान (ASR) सिस्टम कच्चे ध्वनिक तरंगों को संसाधित करते हैं, सुविधाओं को निकालते हैं, और गहरे तंत्रिका नेटवर्क का उपयोग करके ध्वनिक इकाइयों को पाठ में मैप करते हैं।",
                "सर्वम एआई कई भारतीय लहजों, बोलियों और कोड-मिश्रित भाषण के लिए अनुकूलित भारतीय भाषा भाषण मॉडल में माहिर है।",
                "इलेवनलैब्स अत्याधुनिक आवाज संश्लेषण और बहुभाषी स्पीच-टू-टेक्स्ट ट्रांसक्रिप्शन एपीआई प्रदान करता है।"
            ],
            "is_selected": [1, 1, 1]
        }
    }
]


class MSMARCODatasetLoader:
    """Streams and batches records from the Hugging Face ai4bharat/MSMARCO-XI dataset."""

    AVAILABLE_SHARDS = {
        "hi": "validation/hinval.parquet",
        "ben": "validation/benval.parquet",
        "tam": "validation/tamval.parquet",
        "tel": "validation/telval.parquet",
        "guj": "validation/gujval.parquet",
        "mar": "validation/marval.parquet",
        "kan": "validation/kanval.parquet",
        "mal": "validation/malval.parquet",
        "pan": "validation/panval.parquet",
        "urd": "validation/urdval.parquet",
    }

    def __init__(self, repo_id: str = "ai4bharat/MSMARCO-XI", cache_dir: Optional[str] = None):
        self.repo_id = repo_id
        self.cache_dir = cache_dir or settings.DATASET_CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

    def download_shard(self, shard_filename: str) -> Optional[str]:
        """Download a specific parquet shard from Hugging Face Hub."""
        try:
            logger.info(f"Downloading shard {shard_filename} from {self.repo_id}...")
            local_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=shard_filename,
                repo_type="dataset",
                local_dir=self.cache_dir
            )
            logger.info(f"Successfully downloaded {shard_filename} -> {local_path}")
            return local_path
        except Exception as e:
            logger.warning(f"Could not download shard {shard_filename} ({e}). Falling back to local/cached data.")
            return None

    def stream_records(
        self,
        languages: Optional[List[str]] = None,
        max_records: Optional[int] = None,
        use_fallback_if_offline: bool = True
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream MSMARCO-XI records in a memory-efficient generator.
        Yields raw records dictionary.
        """
        languages = languages or ["hi"]
        yielded_count = 0

        downloaded_any = False
        for lang in languages:
            shard_file = self.AVAILABLE_SHARDS.get(lang, "validation/hinval.parquet")
            local_parquet = self.download_shard(shard_file)
            
            if local_parquet and os.path.exists(local_parquet):
                downloaded_any = True
                try:
                    parquet_file = pq.ParquetFile(local_parquet)
                    # Stream row groups to avoid loading entire 100k rows into RAM
                    for i in range(parquet_file.num_row_groups):
                        table = parquet_file.read_row_group(i)
                        pydict = table.to_pydict()
                        keys = list(pydict.keys())
                        num_rows = len(pydict[keys[0]])
                        
                        for row_idx in range(num_rows):
                            row = {k: pydict[k][row_idx] for k in keys}
                            yield row
                            yielded_count += 1
                            if max_records and yielded_count >= max_records:
                                return
                except Exception as ex:
                    logger.error(f"Error reading parquet {local_parquet}: {ex}")

        # Fallback to offline samples if offline or needed
        if not downloaded_any and use_fallback_if_offline:
            logger.info(f"Using offline sample corpus ({len(OFFLINE_SAMPLE_RECORDS)} seed records)")
            for rec in OFFLINE_SAMPLE_RECORDS:
                yield rec
                yielded_count += 1
                if max_records and yielded_count >= max_records:
                    return

    def stream_documents(
        self,
        languages: Optional[List[str]] = None,
        max_records: Optional[int] = None
    ) -> Generator[Document, None, None]:
        """Stream parsed and normalized Document instances."""
        for record in self.stream_records(languages=languages, max_records=max_records):
            docs = TextNormalizer.parse_msmarco_record(record)
            for doc in docs:
                yield doc
