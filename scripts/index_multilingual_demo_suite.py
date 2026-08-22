import os
import sys
import uuid
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r"c:\Users\Saharsh\Desktop\New folder\Actual")

from config.settings import settings
from embeddings.sentence_embedder import MultilingualEmbedder
from retrieval.vector_store import QdrantVectorStore
from retrieval.bm25_retriever import BM25LexicalRetriever
from pipeline.schemas import Chunk

# Complete 5-Language Multilingual Ground Truth Suite
DEMO_MULTILINGUAL_RECORDS = [
    {
        "query_id": 1102432,
        "title": "Corporation Definition",
        "queries": {
            "en": "what is a corporation?",
            "hi": "कॉर्पोरेशन क्या है?",
            "kn": "ಕಾರ್ಪೊರೇಷನ್ ಎಂದರೇನು?",
            "ta": "கார்ப்பரேஷன் என்றால் என்ன?",
            "te": "కార్పొరేషన్ అంటే ఏమిటి?"
        },
        "answers": {
            "en": "A corporation is a company or group of people authorized to act as a single entity and recognized as such in law.",
            "hi": "निगम एक कंपनी या लोगों का समूह होता है जो एक एकल इकाई के रूप में कार्य करने के लिए अधिकृत होता है और कानून में इस प्रकार से मान्यता प्राप्त होती है।",
            "kn": "ಕಾರ್ಪೊರೇಷನ್ ಎನ್ನುವುದು ಕಾನೂನಿನ ಅಡಿಯಲ್ಲಿ ಒಂದೇ ಘಟಕವಾಗಿ ಕಾರ್ಯನಿರ್ವಹಿಸಲು ಅಧಿಕಾರ ಹೊಂದಿರುವ ಕಂಪನಿ ಅಥವಾ ಜನರ ಗುಂಪಾಗಿದೆ.",
            "ta": "கார்ப்பரேஷன் என்பது ஒரு நிறுவனமாக அல்லது சட்டப்படி அங்கீகரிக்கப்பட்ட நபர்களின் குழுவாக செயல்பட அதிகாரம் பெற்ற அமைப்பாகும்.",
            "te": "కార్పొరేషన్ అనేది చట్టంలో ఒకే సంస్థగా వ్యవహరించడానికి మరియు గుర్తించబడటానికి అధికారం కలిగిన సంస్థ లేదా వ్యక్తుల సమూహం."
        },
        "passages": {
            "en": [
                "A corporation is a company or group of people authorized to act as a single entity and recognized as such in law. Early incorporated entities were established by charter.",
                "Corporations have limited liability for their shareholders, meaning shareholders are not personally liable for the company's debts and obligations.",
                "Modern corporations are created by registration under corporate law and possess legal personality separate from their members."
            ],
            "hi": [
                "निगम एक कंपनी या लोगों का समूह होता है जो एक एकल इकाई के रूप में कार्य करने के लिए अधिकृत होता है और कानून में इस प्रकार से मान्यता प्राप्त होती है।",
                "निगमों के पास अपने शेयरधारकों के लिए सीमित देयता होती है, जिसका अर्थ है कि शेयरधारक कंपनी के ऋणों के लिए व्यक्तिगत रूप से उत्तरदायी नहीं हैं।",
                "आधुनिक निगम कॉर्पोरेट कानून के तहत पंजीकरण द्वारा बनाए जाते हैं और उनके सदस्यों से अलग कानूनी व्यक्तित्व रखते हैं।"
            ],
            "kn": [
                "ಕಾರ್ಪೊರೇಷನ್ ಎನ್ನುವುದು ಕಾನೂನಿನ ಅಡಿಯಲ್ಲಿ ಒಂದೇ ಘಟಕವಾಗಿ ಕಾರ್ಯನಿರ್ವಹಿಸಲು ಅಧಿಕಾರ ಹೊಂದಿರುವ ಕಂಪನಿ ಅಥವಾ ಜನರ ಗುಂಪಾಗಿದೆ.",
                "ಕಾರ್ಪೊರೇಷನ್‌ಗಳು ತಮ್ಮ ಷೇರುದಾರರಿಗೆ ಸೀಮಿತ ಹೊಣೆಗಾರಿಕೆಯನ್ನು ಹೊಂದಿರುತ್ತವೆ, ಅಂದರೆ ಕಂಪನಿಯ ಸಾಲಗಳಿಗೆ ಷೇರುದಾರರು ವೈಯಕ್ತಿಕವಾಗಿ ಜವಾಬ್ದಾರರಾಗಿರುವುದಿಲ್ಲ.",
                "ಆಧುನಿಕ ಕಾರ್ಪೊರೇಷನ್‌ಗಳನ್ನು ಕಾರ್ಪೊರೇಟ್ ಕಾನೂನಿನ ಅಡಿಯಲ್ಲಿ ನೋಂದಣಿ ಮಾಡುವ ಮೂಲಕ ರಚಿಸಲಾಗುತ್ತದೆ."
            ],
            "ta": [
                "கார்ப்பரேஷன் என்பது ஒரு நிறுவனமாக அல்லது சட்டப்படி அங்கீகரிக்கப்பட்ட நபர்களின் குழுவாக செயல்பட அதிகாரம் பெற்ற அமைப்பாகும்.",
                "கார்ப்பரேஷன்கள் தங்கள் பங்குதாரர்களுக்கு வரையறுக்கப்பட்ட பொறுப்பைக் கொண்டுள்ளன, அதாவது நிறுவனத்தின் கடன்களுக்கு பங்குதாரர்கள் பொறுப்பல்ல.",
                "நவீன நிறுவனங்கள் கார்ப்பரேட் சட்டத்தின் கீழ் பதிவு செய்வதன் மூலம் உருவாக்கப்படுகின்றன."
            ],
            "te": [
                "కార్పొరేషన్ అనేది చట్టంలో ఒకే సంస్థగా వ్యవహరించడానికి మరియు గుర్తించబడటానికి అధికారం కలిగిన సంస్థ లేదా వ్యక్తుల సమూహం.",
                "కార్పొరేషన్లు తమ వాటాదారులకు పరిమిత బాధ్యతను కలిగి ఉంటాయి, అంటే కంపెనీ రుణాలకు వాటాదారులు వ్యక్తిగతంగా బాధ్యత వహించరు.",
                "ఆధునిక కార్పొరేషన్లు కార్పొరేట్ చట్టం ప్రకారం రిజిస్ట్రేషన్ ద్వారా సృష్టించబడతాయి."
            ]
        }
    },
    {
        "query_id": 1100338,
        "title": "Galantine Definition",
        "queries": {
            "en": "galantine definition",
            "hi": "गैलेंटाइन परिभाषा",
            "kn": "ಗ್ಯಾಲಂಟೈನ್ ವ್ಯಾಖ್ಯಾನ",
            "ta": "கேலன்டைன் வரையறை",
            "te": "గాలంటైన్ నిర్వచనం"
        },
        "answers": {
            "en": "A galantine is a French dish of de-boned stuffed meat, most commonly poultry or fish, that is poached and served cold, often in aspic.",
            "hi": "गैलेंटाइन एक फ्रांसीसी व्यंजन है जिसमें हड्डी निकाला हुआ भरवां मांस होता है, जिसे पोच किया जाता है और ठंडा परोसा जाता है।",
            "kn": "ಗ್ಯಾಲಂಟೈನ್ ಎಂಬುದು ಮೂಳೆಗಳಿಲ್ಲದ ಸ್ಟಫ್ಡ್ ಮಾಂಸದ ಫ್ರೆಂಚ್ ಭಕ್ಷ್ಯವಾಗಿದೆ, ಇದನ್ನು ಬೇಯಿಸಿ ತಣ್ಣಗೆ ಬಡಿಸಲಾಗುತ್ತದೆ.",
            "ta": "கேலன்டைன் என்பது எலும்பு நீக்கப்பட்ட அடைக்கப்பட்ட இறைச்சியின் பிரெஞ்சு உணவாகும், இது வேகவைத்து குளிராக பரிமாறப்படுகிறது.",
            "te": "గాలంటైన్ అనేది ఎముకలు లేని స్టఫ్డ్ మాంసంతో కూడిన ఫ్రెంచ్ వంటకం, దీనిని ఉడకబెట్టి చల్లగా వడ్డిస్తారు."
        },
        "passages": {
            "en": [
                "A galantine is a French dish of de-boned stuffed meat, most commonly poultry or fish, that is poached and served cold, often in aspic.",
                "Galantines are often prepared by pressing them into a cylindrical shape. Since de-boning poultry can be elaborate, this is often an elaborate culinary dish.",
                "The word galantine comes from Old French galant, referring to high culinary craft."
            ],
            "hi": [
                "गैलेंटाइन एक फ्रांसीसी व्यंजन है जिसमें हड्डी निकाला हुआ भरवां मांस होता है, जिसे पोच किया जाता है और ठंडा परोसा जाता है।",
                "गैलेंटाइन को अक्सर बेलनाकार आकार में दबाकर तैयार किया जाता है।",
                "गैलेंटाइन शब्द पुरानी फ्रांसीसी भाषा से आया है।"
            ],
            "kn": [
                "ಗ್ಯಾಲಂಟೈನ್ ಎಂಬುದು ಮೂಳೆಗಳಿಲ್ಲದ ಸ್ಟಫ್ಡ್ ಮಾಂಸದ ಫ್ರೆಂಚ್ ಭಕ್ಷ್ಯವಾಗಿದೆ, ಇದನ್ನು ಬೇಯಿಸಿ ತಣ್ಣಗೆ ಬಡಿಸಲಾಗುತ್ತದೆ.",
                "ಗ್ಯಾಲಂಟೈನ್‌ಗಳನ್ನು ಸಾಮಾನ್ಯವಾಗಿ ಸಿಲಿಂಡರಾಕಾರದ ಆಕಾರದಲ್ಲಿ ತಯಾರಿಸಲಾಗುತ್ತದೆ.",
                "ಗ್ಯಾಲಂಟೈನ್ ಎಂಬ ಪದವು ಹಳೆಯ ಫ್ರೆಂಚ್ ಭಾಷೆಯಿಂದ ಬಂದಿದೆ."
            ],
            "ta": [
                "கேலன்டைன் என்பது எலும்பு நீக்கப்பட்ட அடைக்கப்பட்ட இறைச்சியின் பிரெஞ்சு உணவாகும், இது வேகவைத்து குளிராக பரிமாறப்படுகிறது.",
                "கேலன்டைன்கள் பெரும்பாலும் உருளை வடிவில் தயாரிக்கப்படுகின்றன.",
                "கேலன்டைன் என்ற சொல் பழைய பிரெஞ்சு மொழியிலிருந்து உருவானது."
            ],
            "te": [
                "గాలంటైన్ అనేది ఎముకలు లేని స్టఫ్డ్ మాంసంతో కూడిన ఫ్రెంచ్ వంటకం, దీనిని ఉడకబెట్టి చల్లగా వడ్డిస్తారు.",
                "గాలంటైన్లను తరచుగా స్థూపాకారంలో తయారు చేస్తారు.",
                "గాలంటైన్ అనే పదం పాత ఫ్రెంచ్ భాష నుండి వచ్చింది."
            ]
        }
    },
    {
        "query_id": 205107,
        "title": "Honesty and Integrity Definition",
        "queries": {
            "en": "honesty or integrity definition",
            "hi": "ईमानदारी या सच्चाई की परिभाषा",
            "kn": "ಪ್ರಾಮಾಣಿಕತೆ ಅಥವಾ ಸಮಗ್ರತೆಯ ವ್ಯಾಖ್ಯಾನ",
            "ta": "நேர்மை அல்லது ஒருமைப்பாட்டின் வரையறை",
            "te": "నిజాయితీ లేదా సమగ్రత యొక్క నిర్వచనం"
        },
        "answers": {
            "en": "Honesty: The condition of being honest; truthfulness and fairness. Integrity: Steadfast adherence to a strict moral or ethical code.",
            "hi": "ईमानदारी: ईमानदार होने की स्थिति; सच्चाई। निष्ठा: एक सख्त नैतिक या आचरण संहिता का दृढ़ पालन।",
            "kn": "ಪ್ರಾಮಾಣಿಕತೆ: ಸತ್ಯವಾಗಿರುವುದು ಮತ್ತು ನ್ಯಾಯಯುತವಾಗಿರುವ ಸ್ಥಿತಿ. ಸಮಗ್ರತೆ: ನೈತಿಕ ತತ್ವಗಳಿಗೆ ದೃಢವಾಗಿ ಬದ್ಧವಾಗಿರುವುದು.",
            "ta": "நேர்மை: உண்மையாக இருக்கும் நிலை. ஒருமைப்பாடு: கடுமையான தார்மீಕ அல்லது நெறிமுறைக் கோட்பாட்டை உறுதியாகக் கடைப்பிடிப்பது.",
            "te": "నిజాయితీ: నిజాయితీగా ఉండే పరిస్థితి; సత్యసంధత. సమగ్రత: కఠినమైన నైతిక నియమావళికి కట్టుబడి ఉండటం."
        },
        "passages": {
            "en": [
                "Honesty: The condition of being honest; integrity, truthfulness. Integrity: Steadfast adherence to a strict moral or ethical code.",
                "Honesty refers to a facet of moral character that connotes positive and virtuous attributes such as truthfulness and straightforwardness.",
                "Integrity is the practice of being honest and showing a consistent and uncompromising adherence to strong moral and ethical principles."
            ],
            "hi": [
                "ईमानदारी: ईमानदार होने की स्थिति; सच्चाई। निष्ठा: एक सख्त नैतिक या आचरण संहिता का दृढ़ पालन।",
                "ईमानदारी नैतिक चरित्र का एक पहलू है जो सच्चाई और सीधेपन को दर्शाता है।",
                "निष्ठा मजबूत नैतिक सिद्धांतों का लगातार पालन करने का अभ्यास है।"
            ],
            "kn": [
                "ಪ್ರಾಮಾಣಿಕತೆ: ಸತ್ಯವಾಗಿರುವುದು ಮತ್ತು ನ್ಯಾಯಯುತವಾಗಿರುವ ಸ್ಥಿತಿ. ಸಮಗ್ರತೆ: ನೈತಿಕ ತತ್ವಗಳಿಗೆ ದೃಢವಾಗಿ ಬದ್ಧವಾಗಿರುವುದು.",
                "ಪ್ರಾಮಾಣಿಕತೆಯು ನೈತಿಕ ಗುಣಲಕ್ಷಣಗಳನ್ನು ಸೂಚಿಸುತ್ತದೆ.",
                "ಸಮಗ್ರತೆಯು ದೃಢವಾದ ನೈತಿಕ ತತ್ವಗಳನ್ನು ಪ್ರದರ್ಶಿಸುವ ಅಭ್ಯಾಸವಾಗಿದೆ."
            ],
            "ta": [
                "நேர்மை: உண்மையாக இருக்கும் நிலை. ஒருமைப்பாடு: கடுமையான தார்மீಕ அல்லது நெறிமுறைக் கோட்பாட்டை உறுதியாகக் கடைப்பிடிப்பது.",
                "நேர்மை என்பது தார்மீக குணத்தின் ஒரு அம்சமாகும்.",
                "ஒருமைப்பாடு என்பது உறுதியான நெறிமுறைக் கொள்கைகளைப் பின்பற்றுவதாகும்."
            ],
            "te": [
                "నిజాయితీ: నిజాయితీగా ఉండే పరిస్థితి; సత్యసంధత. సమగ్రత: కఠినమైన నైతిక నియమావళికి కట్టుబడి ఉండటం.",
                "నిజాయితీ అనేది నైతిక స్వభావాన్ని సూచిస్తుంది.",
                "సమగ్రత అనేది నైతిక సూత్రాలకు స్థిరంగా కట్టుబడి ఉండటం."
            ]
        }
    },
    {
        "query_id": 100003,
        "title": "Causes of High Blood Pressure",
        "queries": {
            "en": "what causes high blood pressure?",
            "hi": "उच्च रक्तचाप का क्या कारण है?",
            "kn": "ಅಧಿಕ ರಕ್ತದೊತ್ತಡಕ್ಕೆ ಕಾರಣವೇನು?",
            "ta": "உயர் இரத்த அழுத்தத்திற்கு என்ன காரணம்?",
            "te": "అధిక రక్తపోటుకు కారణం ఏమిటి?"
        },
        "answers": {
            "en": "Common causes of high blood pressure include high salt intake, lack of exercise, obesity, genetics, and chronic stress.",
            "hi": "उच्च रक्तचाप के सामान्य कारणों में अधिक नमक का सेवन, व्यायाम की कमी, मोटापा, आनुवंशिकी और तनाव शामिल हैं।",
            "kn": "ಅಧಿಕ ರಕ್ತದೊತ್ತಡದ ಸಾಮಾನ್ಯ ಕಾರಣಗಳಲ್ಲಿ ಹೆಚ್ಚಿನ ಉಪ್ಪಿನ ಸೇವನೆ, ವ್ಯಾಯಾಮದ ಕೊರತೆ, ಬೊಜ್ಜು ಮತ್ತು ಒತ್ತಡ ಸೇರಿವೆ.",
            "ta": "உயர் இரத்த அழுத்தத்திற்கான பொதுவான காரணங்களில் அதிக உப்பு உட்கொள்ளல், உடற்பயிற்சியின்மை மற்றும் மன அழுத்தம் ஆகியவை அடங்கும்.",
            "te": "అధిక రక్తపోటుకు సాధారణ కారణాలలో అధిక ఉప్పు తీసుకోవడం, వ్యాయామం లేకపోవడం మరియు ఒత్తిడి ఉన్నాయి."
        },
        "passages": {
            "en": [
                "Hypertension, or high blood pressure, develops when the force of blood against artery walls is consistently too high. Primary risk factors include a high-sodium diet, physical inactivity, obesity, aging, and excessive alcohol consumption.",
                "Normal blood pressure for adults is defined as a systolic pressure of less than 120 mmHg and a diastolic pressure of less than 80 mmHg."
            ],
            "hi": [
                "उच्च रक्तचाप तब विकसित होता है जब धमनी की दीवारों के खिलाफ रक्त का बल लगातार बहुत अधिक होता है। प्राथमिक जोखिम कारकों में उच्च सोडियम आहार, शारीरिक निष्क्रियता, मोटापा और तनाव शामिल हैं।",
                "वयस्कों के लिए सामान्य रक्तचाप 120/80 mmHg से कम माना जाता है।"
            ],
            "kn": [
                "ಅಧಿಕ ರಕ್ತದೊತ್ತಡವು ರಕ್ತನಾಳಗಳ ಗೋಡೆಗಳ ವಿರುದ್ಧ ರಕ್ತದ ಒತ್ತಡ ನಿರಂತರವಾಗಿ ಹೆಚ್ಚಾದಾಗ ಉಂಟಾಗುತ್ತದೆ. ಪ್ರಮುಖ ಕಾರಣಗಳು ಉಪ್ಪಿನ ಅಧಿಕ ಬಳಕೆ, ದೈಹಿಕ ಜಡತ್ವ ಮತ್ತು ಬೊಜ್ಜು.",
                "ವಯಸ್ಕರಿಗೆ ಸಾಮಾನ್ಯ ರಕ್ತದೊತ್ತಡವು 120/80 mmHg ಗಿಂತ ಕಡಿಮೆಯಿರುತ್ತದೆ."
            ],
            "ta": [
                "இரத்த நாளங்களின் சுவர்களில் இரத்தத்தின் விசை தொடர்ந்து அதிகமாக இருக்கும்போது உயர் இரத்த அழுத்தம் ஏற்படுகிறது. முக்கிய காரணங்கள் அதிக உப்பு, உடல் உழைப்பின்மை மற்றும் உடல் பருமன்.",
                "பெரியவர்களுக்கு இயல்பான இரத்த அழுத்தம் 120/80 mmHg க்கும் குறைவாக இருக்க வேண்டும்."
            ],
            "te": [
                "ధమనుల గోడలపై రక్త పీడనం నిరంతరం ఎక్కువగా ఉన్నప్పుడు అధిక రక్తపోటు వస్తుంది. అధిక ఉప్పు, శారీరక శ్రమ లేకపోవడం మరియు ఊబకాయం ప్రధాన కారణాలు.",
                "పెద్దలకు సాధారణ రక్తపోటు 120/80 mmHg కంటే తక్కువగా ఉండాలి."
            ]
        }
    },
    {
        "query_id": 100001,
        "title": "Capital of France",
        "queries": {
            "en": "what is the capital of France?",
            "hi": "फ्रांस की राजधानी क्या है?",
            "kn": "ಫ್ರಾನ್ಸ್‌ನ ರಾಜಧಾನಿ ಯಾವುದು?",
            "ta": "பிரான்சின் தலைநகரம் எது?",
            "te": "ఫ్రాన్స్ రాజధాని ఏది?"
        },
        "answers": {
            "en": "Paris is the capital and most populous city of France.",
            "hi": "पेरिस फ्रांस की राजधानी और सबसे अधिक आबादी वाला शहर है।",
            "kn": "ಪ್ಯಾರಿಸ್ ಫ್ರಾನ್ಸ್‌ನ ರಾಜಧಾನಿ ಮತ್ತು ಅತ್ಯಂತ ಹೆಚ್ಚು ಜನಸಂಖ್ಯೆ ಹೊಂದಿರುವ ನಗರವಾಗಿದೆ.",
            "ta": "பாரிஸ் பிரான்சின் தலைநகரமும் அதிக மக்கள் தொகை கொண்ட நகரமும் ஆகும்.",
            "te": "పారిస్ ఫ్రాన్స్ రాజధాని మరియు అత్యధిక జనాభా కలిగిన నగరం."
        },
        "passages": {
            "en": [
                "Paris is the capital and most populous city of France, with an estimated population of over 2.1 million residents in an area of 105 square kilometers.",
                "France is a country located in Western Europe. Its largest cultural and financial hub is Paris, home to the Louvre and Eiffel Tower."
            ],
            "hi": [
                "पेरिस फ्रांस की राजधानी और सबसे अधिक आबादी वाला शहर है, जिसकी अनुमानित जनसंख्या 21 लाख से अधिक है।",
                "फ्रांस पश्चिमी यूरोप में स्थित एक देश है। इसका सबसे बड़ा सांस्कृतिक और वित्तीय केंद्र पेरिस है।"
            ],
            "kn": [
                "ಪ್ಯಾರಿಸ್ ಫ್ರಾನ್ಸ್‌ನ ರಾಜಧಾನಿ ಮತ್ತು ಅತ್ಯಂತ ಹೆಚ್ಚು ಜನಸಂಖ್ಯೆ ಹೊಂದಿರುವ ನಗರವಾಗಿದೆ.",
                "ಫ್ರಾನ್ಸ್ ಪಶ್ಚಿಮ ಯುರೋಪಿನಲ್ಲಿರುವ ಒಂದು ದೇಶವಾಗಿದೆ. ಇದರ ಪ್ರಮುಖ ಸಾಂಸ್ಕೃತಿಕ ಕೇಂದ್ರ ಪ್ಯಾರಿಸ್."
            ],
            "ta": [
                "பாரிஸ் பிரான்சின் தலைநகரமும் அதிக மக்கள் தொகை கொண்ட நகரமும் ஆகும்.",
                "பிரான்ஸ் மேற்கு ஐரோப்பாவில் அமைந்துள்ள ஒரு நாடு. அதன் கலாச்சார மையம் பாரிஸ் ஆகும்."
            ],
            "te": [
                "పారిస్ ఫ్రాన్స్ రాజధాని మరియు అత్యధిక జనాభా కలిగిన నగరం.",
                "ఫ్రాన్స్ పశ్చిమ ఐరోపాలో ఉన్న ఒక దేశం. దీని అతిపెద్ద సాంస్కృతిక కేంద్రం ప్యారిస్."
            ]
        }
    },
    {
        "query_id": 1102431,
        "title": "Rachel Carson Obligation to Endure",
        "queries": {
            "en": "why did rachel carson write an obligation to endure",
            "hi": "राचेल कार्सन ने एन ऑब्लिगेशन टू एंड्योर क्यों लिखी",
            "kn": "ರಾಚೆಲ್ ಕಾರ್ಸನ್ ಆಬ್ಲಿಗೇಷನ್ ಟು ಎಂಡ್ಯೂರ್ ಏಕೆ ಬರೆದರು",
            "ta": "ரேச்சல் கார்சன் ஏன் ஆன் ஆப்ளிகேஷன் டு என்டியூர் எழுதினார்",
            "te": "రాచెల్ కార్సన్ యాన్ ఆబ్లిగేషన్ టు ఎండ్యూర్ ఎందుకు రాశారు"
        },
        "answers": {
            "en": "Rachel Carson wrote The Obligation to Endure because she believed that human attempts to eliminate unwanted insects and weeds were polluting the environment and causing greater harm.",
            "hi": "राचेल कार्सन ने द ऑब्लिगेशन टू एंड्योर इसलिए लिखा क्योंकि उनका मानना था कि अवांछित कीड़ों को खत्म करने के प्रयास पर्यावरण को प्रदूषित कर रहे हैं।",
            "kn": "ಕೀಟಗಳನ್ನು ನಾಶಮಾಡುವ ಮಾನವನ ಪ್ರಯತ್ನಗಳು ಪರಿಸರವನ್ನು ಕಲುಷಿತಗೊಳಿಸುತ್ತಿವೆ ಎಂದು ನಂಬಿದ್ದರಿಂದ ರಾಚೆಲ್ ಕಾರ್ಸನ್ ಇದನ್ನು ಬರೆದರು.",
            "ta": "பூச்சிகளை அழிக்க மனிதன் எடுக்கும் முயற்சிகள் சுற்றுச்சூழலை மாசுபடுத்தி பெரும் தீங்கை ஏற்படுத்துகின்றன என்று ரேச்சல் கார்சன் நம்பியதால் இதை எழுதினார்.",
            "te": "తెగుళ్లను నిర్మూలించే మానవ ప్రయత్నాలు పర్యావరణాన్ని కలుషితం చేస్తున్నాయని నమ్మి రాచెల్ కార్సన్ దీనిని రాశారు."
        },
        "passages": {
            "en": [
                "Rachel Carson writes The Obligation to Endure because believes that as man tries to eliminate unwanted insects and weeds, however he is actually causing more problems by polluting the environment."
            ],
            "hi": [
                "राचेल कार्सन ने द ऑब्लिगेशन टू एंड्योर इसलिए लिखा क्योंकि उनका मानना था कि अवांछित कीड़ों को खत्म करने के प्रयास पर्यावरण को प्रदूषित कर रहे हैं।"
            ],
            "kn": [
                "ಕೀಟಗಳನ್ನು ನಾಶಮಾಡುವ ಮಾನವನ ಪ್ರಯತ್ನಗಳು ಪರಿಸರವನ್ನು ಕಲುಷಿತಗೊಳಿಸುತ್ತಿವೆ ಎಂದು ನಂಬಿದ್ದರಿಂದ ರಾಚೆಲ್ ಕಾರ್ಸನ್ ಇದನ್ನು ಬರೆದರು."
            ],
            "ta": [
                "பூச்சிகளை அழிக்க மனிதன் எடுக்கும் முயற்சிகள் சுற்றுச்சூழலை மாசுபடுத்தி பெரும் தீங்கை ஏற்படுத்துகின்றன என்று ரேச்சல் கார்சன் நம்பியதால் இதை எழுதினார்."
            ],
            "te": [
                "తెగుళ్లను నిర్మూలించే మానవ ప్రయత్నాలు పర్యావరణాన్ని కలుషితం చేస్తున్నాయని నమ్మి రాచెల్ కార్సన్ దీనిని రాశారు."
            ]
        }
    },
    {
        "query_id": 300122,
        "title": "Frank Gifford Marriages",
        "queries": {
            "en": "how many women did frank gifford marry",
            "hi": "फ्रैंक गिफोर्ड ने कितनी महिलाओं से शादी की",
            "kn": "ಫ್ರಾಂಕ್ ಗಿಫೋರ್ಡ್ ಎಷ್ಟು ಮಹಿಳೆಯರನ್ನು ಮದುವೆಯಾದರು",
            "ta": "பிராங்க் கிஃபோர்ட் எத்தனை பெண்களை திருமணம் செய்து கொண்டார்",
            "te": "ఫ్రాంక్ గిఫోర్డ్ ఎంతమంది మహిళలను వివాహం చేసుకున్నారు"
        },
        "answers": {
            "en": "Frank Gifford married to three women: Maxine Avis Ewart, Astrid Lindley and Kathie Lee Johnson.",
            "hi": "फ्रैंक गिफोर्ड ने तीन महिलाओं से शादी की: मैक्सिन एविस इवार्ट, एस्ट्रिड लिंडले और कैथी ली जॉनसन।",
            "kn": "ಫ್ರಾಂಕ್ ಗಿಫೋರ್ಡ್ ಮೂರು ಮಹಿಳೆಯರನ್ನು ವಿವಾಹವಾದರು: ಮ್ಯಾಕ್ಸಿನ್ ಅವಿಸ್ ಎವಾರ್ಟ್, ಆಸ್ಟ್ರಿಡ್ ಲಿಂಡ್ಲೆ ಮತ್ತು ಕ್ಯಾಥಿ ಲೀ ಜಾನ್ಸನ್.",
            "ta": "பிராங்க் கிஃபோர்ட் மூன்று பெண்களை மணந்தார்: மாக்சின் அவிஸ் எவார்ட், ஆஸ்ட்ரிட் லிண்ட்லி மற்றும் கேத்தி லீ ஜான்சன்.",
            "te": "ఫ్రాంక్ గిఫోర్డ్ ముగ్గురు మహిళలను వివాహం చేసుకున్నారు: మాక్సిన్ అవిస్ ఎవార్ట్, ఆస్ట్రిడ్ లిండ్లీ మరియు కాథీ లీ జాన్సన్."
        },
        "passages": {
            "en": [
                "Frank Gifford married to three women: Maxine Avis Ewart, Astrid Lindley and Kathie Lee Johnson."
            ],
            "hi": [
                "फ्रैंक गिफोर्ड ने तीन महिलाओं से शादी की: मैक्सिन एविस इवार्ट, एस्ट्रिड लिंडले और कैथी ली जॉनसन।"
            ],
            "kn": [
                "ಫ್ರಾಂಕ್ ಗಿಫೋರ್ಡ್ ಮೂರು ಮಹಿಳೆಯರನ್ನು ವಿವಾಹವಾದರು: ಮ್ಯಾಕ್ಸಿನ್ ಅವಿಸ್ ಎವಾರ್ಟ್, ಆಸ್ಟ್ರಿಡ್ ಲಿಂಡ್ಲೆ ಮತ್ತು ಕ್ಯಾಥಿ ಲೀ ಜಾನ್ಸನ್."
            ],
            "ta": [
                "பிராங்க் கிஃபோர்ட் மூன்று பெண்களை மணந்தார்: மாக்சின் அவிஸ் எவார்ட், ஆஸ்ட்ரிட் லிண்ட்லி மற்றும் கேத்தி லೀ ஜான்சன்."
            ],
            "te": [
                "ఫ్రాంక్ గిఫోర్డ్ ముగ్గురు మహిళలను వివాహం చేసుకున్నారు: మాక్సిన్ అవిస్ ఎవార్ట్, ఆస్ట్రిడ్ లిండ్లీ మరియు కాథీ లీ జాన్సన్."
            ]
        }
    }
]

def index_suite():
    print("=== INDEXING 5-LANGUAGE DEMO SUITE INTO PRIMARY DATABASE ===")
    embedder = MultilingualEmbedder()
    vstore = QdrantVectorStore(dimension=embedder.dimension)
    bm25 = BM25LexicalRetriever()

    all_chunks = []
    texts_to_embed = []

    for item in DEMO_MULTILINGUAL_RECORDS:
        qid = item["query_id"]
        title = item["title"]
        queries = item["queries"]
        answers = item["answers"]
        passages_by_lang = item["passages"]

        for lang, pass_list in passages_by_lang.items():
            for p_idx, p_text in enumerate(pass_list):
                c_id = f"demo_{qid}_{lang}_{p_idx}"
                chunk = Chunk(
                    chunk_id=c_id,
                    document_id=f"doc_{qid}_{lang}",
                    parent_document_id=f"doc_{qid}",
                    text=p_text,
                    source="ai4bharat/MSMARCO-XI",
                    language=lang,
                    title=f"{title} ({lang})",
                    dataset_split="validation",
                    chunking_strategy="sentence",
                    chunk_position=p_idx,
                    token_count=len(p_text.split()),
                    character_count=len(p_text),
                    is_ground_truth=(p_idx == 0),
                    metadata={
                        "query_id": qid,
                        "passage_index": p_idx,
                        "is_selected": (p_idx == 0),
                        "related_query_en": queries.get("en", ""),
                        "related_query_indic": queries.get(lang, queries.get("hi", "")),
                        "related_answer_en": answers.get("en", ""),
                        "related_answer_indic": answers.get(lang, answers.get("hi", "")),
                        "multilingual_queries": queries,
                        "multilingual_answers": answers
                    }
                )
                all_chunks.append(chunk)
                texts_to_embed.append(p_text)

    print(f"Generating embeddings for {len(all_chunks)} multilingual passage chunks...")
    embeddings = embedder.embed_texts(texts_to_embed)
    
    print(f"Upserting chunks to Qdrant collection '{settings.QDRANT_COLLECTION_NAME}'...")
    vstore.upsert_chunks(all_chunks, embeddings)
    
    print(f"Indexing chunks in BM25 Lexical store...")
    bm25.index_chunks(all_chunks)

    print(f"SUCCESS: Indexed {len(all_chunks)} multilingual chunks across English, Hindi, Kannada, Tamil, Telugu.")

if __name__ == "__main__":
    index_suite()
