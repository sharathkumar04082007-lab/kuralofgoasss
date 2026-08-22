// Production Voice RAG Frontend Client (HH Goa 2026)
// Complete Audio In -> Real STT -> RAG -> Real TTS -> Audio Out

let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let audioContext = null;
let analyser = null;
let animationId = null;
let activeAudioElement = null;
let lastSynthesizedText = "";
let lastAudioBase64 = null;
let recordingStartTime = 0;

// DOM Elements
const micButton = document.getElementById('mic-button');
const micRing = document.getElementById('mic-ring');
const micStatusText = document.getElementById('mic-status-text');
const micSubHint = document.getElementById('mic-sub-hint');
const waveformCanvas = document.getElementById('audio-waveform');
const canvasCtx = waveformCanvas ? waveformCanvas.getContext('2d') : null;

const languageSelect = document.getElementById('language-select');

const textQueryForm = document.getElementById('text-query-form');
const textQueryInput = document.getElementById('text-query-input');

const transcriptDisplay = document.getElementById('transcript-display');
const answerDisplay = document.getElementById('answer-display');
const detectedLangBadge = document.getElementById('detected-lang-badge');
const groundedBadge = document.getElementById('grounded-badge');
const totalLatencyBadge = document.getElementById('total-latency-badge');
const sourcesList = document.getElementById('sources-list');
const sourceCount = document.getElementById('source-count');

const audioPlayerPanel = document.getElementById('audio-player-panel');
const audioStatusText = document.getElementById('audio-status-text');
const autoSpeakToggle = document.getElementById('auto-speak-toggle');
const playPauseBtn = document.getElementById('play-pause-btn');
const playPauseIcon = document.getElementById('play-pause-icon');
const playPauseText = document.getElementById('play-pause-text');
const replayBtn = document.getElementById('replay-btn');

// Latency DOM elements
const barStt = document.getElementById('bar-stt');
const barGuard = document.getElementById('bar-guard');
const barRet = document.getElementById('bar-ret');
const barGen = document.getElementById('bar-gen');
const barGround = document.getElementById('bar-ground');
const barTts = document.getElementById('bar-tts');

const latValStt = document.getElementById('lat-val-stt');
const latValGuard = document.getElementById('lat-val-guard');
const latValRet = document.getElementById('lat-val-ret');
const latValGen = document.getElementById('lat-val-gen');
const latValGround = document.getElementById('lat-val-ground');
const latValTts = document.getElementById('lat-val-tts');

// Supported MIME types detection for MediaRecorder
function getSupportedMimeType() {
  const types = [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/ogg;codecs=opus',
    'audio/ogg',
    'audio/mp4',
    'audio/wav'
  ];
  for (const t of types) {
    if (window.MediaRecorder && MediaRecorder.isTypeSupported(t)) {
      return t;
    }
  }
  return '';
}

// System Health Poll
async function checkSystemHealth() {
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    const dot = document.getElementById('system-status-dot');
    const label = document.getElementById('system-status-label');
    if (data.status === 'healthy') {
      const count = data.vector_store?.vector_count || data.bm25_indexed_chunks || 0;
      dot.style.background = '#10b981';
      dot.style.boxShadow = '0 0 8px #10b981';
      const stt = data.stt_provider || 'STT';
      label.textContent = `Qdrant Ready (${count} Chunks • ${stt})`;
    } else {
      dot.style.background = '#f59e0b';
      label.textContent = 'System Initializing';
    }
  } catch (err) {
    console.warn('Health check poll failed:', err);
  }
}

// Waveform Animation
function drawWaveform() {
  if (!analyser || !canvasCtx || !waveformCanvas) return;
  const bufferLength = analyser.fftSize;
  const dataArray = new Uint8Array(bufferLength);
  
  function render() {
    if (!isRecording) {
      canvasCtx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
      return;
    }
    animationId = requestAnimationFrame(render);
    analyser.getByteTimeDomainData(dataArray);

    canvasCtx.fillStyle = 'rgba(15, 23, 42, 0.4)';
    canvasCtx.fillRect(0, 0, waveformCanvas.width, waveformCanvas.height);
    canvasCtx.lineWidth = 2.5;
    canvasCtx.strokeStyle = '#38bdf8';
    canvasCtx.beginPath();

    const sliceWidth = waveformCanvas.width * 1.0 / bufferLength;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
      const v = dataArray[i] / 128.0;
      const y = v * (waveformCanvas.height / 2);

      if (i === 0) canvasCtx.moveTo(x, y);
      else canvasCtx.lineTo(x, y);

      x += sliceWidth;
    }
    canvasCtx.lineTo(waveformCanvas.width, waveformCanvas.height / 2);
    canvasCtx.stroke();
  }
  render();
}

let speechRecognitionInstance = null;
let liveTranscribedText = "";

// Microphone Recording
async function startRecording() {
  stopCurrentAudio();
  liveTranscribedText = "";
  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error('Your browser does not support audio recording API.');
    }

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true
      }
    });

    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 256;
    const source = audioContext.createMediaStreamSource(stream);
    source.connect(analyser);

    const mimeType = getSupportedMimeType();
    const options = mimeType ? { mimeType } : {};
    
    audioChunks = [];
    mediaRecorder = new MediaRecorder(stream, options);
    
    mediaRecorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        audioChunks.push(event.data);
      }
    };

    recordingStartTime = Date.now();
    mediaRecorder.onstop = async () => {
      const effectiveMime = mediaRecorder.mimeType || 'audio/webm';
      const audioBlob = new Blob(audioChunks, { type: effectiveMime });
      const durationSec = ((Date.now() - recordingStartTime) / 1000).toFixed(2);
      stream.getTracks().forEach(track => track.stop());

      console.log('=== AUDIO RECORDING DEBUG ===');
      console.log(`MIME TYPE: ${effectiveMime}`);
      console.log(`BLOB SIZE: ${audioBlob.size} bytes`);
      console.log(`DURATION: ${durationSec} s`);
      console.log(`RECORDER STATE: ${mediaRecorder ? mediaRecorder.state : 'inactive'}`);
      console.log(`LIVE SPEECH TRANSCRIPT: ${liveTranscribedText}`);

      window.lastRecordingDebug = {
        mimeType: effectiveMime,
        blobSize: audioBlob.size,
        duration: durationSec,
        recorderState: mediaRecorder ? mediaRecorder.state : 'inactive',
        browserTranscript: liveTranscribedText
      };

      if (audioBlob.size < 100 && !liveTranscribedText) {
        setRecordingState('ERROR', 'No audio recorded. Please speak louder into your microphone.');
        return;
      }

      await sendVoiceQuery(audioBlob, effectiveMime);
    };

    // Initialize Browser SpeechRecognition in parallel for real-time transcription
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      try {
        speechRecognitionInstance = new SpeechRecognition();
        speechRecognitionInstance.continuous = true;
        speechRecognitionInstance.interimResults = true;
        
        let selectedLang = (languageSelect && languageSelect.value !== 'unknown') ? languageSelect.value : (navigator.language || 'hi-IN');
        speechRecognitionInstance.lang = selectedLang;
        console.log(`Initialized SpeechRecognition with language: ${selectedLang}`);

        speechRecognitionInstance.onresult = (event) => {
          let current = "";
          for (let i = 0; i < event.results.length; ++i) {
            current += event.results[i][0].transcript;
          }
          liveTranscribedText = current.trim();
          if (liveTranscribedText) {
            transcriptDisplay.textContent = liveTranscribedText;
          }
        };

        speechRecognitionInstance.onerror = (e) => {
          console.warn('SpeechRecognition notice:', e.error);
        };

        speechRecognitionInstance.start();
      } catch (err) {
        console.warn('Could not start SpeechRecognition:', err);
      }
    }

    mediaRecorder.start(250); // Slice chunks every 250ms
    isRecording = true;
    setRecordingState('RECORDING', 'Listening... Click to finish speaking');
    drawWaveform();
  } catch (err) {
    console.error('Microphone access failed:', err);
    let errMsg = 'Microphone permission denied. Please allow microphone access in browser settings.';
    if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
      errMsg = 'No microphone device found on your computer.';
    } else if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
      errMsg = 'Microphone permission was denied. Click the lock icon in your URL bar to allow access.';
    }
    setRecordingState('ERROR', errMsg);
  }
}

function stopRecording() {
  if (speechRecognitionInstance) {
    try { speechRecognitionInstance.stop(); } catch (e) {}
  }

  if (mediaRecorder && isRecording) {
    mediaRecorder.stop();
    isRecording = false;
    setRecordingState('PROCESSING', 'Transcribing speech & searching dataset...');
    if (animationId) cancelAnimationFrame(animationId);
  }
}

function setRecordingState(state, message) {
  if (micStatusText) micStatusText.textContent = message;
  
  if (state === 'RECORDING') {
    micRing.className = 'mic-visualizer-ring recording';
    if (micSubHint) micSubHint.textContent = 'Listening to your voice...';
  } else if (state === 'PROCESSING') {
    micRing.className = 'mic-visualizer-ring processing';
    if (micSubHint) micSubHint.textContent = 'Transcribing speech & querying RAG...';
  } else if (state === 'ERROR') {
    micRing.className = 'mic-visualizer-ring';
    if (micSubHint) micSubHint.textContent = 'Try again or use the text box below';
  } else {
    micRing.className = 'mic-visualizer-ring';
    if (micSubHint) micSubHint.textContent = 'Speak clearly in English, Hindi, or Indian languages';
  }
}

// Mic Button Toggle Handler
if (micButton) {
  micButton.addEventListener('click', () => {
    if (!isRecording) {
      startRecording();
    } else {
      stopRecording();
    }
  });
}

// Send Voice Query to Backend
async function sendVoiceQuery(audioBlob, mimeType) {
  const formData = new FormData();
  
  let extension = 'webm';
  if (mimeType.includes('wav')) extension = 'wav';
  else if (mimeType.includes('ogg')) extension = 'ogg';
  else if (mimeType.includes('mp4')) extension = 'mp4';

  formData.append('audio', audioBlob, `voice_query.${extension}`);
  formData.append('language_code', languageSelect.value);
  if (liveTranscribedText) {
    formData.append('browser_transcript', liveTranscribedText);
  }
  formData.append('top_k', 5);

  setLoadingState(liveTranscribedText ? `Querying: "${liveTranscribedText}"...` : 'Transcribing your voice...');

  try {
    const res = await fetch('/api/voice/query', {
      method: 'POST',
      body: formData
    });

    if (!res.ok) {
      const errJson = await res.json().catch(() => ({}));
      throw new Error(errJson.detail || `Voice request failed with status ${res.status}`);
    }

    const data = await res.json();
    renderResponse(data);
  } catch (err) {
    renderError(err.message);
  } finally {
    setRecordingState('IDLE', 'Click microphone to speak (or use text box below)');
  }
}

// Send Text Query to Backend
async function sendTextQuery(queryText) {
  if (!queryText || !queryText.trim()) return;
  stopCurrentAudio();

  setLoadingState('Retrieving and generating answer...');

  try {
    const res = await fetch('/api/text/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: queryText.trim(),
        top_k: 5
      })
    });

    if (!res.ok) {
      const errJson = await res.json().catch(() => ({}));
      throw new Error(errJson.detail || `Text request failed with status ${res.status}`);
    }

    const data = await res.json();
    renderResponse(data);
  } catch (err) {
    renderError(err.message);
  }
}


// Set UI Loading State
function setLoadingState(message) {
  transcriptDisplay.textContent = message;
  answerDisplay.innerHTML = `<p class="placeholder-text" style="color:#818cf8;">Processing query through Voice RAG pipeline...</p>`;
  if (detectedLangBadge) {
    detectedLangBadge.textContent = '🌐 Detecting...';
    detectedLangBadge.style.color = '#38bdf8';
  }
  groundedBadge.className = 'badge';
  groundedBadge.textContent = 'Processing';
  totalLatencyBadge.textContent = '...';
  audioStatusText.textContent = 'Synthesizing voice...';
}

// Render Response & Spoken Audio Playback
function renderResponse(data) {
  lastSynthesizedText = data.answer || '';
  lastAudioBase64 = data.audio_base64 || null;
  window.lastDetectedLanguage = data.detected_language || 'en';
  window.lastDetectedBcp47 = data.detected_bcp47 || (
    data.detected_language === 'hi' ? 'hi-IN' :
    data.detected_language === 'kn' ? 'kn-IN' :
    data.detected_language === 'ta' ? 'ta-IN' :
    data.detected_language === 'te' ? 'te-IN' : 'en-IN'
  );

  // 1. Transcript & Answer
  transcriptDisplay.textContent = data.transcript || '(Empty speech transcript)';
  answerDisplay.innerHTML = `<p>${escapeHtml(data.answer)}</p>`;

  // 2. Badges
  if (detectedLangBadge) {
    const langName = data.detected_language_name || 'English';
    const confPct = Math.round((data.language_confidence || 1.0) * 100);
    detectedLangBadge.textContent = `🌐 ${langName} (${confPct}%)`;
    detectedLangBadge.title = `Detected Language: ${langName} (${window.lastDetectedBcp47}), Confidence: ${confPct}%`;
  }

  if (data.grounded) {
    groundedBadge.className = 'badge badge-grounded';
    groundedBadge.textContent = `Grounded (${Math.round(data.confidence * 100)}%)`;
  } else if (data.query_classification === 'off_topic') {
    groundedBadge.className = 'badge badge-unsupported';
    groundedBadge.textContent = 'Off-Topic Refusal';
  } else if (data.query_classification === 'unsafe') {
    groundedBadge.className = 'badge badge-unsupported';
    groundedBadge.textContent = 'Safety Blocked';
  } else if (data.query_classification === 'stt_failure') {
    groundedBadge.className = 'badge badge-unsupported';
    groundedBadge.textContent = 'STT Error';
  } else {
    groundedBadge.className = 'badge badge-unsupported';
    groundedBadge.textContent = 'Unverified / Refusal';
  }

  // 3. Latency Waterfall & Hero Banner
  const lat = data.latency_ms || {};
  totalLatencyBadge.textContent = `${lat.total_ms || 0} ms`;

  const heroLatencyVal = document.getElementById('hero-latency-val');
  const latencyBudgetPill = document.getElementById('latency-budget-pill');
  if (heroLatencyVal) {
    heroLatencyVal.innerHTML = `${lat.total_ms || 0} <small>ms</small>`;
  }
  if (latencyBudgetPill) {
    if ((lat.total_ms || 0) <= 200) {
      latencyBudgetPill.textContent = `🎯 Sub-200ms Budget: PASS (${lat.total_ms || 0} ms)`;
      latencyBudgetPill.style.background = '#10b981';
    } else {
      latencyBudgetPill.textContent = `⚠️ Latency: ${lat.total_ms || 0} ms`;
      latencyBudgetPill.style.background = '#f59e0b';
    }
  }

  const maxBudget = Math.max(200, lat.total_ms || 200);

  updateBar(barStt, latValStt, lat.stt_ms || 0, maxBudget);
  updateBar(barGuard, latValGuard, lat.guardrails_ms || 0, maxBudget);
  updateBar(barRet, latValRet, lat.retrieval_ms || 0, maxBudget);
  updateBar(barGen, latValGen, lat.generation_ms || 0, maxBudget);
  updateBar(barGround, latValGround, lat.grounding_ms || 0, maxBudget);
  updateBar(barTts, latValTts, lat.tts_ms || 0, maxBudget);

  // 4. Render Sources List
  const sources = data.sources || [];
  sourceCount.textContent = sources.length;

  if (sources.length === 0) {
    sourcesList.innerHTML = `<div class="source-card empty-source">No context sources retrieved.</div>`;
  } else {
    sourcesList.innerHTML = sources.map((s) => `
      <div class="source-card">
        <div class="source-card-header">
          <div class="source-meta">
            <span class="strategy-tag">${escapeHtml(s.chunking_strategy)}</span>
            <span class="score-tag">Score: ${s.relevance_score}</span>
          </div>
          <span style="font-size:0.75rem; color:#64748b;">${escapeHtml(s.source_id.slice(-8))}</span>
        </div>
        <div class="source-excerpt">${escapeHtml(s.text_excerpt)}</div>
      </div>
    `).join('');
  }

  // 5. Automatic Voice Readout (TTS)
  if (autoSpeakToggle && autoSpeakToggle.checked) {
    playSpokenAnswer();
  } else {
    audioStatusText.textContent = 'Audio ready. Click Play Answer to listen.';
  }
}

// Spoken Audio Execution
function playSpokenAnswer() {
  stopCurrentAudio();
  if (!lastSynthesizedText) return;

  if (lastAudioBase64) {
    // Play native synthesized audio (Sarvam AI / Multilingual Neural TTS)
    try {
      let mimeType = 'audio/mpeg';
      if (lastAudioBase64.startsWith('UklGR')) {
        mimeType = 'audio/wav';
      }
      const audioSrc = `data:${mimeType};base64,` + lastAudioBase64;
      activeAudioElement = new Audio(audioSrc);
      
      const langLabel = window.lastDetectedLanguage ? ` (${window.lastDetectedLanguage.toUpperCase()})` : '';
      audioStatusText.textContent = `Speaking answer${langLabel}...`;
      setAudioBtnPlaying(true);

      activeAudioElement.onended = () => {
        audioStatusText.textContent = 'Audio playback completed.';
        setAudioBtnPlaying(false);
      };

      activeAudioElement.onerror = (e) => {
        console.warn('Audio element playback error, falling back to browser speech:', e);
        fallbackBrowserSpeech(lastSynthesizedText);
      };

      activeAudioElement.play().catch((err) => {
        console.warn('Autoplay notice (click button to play):', err);
        audioStatusText.textContent = 'Click "▶ Play Answer" to listen.';
        setAudioBtnPlaying(false);
      });
    } catch (err) {
      console.warn('Error creating audio element:', err);
      fallbackBrowserSpeech(lastSynthesizedText);
    }
  } else {
    // Native Browser Web Speech API Synthesis Fallback
    fallbackBrowserSpeech(lastSynthesizedText);
  }
}

function fallbackBrowserSpeech(text) {
  if (!('speechSynthesis' in window)) {
    audioStatusText.textContent = 'Voice playback not supported by browser.';
    return;
  }

  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  
  // Exact detected language code mapping
  const targetBcp47 = window.lastDetectedBcp47 || (
    /[\u0C80-\u0CFF]/.test(text) ? 'kn-IN' :
    /[\u0B80-\u0BFF]/.test(text) ? 'ta-IN' :
    /[\u0C00-\u0C7F]/.test(text) ? 'te-IN' :
    /[\u0900-\u097F]/.test(text) ? 'hi-IN' : 'en-IN'
  );

  utterance.lang = targetBcp47;
  utterance.rate = 1.0;

  // Try matching browser installed speech synthesis voice for target language
  if (window.speechSynthesis.getVoices) {
    const voices = window.speechSynthesis.getVoices();
    const langPrefix = targetBcp47.split('-')[0];
    const matchVoice = voices.find(v => v.lang === targetBcp47 || v.lang.startsWith(langPrefix));
    if (matchVoice) {
      utterance.voice = matchVoice;
    }
  }

  audioStatusText.textContent = `Speaking answer (${targetBcp47})...`;
  setAudioBtnPlaying(true);

  utterance.onend = () => {
    audioStatusText.textContent = 'Audio playback completed.';
    setAudioBtnPlaying(false);
  };

  utterance.onerror = () => {
    audioStatusText.textContent = 'Audio playback ended.';
    setAudioBtnPlaying(false);
  };

  window.speechSynthesis.speak(utterance);
}

function stopCurrentAudio() {
  if (activeAudioElement) {
    activeAudioElement.pause();
    activeAudioElement.currentTime = 0;
    activeAudioElement = null;
  }
  if ('speechSynthesis' in window) {
    window.speechSynthesis.cancel();
  }
  setAudioBtnPlaying(false);
}

function setAudioBtnPlaying(isPlaying) {
  if (!playPauseIcon || !playPauseText) return;
  if (isPlaying) {
    playPauseIcon.textContent = '⏸';
    playPauseText.textContent = 'Pause';
  } else {
    playPauseIcon.textContent = '▶';
    playPauseText.textContent = 'Play Answer';
  }
}

// Audio Button Handlers
if (playPauseBtn) {
  playPauseBtn.addEventListener('click', () => {
    if (playPauseIcon.textContent === '⏸') {
      stopCurrentAudio();
      audioStatusText.textContent = 'Audio paused.';
    } else {
      playSpokenAnswer();
    }
  });
}

if (replayBtn) {
  replayBtn.addEventListener('click', () => {
    playSpokenAnswer();
  });
}

function updateBar(barEl, valEl, valMs, maxBudget) {
  if (!barEl || !valEl) return;
  const pct = Math.min(100, (valMs / maxBudget) * 100);
  barEl.style.width = `${Math.max(2, pct)}%`;
  valEl.textContent = `${valMs} ms`;
}

function renderError(errMsg) {
  transcriptDisplay.textContent = 'Error occurred during request';
  answerDisplay.innerHTML = `<p style="color:#ef4444;">${escapeHtml(errMsg)}</p>`;
  groundedBadge.className = 'badge badge-unsupported';
  groundedBadge.textContent = 'Error';
  audioStatusText.textContent = 'Error';
}

function escapeHtml(str) {
  if (!str) return '';
  return str.replace(/[&<>"']/g, function(m) {
    return {
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#39;'
    }[m];
  });
}

// Text Query Form Submission Handler
if (textQueryForm) {
  textQueryForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = textQueryInput ? textQueryInput.value.trim() : '';
    if (query) {
      transcriptDisplay.textContent = query;
      await sendTextQuery(query);
    }
  });
}

// Preset Beach Signpost Pills Handler
document.querySelectorAll('.pill-btn').forEach(btn => {
  btn.addEventListener('click', async () => {
    const query = btn.getAttribute('data-query');
    const pillLang = btn.getAttribute('data-lang');
    if (pillLang && languageSelect) {
      languageSelect.value = pillLang;
      syncLanguageChips(pillLang);
    }
    if (query) {
      if (textQueryInput) {
        textQueryInput.value = query;
      }
      transcriptDisplay.textContent = query;
      await sendTextQuery(query);
    }
  });
});

// Sync 14 Language Chips with Select Dropdown
function syncLanguageChips(selectedCode) {
  document.querySelectorAll('.lang-chip').forEach(chip => {
    if (chip.getAttribute('data-lang-code') === selectedCode) {
      chip.classList.add('active');
    } else {
      chip.classList.remove('active');
    }
  });
}

document.querySelectorAll('.lang-chip').forEach(chip => {
  chip.addEventListener('click', () => {
    const code = chip.getAttribute('data-lang-code');
    if (code && languageSelect) {
      languageSelect.value = code;
      syncLanguageChips(code);
    }
  });
});

if (languageSelect) {
  languageSelect.addEventListener('change', () => {
    syncLanguageChips(languageSelect.value);
  });
}

// Smooth Back to Top Handlers
function scrollToTop() {
  window.scrollTo({
    top: 0,
    behavior: 'smooth'
  });
}

const btnBackToTop = document.getElementById('btn-back-to-top');
const floatingBackToTop = document.getElementById('floating-back-to-top');

if (btnBackToTop) {
  btnBackToTop.addEventListener('click', scrollToTop);
}

if (floatingBackToTop) {
  floatingBackToTop.addEventListener('click', scrollToTop);

  window.addEventListener('scroll', () => {
    if (window.scrollY > 280) {
      floatingBackToTop.classList.add('visible');
    } else {
      floatingBackToTop.classList.remove('visible');
    }
  });
}

// Initial health poll
checkSystemHealth();
setInterval(checkSystemHealth, 10000);


