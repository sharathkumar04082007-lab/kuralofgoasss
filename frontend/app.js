// Production Voice RAG Frontend Client (HH Goa 2026)

let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;
let audioContext = null;
let analyser = null;
let animationId = null;

// DOM Elements
const micButton = document.getElementById('mic-button');
const micRing = document.getElementById('mic-ring');
const micStatusText = document.getElementById('mic-status-text');
const waveformCanvas = document.getElementById('audio-waveform');
const canvasCtx = waveformCanvas ? waveformCanvas.getContext('2d') : null;

const textQueryForm = document.getElementById('text-query-form');
const textQueryInput = document.getElementById('text-query-input');
const strategySelect = document.getElementById('chunking-strategy-select');

const transcriptDisplay = document.getElementById('transcript-display');
const answerDisplay = document.getElementById('answer-display');
const groundedBadge = document.getElementById('grounded-badge');
const totalLatencyBadge = document.getElementById('total-latency-badge');
const sourcesList = document.getElementById('sources-list');
const sourceCount = document.getElementById('source-count');

// Latency DOM elements
const barStt = document.getElementById('bar-stt');
const barGuard = document.getElementById('bar-guard');
const barRet = document.getElementById('bar-ret');
const barGen = document.getElementById('bar-gen');
const barGround = document.getElementById('bar-ground');

const latValStt = document.getElementById('lat-val-stt');
const latValGuard = document.getElementById('lat-val-guard');
const latValRet = document.getElementById('lat-val-ret');
const latValGen = document.getElementById('lat-val-gen');
const latValGround = document.getElementById('lat-val-ground');

// Initialize System Health
async function checkSystemHealth() {
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    const dot = document.getElementById('system-status-dot');
    const label = document.getElementById('system-status-label');
    if (data.status === 'healthy') {
      const count = data.vector_store?.points_count || data.bm25_indexed_chunks || 0;
      dot.style.background = '#10b981';
      label.textContent = `Qdrant Ready (${count} Chunks)`;
    } else {
      dot.style.background = '#f59e0b';
      label.textContent = 'System Initializing';
    }
  } catch (err) {
    console.warn('Health check failed:', err);
  }
}

// Draw Audio Waveform
function drawWaveform() {
  if (!analyser || !canvasCtx) return;
  const bufferLength = analyser.fftSize;
  const dataArray = new Uint8Array(bufferLength);
  
  function render() {
    if (!isRecording) {
      canvasCtx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
      return;
    }
    animationId = requestAnimationFrame(render);
    analyser.getByteTimeDomainData(dataArray);

    canvasCtx.fillStyle = 'rgba(15, 21, 36, 0.5)';
    canvasCtx.fillRect(0, 0, waveformCanvas.width, waveformCanvas.height);
    canvasCtx.lineWidth = 2;
    canvasCtx.strokeStyle = '#6366f1';
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

// Start Voice Recording
async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 256;
    const source = audioContext.createMediaStreamSource(stream);
    source.connect(analyser);

    audioChunks = [];
    mediaRecorder = new MediaRecorder(stream);
    
    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        audioChunks.push(event.data);
      }
    };

    mediaRecorder.onstop = async () => {
      const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
      await sendVoiceQuery(audioBlob);
      stream.getTracks().forEach(track => track.stop());
    };

    mediaRecorder.start();
    isRecording = true;
    micRing.classList.add('recording');
    micStatusText.textContent = 'Listening... Click to finish speaking';
    drawWaveform();
  } catch (err) {
    console.error('Microphone error:', err);
    micStatusText.textContent = 'Microphone permission denied or unavailable. You can use text input.';
  }
}

// Stop Voice Recording
function stopRecording() {
  if (mediaRecorder && isRecording) {
    mediaRecorder.stop();
    isRecording = false;
    micRing.classList.remove('recording');
    micStatusText.textContent = 'Transcribing and querying RAG pipeline...';
    if (animationId) cancelAnimationFrame(animationId);
  }
}

// Mic Button Click Toggle
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
async function sendVoiceQuery(audioBlob) {
  const formData = new FormData();
  formData.append('audio', audioBlob, 'query.wav');
  formData.append('chunking_strategy', strategySelect.value);
  formData.append('top_k', 5);

  setLoadingState('Transcribing speech...');

  try {
    const res = await fetch('/api/voice/query', {
      method: 'POST',
      body: formData
    });

    if (!res.ok) {
      throw new Error(`Voice query failed with status ${res.status}`);
    }

    const data = await res.json();
    renderResponse(data);
  } catch (err) {
    renderError(err.message);
  } finally {
    micStatusText.textContent = 'Click microphone to speak (or use text box below)';
  }
}

// Send Text Query to Backend
async function sendTextQuery(queryText) {
  if (!queryText || !queryText.trim()) return;

  setLoadingState('Retrieving and generating answer...');

  try {
    const res = await fetch('/api/text/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: queryText.trim(),
        chunking_strategy: strategySelect.value,
        top_k: 5
      })
    });

    if (!res.ok) {
      throw new Error(`Text query failed with status ${res.status}`);
    }

    const data = await res.json();
    renderResponse(data);
  } catch (err) {
    renderError(err.message);
  }
}

// Text Form Submission Handler
if (textQueryForm) {
  textQueryForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const q = textQueryInput.value;
    sendTextQuery(q);
  });
}

// Preset Buttons Handler
document.querySelectorAll('.pill-btn').forEach((btn) => {
  btn.addEventListener('click', () => {
    const query = btn.getAttribute('data-query');
    if (textQueryInput) textQueryInput.value = query;
    sendTextQuery(query);
  });
});

// Set UI Loading State
function setLoadingState(message) {
  transcriptDisplay.textContent = message;
  answerDisplay.innerHTML = `<p class="placeholder-text" style="color:#6366f1;">Processing request through pipeline...</p>`;
  groundedBadge.className = 'badge';
  groundedBadge.textContent = 'Processing';
  totalLatencyBadge.textContent = '...';
}

// Render Completed Response
function renderResponse(data) {
  // 1. Transcript & Answer
  transcriptDisplay.textContent = data.transcript || '(No transcript)';
  answerDisplay.innerHTML = `<p>${escapeHtml(data.answer)}</p>`;

  // 2. Grounding & Status Badges
  if (data.grounded) {
    groundedBadge.className = 'badge badge-grounded';
    groundedBadge.textContent = `Grounded (${Math.round(data.confidence * 100)}%)`;
  } else if (data.query_classification === 'off_topic') {
    groundedBadge.className = 'badge badge-unsupported';
    groundedBadge.textContent = 'Off-Topic Refusal';
  } else if (data.query_classification === 'unsafe') {
    groundedBadge.className = 'badge badge-unsupported';
    groundedBadge.textContent = 'Safety Blocked';
  } else {
    groundedBadge.className = 'badge badge-unsupported';
    groundedBadge.textContent = 'Unverified / Refusal';
  }

  // 3. Latency Waterfall
  const lat = data.latency_ms || {};
  totalLatencyBadge.textContent = `${lat.total_ms || 0} ms`;

  const maxBudget = Math.max(200, lat.total_ms || 200);

  updateBar(barStt, latValStt, lat.stt_ms || 0, maxBudget);
  updateBar(barGuard, latValGuard, lat.guardrails_ms || 0, maxBudget);
  updateBar(barRet, latValRet, lat.retrieval_ms || 0, maxBudget);
  updateBar(barGen, latValGen, lat.generation_ms || 0, maxBudget);
  updateBar(barGround, latValGround, lat.grounding_ms || 0, maxBudget);

  // 4. Render Sources List
  const sources = data.sources || [];
  sourceCount.textContent = sources.length;

  if (sources.length === 0) {
    sourcesList.innerHTML = `<div class="source-card empty-source">No context sources retrieved.</div>`;
  } else {
    sourcesList.innerHTML = sources.map((s, idx) => `
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
}

function updateBar(barEl, valEl, valMs, maxBudget) {
  if (!barEl || !valEl) return;
  const pct = Math.min(100, (valMs / maxBudget) * 100);
  barEl.style.width = `${Math.max(2, pct)}%`;
  valEl.textContent = `${valMs} ms`;
}

function renderError(errMsg) {
  transcriptDisplay.textContent = 'Error during request execution';
  answerDisplay.innerHTML = `<p style="color:#ef4444;">${escapeHtml(errMsg)}</p>`;
  groundedBadge.className = 'badge badge-unsupported';
  groundedBadge.textContent = 'Error';
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

// Initial health poll
checkSystemHealth();
setInterval(checkSystemHealth, 10000);
