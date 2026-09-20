const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const previewImg = document.getElementById('preview-img');
const dropzoneEmpty = document.getElementById('dropzone-empty');
const analyzeBtn = document.getElementById('analyze-btn');
const statusMsg = document.getElementById('status-msg');

const resultsEmpty = document.getElementById('results-empty');
const resultsContent = document.getElementById('results-content');
const verdictDot = document.getElementById('verdict-dot');
const verdictClass = document.getElementById('verdict-class');
const verdictConfidence = document.getElementById('verdict-confidence');
const probBars = document.getElementById('prob-bars');

let selectedFile = null;
const STORAGE_KEY = 'neuroscan_last_result';

// ---- File selection (click or drag-drop) ----

dropzone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => {
  if (e.target.files.length) handleFile(e.target.files[0]);
});

['dragenter', 'dragover'].forEach(evt => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add('drag-over');
  });
});

['dragleave', 'drop'].forEach(evt => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove('drag-over');
  });
});

dropzone.addEventListener('drop', (e) => {
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

function handleFile(file) {
  if (!file.type.match(/image\/(png|jpe?g)/)) {
    setStatus('Please choose a PNG or JPG image.', true);
    return;
  }

  selectedFile = file;
  setStatus('');

  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewImg.classList.add('visible');
    dropzoneEmpty.classList.add('hidden');
  };
  reader.readAsDataURL(file);

  analyzeBtn.disabled = false;
  resetResults();
}

// ---- Analyze ----

analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  analyzeBtn.disabled = true;
  setStatus('Analyzing scan…');

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const res = await fetch('/predict', { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok) {
      setStatus(data.error || 'Something went wrong.', true);
      analyzeBtn.disabled = false;
      return;
    }

    showResults(data);
    setStatus('');

    // Save so this survives navigating to /performance and back to this
    // same tab - sessionStorage persists per-tab until the tab is closed.
    try {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify({ previewSrc: previewImg.src, data }));
    } catch (storageErr) {
      // Quota exceeded or storage disabled - not critical, just skip persistence
    }
  } catch (err) {
    setStatus('Could not reach the server. Is app.py running?', true);
  }

  analyzeBtn.disabled = false;
});

function setStatus(msg, isError = false) {
  statusMsg.textContent = msg;
  statusMsg.classList.toggle('error', isError);
}

function resetResults() {
  resultsEmpty.classList.remove('hidden');
  resultsContent.classList.add('hidden');
}

function showResults(data) {
  resultsEmpty.classList.add('hidden');
  resultsContent.classList.remove('hidden');

  verdictClass.textContent = data.predicted_class;
  verdictConfidence.textContent = `${data.confidence}%`;
  verdictDot.classList.toggle('alert', data.is_tumor);

  const badge = document.getElementById('confidence-level-badge');
  badge.textContent = `${data.confidence_level} CONFIDENCE`;
  badge.className = 'confidence-level-badge level-' + data.confidence_level.toLowerCase();

  const reliabilityBanner = document.getElementById('reliability-banner');
  const reliabilityText = document.getElementById('reliability-text');
  if (data.low_confidence) {
    reliabilityText.textContent =
      'Low confidence result — neither model was very sure about this one. Treat this prediction with caution.';
    reliabilityBanner.classList.remove('hidden');
  } else {
    reliabilityBanner.classList.add('hidden');
  }

  const agreementNote = document.getElementById('agreement-note');
  if (data.models_agree === false) {
    agreementNote.textContent =
      `The two models disagreed: baseline CNN predicted "${data.model_breakdown.baseline_cnn}", ` +
      `transfer learning predicted "${data.model_breakdown.transfer_learning}". Shown result is their averaged prediction.`;
  } else {
    agreementNote.textContent = 'Both models agree on this prediction.';
  }

  const gradcamSection = document.getElementById('gradcam-section');
  const gradcamImgBaseline = document.getElementById('gradcam-img-baseline');
  const gradcamImgTransfer = document.getElementById('gradcam-img-transfer');
  const occlusionImgBaseline = document.getElementById('occlusion-img-baseline');
  const hasAnyExplainability = data.gradcam_image_baseline || data.gradcam_image_transfer || data.occlusion_image_baseline;
  if (hasAnyExplainability) {
    if (data.gradcam_image_baseline) gradcamImgBaseline.src = data.gradcam_image_baseline;
    if (data.gradcam_image_transfer) gradcamImgTransfer.src = data.gradcam_image_transfer;
    if (data.occlusion_image_baseline) occlusionImgBaseline.src = data.occlusion_image_baseline;
    gradcamSection.classList.remove('hidden');
  } else {
    gradcamSection.classList.add('hidden');
  }

  // Sort classes by probability, descending
  const sorted = Object.entries(data.probabilities).sort((a, b) => b[1] - a[1]);
  const topClass = sorted[0][0];

  probBars.innerHTML = '';
  sorted.forEach(([label, value]) => {
    const row = document.createElement('div');
    row.className = 'prob-row';
    row.innerHTML = `
      <span>${label}</span>
      <div class="prob-track">
        <div class="prob-fill ${label === topClass ? 'top' : ''}" style="width: ${value}%"></div>
      </div>
      <span class="prob-value">${value}%</span>
    `;
    probBars.appendChild(row);
  });
}

// ---- Restore previous result on page load ----
// If you already analyzed a scan, navigated to /performance, then came
// back to this page, this restores exactly what you had before -
// otherwise this same-tab navigation would show an empty upload screen.
(function restorePreviousResult() {
  try {
    const saved = sessionStorage.getItem(STORAGE_KEY);
    if (!saved) return;

    const { previewSrc, data } = JSON.parse(saved);

    previewImg.src = previewSrc;
    previewImg.classList.add('visible');
    dropzoneEmpty.classList.add('hidden');

    showResults(data);
  } catch (err) {
    // Corrupt or missing saved state - just show the normal empty upload screen
  }
})();
