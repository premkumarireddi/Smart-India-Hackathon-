const API_BASE = window.RETINASCAN_API_BASE || "http://localhost:8000";

const els = {
  dropzone: document.getElementById("dropzone"),
  dropzoneEmpty: document.getElementById("dropzoneEmpty"),
  fileInput: document.getElementById("fileInput"),
  previewImg: document.getElementById("previewImg"),
  screenBtn: document.getElementById("screenBtn"),
  errorMsg: document.getElementById("errorMsg"),
  resultsPanel: document.getElementById("resultsPanel"),
  rejectedBlock: document.getElementById("rejectedBlock"),
  rejectReasons: document.getElementById("rejectReasons"),
  acceptedBlock: document.getElementById("acceptedBlock"),
  enhancedImg: document.getElementById("enhancedImg"),
  gradcamImg: document.getElementById("gradcamImg"),
  severityBadge: document.getElementById("severityBadge"),
  severityDesc: document.getElementById("severityDesc"),
  confidenceText: document.getElementById("confidenceText"),
  recommendationBox: document.getElementById("recommendationBox"),
  probsBars: document.getElementById("probsBars"),
  qualityJson: document.getElementById("qualityJson"),
  apiDot: document.getElementById("apiDot"),
  apiStatusText: document.getElementById("apiStatusText"),
};

let selectedFile = null;

async function checkApiHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    const data = await res.json();
    if (data.model_loaded) {
      els.apiDot.className = "dot online";
      const trainedOn = (data.trained_on || "").includes("synthetic")
        ? "synthetic demo data"
        : (data.trained_on || "").includes("aptos")
        ? "real APTOS data"
        : "unknown data";
      els.apiStatusText.textContent = `backend online (${data.architecture || "model"}, trained on ${trainedOn})`;
    } else {
      els.apiDot.className = "dot offline";
      els.apiStatusText.textContent = "backend online, model not trained yet — run ml/train.py";
    }
  } catch (e) {
    els.apiDot.className = "dot offline";
    els.apiStatusText.textContent = "backend unreachable — start uvicorn (see README)";
  }
}

function setFile(file) {
  if (!file) return;
  selectedFile = file;
  const url = URL.createObjectURL(file);
  els.previewImg.src = url;
  els.previewImg.hidden = false;
  els.dropzoneEmpty.hidden = true;
  els.screenBtn.disabled = false;
  els.errorMsg.hidden = true;
  els.resultsPanel.hidden = true;
}

els.dropzone.addEventListener("click", () => els.fileInput.click());
els.fileInput.addEventListener("change", (e) => setFile(e.target.files[0]));

["dragenter", "dragover"].forEach((evt) =>
  els.dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    els.dropzone.classList.add("drag-over");
  })
);
["dragleave", "drop"].forEach((evt) =>
  els.dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    els.dropzone.classList.remove("drag-over");
  })
);
els.dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  setFile(file);
});

function severityLevelClass(level) {
  return `level-${level}`;
}

function renderProbabilities(probs) {
  els.probsBars.innerHTML = "";
  probs.forEach((p) => {
    const row = document.createElement("div");
    row.className = "prob-row";
    const pct = Math.round(p.probability * 100);
    row.innerHTML = `
      <span>${p.label.replace("_", " ")}</span>
      <span class="prob-bar-bg"><span class="prob-bar-fill" style="width:${pct}%"></span></span>
      <span>${pct}%</span>
    `;
    els.probsBars.appendChild(row);
  });
}

async function runScreening() {
  if (!selectedFile) return;
  els.screenBtn.disabled = true;
  els.screenBtn.textContent = "Screening…";
  els.errorMsg.hidden = true;

  try {
    const formData = new FormData();
    formData.append("file", selectedFile);

    const res = await fetch(`${API_BASE}/api/predict`, { method: "POST", body: formData });
    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}));
      throw new Error(errBody.detail || `Request failed (${res.status})`);
    }
    const data = await res.json();

    els.resultsPanel.hidden = false;

    if (!data.accepted) {
      els.rejectedBlock.hidden = false;
      els.acceptedBlock.hidden = true;
      els.rejectReasons.innerHTML = "";
      data.quality.reasons.forEach((r) => {
        const li = document.createElement("li");
        li.textContent = r;
        els.rejectReasons.appendChild(li);
      });
      return;
    }

    els.rejectedBlock.hidden = true;
    els.acceptedBlock.hidden = false;

    els.enhancedImg.src = `data:image/png;base64,${data.enhanced_image_png_base64}`;
    els.gradcamImg.src = `data:image/png;base64,${data.gradcam_overlay_png_base64}`;

    els.severityBadge.textContent = `Level ${data.severity_level}`;
    els.severityBadge.className = `severity-badge ${severityLevelClass(data.severity_level)}`;
    els.severityDesc.textContent = `${data.severity_name.replace("_", " ")} — ${data.severity_description}`;
    els.confidenceText.textContent = `Model confidence: ${(data.confidence * 100).toFixed(1)}%`;

    els.recommendationBox.textContent = data.recommendation;
    els.recommendationBox.style.borderLeftColor = data.is_referable ? "#c0392b" : "#1e7b34";

    renderProbabilities(data.class_probabilities);
    els.qualityJson.textContent = JSON.stringify(data.quality, null, 2);
  } catch (e) {
    els.errorMsg.textContent = e.message || "Something went wrong.";
    els.errorMsg.hidden = false;
  } finally {
    els.screenBtn.disabled = false;
    els.screenBtn.textContent = "Run Screening";
  }
}

els.screenBtn.addEventListener("click", runScreening);

checkApiHealth();
setInterval(checkApiHealth, 15000);
