const DATASETS = {
  names: "./datasets/names.txt",
  dinos: "./datasets/dinos.txt",
  words: "./datasets/words.txt",
  ycstartups: "./datasets/ycstartups.txt",
};

const els = {
  datasetSelect: document.getElementById("datasetSelect"),
  loadPresetBtn: document.getElementById("loadPresetBtn"),
  loadUploadBtn: document.getElementById("loadUploadBtn"),
  uploadFile: document.getElementById("uploadFile"),
  docsCount: document.getElementById("docsCount"),
  vocabSize: document.getElementById("vocabSize"),
  datasetStatus: document.getElementById("datasetStatus"),
  nLayer: document.getElementById("nLayer"),
  nEmbd: document.getElementById("nEmbd"),
  nHead: document.getElementById("nHead"),
  blockSize: document.getElementById("blockSize"),
  steps: document.getElementById("steps"),
  seed: document.getElementById("seed"),
  learningRate: document.getElementById("learningRate"),
  temperature: document.getElementById("temperature"),
  initBtn: document.getElementById("initBtn"),
  trainBtn: document.getElementById("trainBtn"),
  stopBtn: document.getElementById("stopBtn"),
  generateBtn: document.getElementById("generateBtn"),
  promptInput: document.getElementById("promptInput"),
  sampleCount: document.getElementById("sampleCount"),
  progressFill: document.getElementById("progressFill"),
  stepValue: document.getElementById("stepValue"),
  lossValue: document.getElementById("lossValue"),
  stepTimeValue: document.getElementById("stepTimeValue"),
  elapsedValue: document.getElementById("elapsedValue"),
  paramCount: document.getElementById("paramCount"),
  stateValue: document.getElementById("stateValue"),
  liveSample: document.getElementById("liveSample"),
  lossChartLine: document.getElementById("lossChartLine"),
  stepTimeChartLine: document.getElementById("stepTimeChartLine"),
  samplesList: document.getElementById("samplesList"),
  logBox: document.getElementById("logBox"),
  nodeDataset: document.getElementById("node-dataset"),
  nodeTokenizer: document.getElementById("node-tokenizer"),
  nodeConfig: document.getElementById("node-config"),
  nodeTraining: document.getElementById("node-training"),
  nodeGenerate: document.getElementById("node-generate"),
  nodeMetrics: document.getElementById("node-metrics"),
  edgeDatasetTokenizer: document.getElementById("edge-dataset-tokenizer"),
  edgeTokenizerConfig: document.getElementById("edge-tokenizer-config"),
  edgeConfigTraining: document.getElementById("edge-config-training"),
  edgeTrainingMetrics: document.getElementById("edge-training-metrics"),
  edgeTrainingGenerate: document.getElementById("edge-training-generate"),
};

const state = {
  worker: null,
  wasmReady: false,
  datasetReady: false,
  modelReady: false,
  training: false,
  nextRequestID: 1,
  pending: new Map(),
};

function log(line) {
  const stamp = new Date().toLocaleTimeString();
  els.logBox.value += `[${stamp}] ${line}\n`;
  els.logBox.scrollTop = els.logBox.scrollHeight;
}

function setStateText(text) {
  els.stateValue.textContent = text;
}

function setDatasetStatus(text) {
  els.datasetStatus.innerHTML = `<strong>Status:</strong> ${text}`;
}

function setButtons() {
  els.initBtn.disabled = !state.wasmReady || !state.datasetReady || state.training;
  els.trainBtn.disabled = !state.wasmReady || !state.datasetReady || !state.modelReady || state.training;
  els.stopBtn.disabled = !state.training;
  els.generateBtn.disabled = !state.modelReady || state.training;
  updatePipelineDiagram();
}

function setActive(el, active) {
  if (!el) return;
  el.classList.toggle("active", !!active);
}

function updatePipelineDiagram() {
  const dataset = state.datasetReady;
  const tokenizer = state.datasetReady;
  const config = state.datasetReady;
  const training = state.training || state.modelReady;
  const metrics = state.training || state.modelReady;
  const generate = state.modelReady;

  setActive(els.nodeDataset, dataset);
  setActive(els.nodeTokenizer, tokenizer);
  setActive(els.nodeConfig, config);
  setActive(els.nodeTraining, training);
  setActive(els.nodeMetrics, metrics);
  setActive(els.nodeGenerate, generate);

  setActive(els.edgeDatasetTokenizer, dataset && tokenizer);
  setActive(els.edgeTokenizerConfig, tokenizer && config);
  setActive(els.edgeConfigTraining, config && training);
  setActive(els.edgeTrainingMetrics, training && metrics);
  setActive(els.edgeTrainingGenerate, training && generate);
}

function formatMs(ms) {
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function parseIntSafe(value, fallback) {
  const x = Number.parseInt(value, 10);
  return Number.isFinite(x) ? x : fallback;
}

function parseFloatSafe(value, fallback) {
  const x = Number.parseFloat(value);
  return Number.isFinite(x) ? x : fallback;
}

function modelConfigFromUI() {
  return {
    nLayer: parseIntSafe(els.nLayer.value, 1),
    nEmbd: parseIntSafe(els.nEmbd.value, 16),
    nHead: parseIntSafe(els.nHead.value, 4),
    blockSize: parseIntSafe(els.blockSize.value, 16),
  };
}

function trainOptionsFromUI() {
  return {
    steps: parseIntSafe(els.steps.value, 1000),
    learningRate: parseFloatSafe(els.learningRate.value, 0.01),
    temperature: parseFloatSafe(els.temperature.value, 0.5),
    sampleEvery: 25,
    finalSamples: 12,
  };
}

function resetTrainingStats() {
  els.progressFill.style.width = "0%";
  els.stepValue.textContent = "0 / 0";
  els.lossValue.textContent = "-";
  els.stepTimeValue.textContent = "-";
  els.elapsedValue.textContent = "-";
  els.liveSample.textContent = "(live sample will appear here)";
  els.lossChartLine.setAttribute("d", "");
  els.stepTimeChartLine.setAttribute("d", "");
}

function renderSamples(samples) {
  els.samplesList.innerHTML = "";
  for (const sample of samples) {
    const li = document.createElement("li");
    li.textContent = sample || "(empty)";
    els.samplesList.appendChild(li);
  }
}

function renderSeriesPath(pathEl, series) {
  if (!Array.isArray(series) || series.length === 0) {
    pathEl.setAttribute("d", "");
    return;
  }

  if (series.length === 1) {
    const y = 30 - Number(series[0]) * 30;
    pathEl.setAttribute("d", `M 0 ${y} L 100 ${y}`);
    return;
  }

  const points = [];
  for (let i = 0; i < series.length; i++) {
    const x = (i / (series.length - 1)) * 100;
    const y = 30 - Math.max(0, Math.min(1, Number(series[i] || 0))) * 30;
    points.push(`${x.toFixed(2)} ${y.toFixed(2)}`);
  }
  pathEl.setAttribute("d", `M ${points.join(" L ")}`);
}

function handleWorkerMessage(event) {
  const msg = event.data || {};

  if (msg.type === "ready") {
    state.wasmReady = true;
    setStateText(`ready (${msg.version || "unknown"})`);
    setDatasetStatus("select a preset or upload a file");
    setButtons();
    log(`kernel ready (${msg.version || "unknown"}) in worker`);
    return;
  }

  if (msg.type === "fatal") {
    setStateText("worker error");
    log(`worker fatal: ${msg.error}`);
    return;
  }

  if (msg.type === "progress") {
    const pending = state.pending.get(msg.id);
    if (pending && typeof pending.onProgress === "function") {
      pending.onProgress(msg.progress || {});
    }
    return;
  }

  if (msg.type !== "response") return;
  const pending = state.pending.get(msg.id);
  if (!pending) return;
  state.pending.delete(msg.id);

  if (msg.ok) pending.resolve(msg.result);
  else pending.reject(new Error(msg.error || "worker request failed"));
}

function initWorker() {
  state.worker = new Worker("./worker.js");
  state.worker.addEventListener("message", handleWorkerMessage);
  state.worker.addEventListener("error", (event) => {
    setStateText("worker error");
    log(`worker error: ${event.message}`);
  });
}

function callWorker(method, data = {}, onProgress = null) {
  if (!state.worker) return Promise.reject(new Error("worker not initialized"));
  const id = state.nextRequestID++;

  return new Promise((resolve, reject) => {
    state.pending.set(id, { resolve, reject, onProgress });
    state.worker.postMessage({ id, method, data });
  });
}

async function loadDatasetFromText(text) {
  const seed = parseIntSafe(els.seed.value, 42);
  const result = await callWorker("loadDataset", { text, seed });
  if (!result?.ok) {
    throw new Error(result?.error || "loadDataset failed");
  }

  state.datasetReady = true;
  state.modelReady = false;
  els.docsCount.textContent = String(result.numDocs);
  els.vocabSize.textContent = String(result.vocabSize);

  const suggested = Math.min(3000, Math.max(500, result.numDocs * 3));
  const rounded = Math.round(suggested / 50) * 50;
  els.steps.value = String(rounded);

  setDatasetStatus(`loaded ${result.numDocs} docs / vocab ${result.vocabSize}`);
  setStateText("dataset ready");
  resetTrainingStats();
  renderSamples([]);
  els.paramCount.textContent = "-";
  setButtons();

  log(`dataset stats: docs=${result.numDocs}, vocab=${result.vocabSize}, suggested_steps=${rounded}`);
  if (Array.isArray(result.sampleDocs) && result.sampleDocs.length > 0) {
    log(`sample docs: ${result.sampleDocs.slice(0, 3).join(" | ")}`);
  }
}

async function loadPreset() {
  const key = els.datasetSelect.value;
  if (!key || !DATASETS[key]) {
    log("choose a preset first");
    return;
  }
  setDatasetStatus(`loading "${key}"...`);
  const response = await fetch(DATASETS[key]);
  if (!response.ok) {
    throw new Error(`fetch ${DATASETS[key]} failed (${response.status})`);
  }
  const text = await response.text();
  await loadDatasetFromText(text);
  log(`dataset "${key}" loaded`);
}

async function loadUploadedFile(file) {
  const text = await file.text();
  await loadDatasetFromText(text);
  log(`custom dataset loaded: ${file.name}`);
}

async function initModel() {
  if (!state.datasetReady) {
    log("load a dataset first");
    return;
  }
  setStateText("initializing model");
  const result = await callWorker("initModel", { config: modelConfigFromUI() });
  if (!result?.ok) {
    throw new Error(result?.error || "initModel failed");
  }
  state.modelReady = true;
  els.paramCount.textContent = String(result.numParams);
  setStateText("model ready");
  log(`model initialized (${result.numParams} params)`);
  setButtons();
}

async function train() {
  if (!state.modelReady) {
    log("initialize the model first");
    return;
  }

  state.training = true;
  setButtons();
  resetTrainingStats();
  setStateText("training");
  log("training started");

  const opts = trainOptionsFromUI();
  const progress = (m) => {
    const pct = (m.step / m.totalSteps) * 100;
    els.progressFill.style.width = `${Math.min(100, Math.max(0, pct))}%`;
    els.stepValue.textContent = `${m.step} / ${m.totalSteps}`;
    els.lossValue.textContent = Number(m.loss).toFixed(4);
    els.stepTimeValue.textContent = formatMs(Number(m.stepTimeMs || 0));
    els.elapsedValue.textContent = formatMs(Number(m.elapsedMs || 0));
    if (m.sample) {
      els.liveSample.textContent = m.sample;
    }
    if (Array.isArray(m.lossSeries)) {
      renderSeriesPath(els.lossChartLine, m.lossSeries);
    }
    if (Array.isArray(m.stepTimeSeries)) {
      renderSeriesPath(els.stepTimeChartLine, m.stepTimeSeries);
    }
  };

  try {
    const result = await callWorker("train", { options: opts }, progress);
    if (result.stopped) {
      setStateText("stopped");
      log(`training stopped at step ${result.stepsDone}`);
    } else {
      setStateText("trained");
      renderSamples(result.samples || []);
      renderSeriesPath(els.lossChartLine, result.lossSeries || []);
      renderSeriesPath(els.stepTimeChartLine, result.stepTimeSeries || []);
      log(`training completed in ${formatMs(Number(result.totalTimeMs || 0))}`);
    }
  } finally {
    state.training = false;
    setButtons();
  }
}

async function generate() {
  if (!state.modelReady) return;
  setStateText("sampling");
  const temp = parseFloatSafe(els.temperature.value, 0.5);
  const count = parseIntSafe(els.sampleCount.value, 12);
  const prompt = els.promptInput.value || "";
  const out = await callWorker("generate", { options: { count, temperature: temp, prompt } });
  renderSamples(out.samples || []);
  setStateText("trained");
  if (prompt.trim() !== "") {
    log(`generated ${count} samples with prompt="${prompt}"`);
  } else {
    log(`generated ${count} samples`);
  }
}

function stop() {
  callWorker("stop").catch((err) => {
    log(`stop failed: ${err.message}`);
  });
  log("stop requested");
}

function wireUI() {
  els.loadPresetBtn.addEventListener("click", () => {
    loadPreset().catch((err) => {
      log(`dataset load failed: ${err.message}`);
      setDatasetStatus(`error: ${err.message}`);
    });
  });

  els.loadUploadBtn.addEventListener("click", () => {
    els.uploadFile.click();
  });

  els.uploadFile.addEventListener("change", (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    loadUploadedFile(file).catch((err) => {
      log(`dataset load failed: ${err.message}`);
      setDatasetStatus(`error: ${err.message}`);
    });
    event.target.value = "";
  });

  els.initBtn.addEventListener("click", () => {
    initModel().catch((err) => {
      log(`init failed: ${err.message}`);
      setStateText("error");
    });
  });

  els.trainBtn.addEventListener("click", () => {
    train().catch((err) => {
      state.training = false;
      setButtons();
      log(`train failed: ${err.message}`);
      setStateText("error");
    });
  });

  els.stopBtn.addEventListener("click", stop);
  els.generateBtn.addEventListener("click", () => {
    generate().catch((err) => {
      log(`generate failed: ${err.message}`);
      setStateText("error");
    });
  });
}

function main() {
  setButtons();
  wireUI();
  initWorker();
  setStateText("booting worker");
  setDatasetStatus("starting worker...");
  log("starting wasm worker");
}

main();
