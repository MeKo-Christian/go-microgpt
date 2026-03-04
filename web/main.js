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
  progressFill: document.getElementById("progressFill"),
  stepValue: document.getElementById("stepValue"),
  lossValue: document.getElementById("lossValue"),
  stepTimeValue: document.getElementById("stepTimeValue"),
  elapsedValue: document.getElementById("elapsedValue"),
  paramCount: document.getElementById("paramCount"),
  stateValue: document.getElementById("stateValue"),
  liveSample: document.getElementById("liveSample"),
  samplesList: document.getElementById("samplesList"),
  logBox: document.getElementById("logBox"),
};

const state = {
  kernel: null,
  wasmReady: false,
  datasetReady: false,
  modelReady: false,
  training: false,
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
}

function renderSamples(samples) {
  els.samplesList.innerHTML = "";
  for (const sample of samples) {
    const li = document.createElement("li");
    li.textContent = sample || "(empty)";
    els.samplesList.appendChild(li);
  }
}

async function bootWasm() {
  setStateText("booting wasm");
  log("booting wasm runtime");
  const go = new Go();

  let instance;
  try {
    const result = await WebAssembly.instantiateStreaming(fetch("./microgpt.wasm"), go.importObject);
    instance = result.instance;
  } catch (_) {
    const response = await fetch("./microgpt.wasm");
    const bytes = await response.arrayBuffer();
    const result = await WebAssembly.instantiate(bytes, go.importObject);
    instance = result.instance;
  }

  go.run(instance);

  state.kernel = globalThis.MicroGPTKernel;
  if (!state.kernel) {
    throw new Error("MicroGPTKernel not found after wasm startup");
  }

  state.wasmReady = true;
  setStateText(`ready (${state.kernel.version || "unknown"})`);
  log(`kernel ready (${state.kernel.version || "unknown"})`);
  setButtons();
}

async function loadDatasetFromText(text) {
  const seed = parseIntSafe(els.seed.value, 42);
  const result = state.kernel.loadDataset(text, seed);
  if (!result?.ok) {
    throw new Error(result?.error || "loadDataset failed");
  }

  state.datasetReady = true;
  state.modelReady = false;
  els.docsCount.textContent = String(result.numDocs);
  els.vocabSize.textContent = String(result.vocabSize);
  setDatasetStatus(`loaded ${result.numDocs} docs / vocab ${result.vocabSize}`);
  setStateText("dataset ready");
  resetTrainingStats();
  renderSamples([]);
  els.paramCount.textContent = "-";
  setButtons();
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
  const result = state.kernel.initModel(modelConfigFromUI());
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
  };

  try {
    const result = await state.kernel.train(opts, progress);
    if (result.stopped) {
      setStateText("stopped");
      log(`training stopped at step ${result.stepsDone}`);
    } else {
      setStateText("trained");
      renderSamples(result.samples || []);
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
  const out = await state.kernel.generate({ count: 12, temperature: temp });
  renderSamples(out.samples || []);
  setStateText("trained");
  log("generated new samples");
}

function stop() {
  state.kernel.stop();
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

async function main() {
  setButtons();
  wireUI();
  try {
    await bootWasm();
    setDatasetStatus("select a preset or upload a file");
  } catch (err) {
    log(`wasm boot failed: ${err.message}`);
    setStateText("boot error");
    setDatasetStatus("wasm boot failed");
  }
}

main();
