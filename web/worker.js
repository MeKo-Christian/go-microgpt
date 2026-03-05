let kernel = null;
let bootPromise = null;
const assetVersion = Date.now().toString(36);
const wasmExecURL = `./wasm_exec.js?v=${assetVersion}`;
const wasmURL = `./microgpt.wasm?v=${assetVersion}`;

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function bootKernel() {
  if (bootPromise) return bootPromise;

  bootPromise = (async () => {
    importScripts(wasmExecURL);

    const go = new Go();

    let instance;
    try {
      const response = await fetch(wasmURL, { cache: "no-store" });
      const result = await WebAssembly.instantiateStreaming(response, go.importObject);
      instance = result.instance;
    } catch (_) {
      const response = await fetch(wasmURL, { cache: "no-store" });
      const bytes = await response.arrayBuffer();
      const result = await WebAssembly.instantiate(bytes, go.importObject);
      instance = result.instance;
    }

    go.run(instance);

    for (let i = 0; i < 200; i++) {
      if (self.MicroGPTKernel) break;
      await sleep(10);
    }

    if (!self.MicroGPTKernel) {
      throw new Error("MicroGPTKernel not found after wasm startup");
    }

    kernel = self.MicroGPTKernel;
    self.postMessage({
      type: "ready",
      version: kernel.version || "unknown",
    });
  })();

  return bootPromise;
}

function fail(id, error) {
  self.postMessage({
    type: "response",
    id,
    ok: false,
    error: error instanceof Error ? error.message : String(error),
  });
}

function ok(id, result) {
  self.postMessage({
    type: "response",
    id,
    ok: true,
    result,
  });
}

self.onmessage = async (event) => {
  const { id, method, data } = event.data || {};
  if (!method) return;

  try {
    await bootKernel();
  } catch (error) {
    if (id) fail(id, error);
    else {
      self.postMessage({
        type: "fatal",
        error: error instanceof Error ? error.message : String(error),
      });
    }
    return;
  }

  try {
    if (method === "loadDataset") {
      const result = kernel.loadDataset(data?.text || "", data?.seed || 42);
      ok(id, result);
      return;
    }

    if (method === "initModel") {
      const result = kernel.initModel(data?.config || {});
      ok(id, result);
      return;
    }

    if (method === "train") {
      const result = await kernel.train(data?.options || {}, (progress) => {
        self.postMessage({
          type: "progress",
          id,
          progress,
        });
      });
      ok(id, result);
      return;
    }

    if (method === "generate") {
      const result = await kernel.generate(data?.options || {});
      ok(id, result);
      return;
    }

    if (method === "stop") {
      const result = kernel.stop();
      ok(id, result);
      return;
    }

    fail(id, `unknown method: ${method}`);
  } catch (error) {
    fail(id, error);
  }
};

bootKernel().catch((error) => {
  self.postMessage({
    type: "fatal",
    error: error instanceof Error ? error.message : String(error),
  });
});
