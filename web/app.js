"use strict";

const MODEL_URL = "model/best.onnx";
const SAMPLE_VIDEO_URL = "sample/infrared_test.mp4";
const INPUT_SIZE = 640;
const CONF_THRESH = 0.25;
const LETTERBOX_COLOR = 114; 

const videoEl = document.getElementById("video");
const canvasEl = document.getElementById("overlay");
const ctx = canvasEl.getContext("2d");
const stageEl = document.getElementById("stage");
const startBtn = document.getElementById("start");
const modeFileBtn = document.getElementById("mode-file");
const modeCameraBtn = document.getElementById("mode-camera");
const backendLabel = document.getElementById("backend");
const fpsLabel = document.getElementById("fps");

let mode = "file";
let session = null;
let backendUsed = "-";
let busy = false;
let lastFrameTime = performance.now();
let fpsEma = null;
let started = false;

const preCanvas = document.createElement("canvas");
preCanvas.width = INPUT_SIZE;
preCanvas.height = INPUT_SIZE;
const preCtx = preCanvas.getContext("2d", { willReadFrequently: true });

function setMode(newMode) {
  if (started) return; 
  mode = newMode;
  modeFileBtn.classList.toggle("active", mode === "file");
  modeCameraBtn.classList.toggle("active", mode === "camera");
}

modeFileBtn.addEventListener("click", () => setMode("file"));
modeCameraBtn.addEventListener("click", () => setMode("camera"));

videoEl.addEventListener("loadedmetadata", () => {
  canvasEl.width = videoEl.videoWidth;
  canvasEl.height = videoEl.videoHeight;
  stageEl.style.aspectRatio = `${videoEl.videoWidth} / ${videoEl.videoHeight}`;
});

async function loadSession() {
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1/dist/";
  try {
    session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ["webgpu"] });
    backendUsed = "webgpu";
  } catch (err) {
    console.warn("webgpu unavailable, falling back to wasm:", err);
    session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ["wasm"] });
    backendUsed = "wasm";
  }
  backendLabel.textContent = `backend: ${backendUsed}`;
}

async function startVideo() {
  if (mode === "camera") {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" },
      audio: false,
    });
    videoEl.srcObject = stream;
  } else {
    videoEl.src = SAMPLE_VIDEO_URL;
    videoEl.loop = true;
  }
  await videoEl.play();
}

function preprocess() {
  const vw = videoEl.videoWidth;
  const vh = videoEl.videoHeight;
  const scale = Math.min(INPUT_SIZE / vw, INPUT_SIZE / vh);
  const newW = Math.round(vw * scale);
  const newH = Math.round(vh * scale);
  const padX = (INPUT_SIZE - newW) / 2;
  const padY = (INPUT_SIZE - newH) / 2;

  preCtx.fillStyle = `rgb(${LETTERBOX_COLOR},${LETTERBOX_COLOR},${LETTERBOX_COLOR})`;
  preCtx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
  preCtx.drawImage(videoEl, 0, 0, vw, vh, padX, padY, newW, newH);

  const { data } = preCtx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
  const plane = INPUT_SIZE * INPUT_SIZE;
  const tensorData = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    const o = i * 4;
    tensorData[i] = data[o] / 255; 
    tensorData[plane + i] = data[o + 1] / 255; 
    tensorData[2 * plane + i] = data[o + 2] / 255; 
  }

  return { tensorData, scale, padX, padY };
}

function drawDetections(output, scale, padX, padY) {
  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

  const dims = output.dims; 
  const numDet = dims[1];
  const stride = dims[2];
  const data = output.data;

  ctx.lineWidth = Math.max(2, canvasEl.width / 320);
  ctx.strokeStyle = "#ff3b30";
  ctx.fillStyle = "#ff3b30";
  ctx.font = `${Math.max(14, canvasEl.width / 40)}px sans-serif`;

  for (let i = 0; i < numDet; i++) {
    const base = i * stride;
    const conf = data[base + 4];
    if (conf < CONF_THRESH) continue;

    const rx1 = (data[base] - padX) / scale;
    const ry1 = (data[base + 1] - padY) / scale;
    const rx2 = (data[base + 2] - padX) / scale;
    const ry2 = (data[base + 3] - padY) / scale;

    ctx.strokeRect(rx1, ry1, rx2 - rx1, ry2 - ry1);
    ctx.fillText(`UAV ${conf.toFixed(2)}`, rx1, Math.max(12, ry1 - 4));
  }
}

function updateFps() {
  const now = performance.now();
  const dt = now - lastFrameTime;
  lastFrameTime = now;
  const instFps = 1000 / dt;
  fpsEma = fpsEma === null ? instFps : fpsEma * 0.9 + instFps * 0.1;
  fpsLabel.textContent = `FPS: ${fpsEma.toFixed(1)}`;
}

async function processFrame() {
  const { tensorData, scale, padX, padY } = preprocess();
  const inputTensor = new ort.Tensor("float32", tensorData, [1, 3, INPUT_SIZE, INPUT_SIZE]);
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];
  const results = await session.run({ [inputName]: inputTensor });
  drawDetections(results[outputName], scale, padX, padY);
  updateFps();
}

function loop() {
  requestAnimationFrame(loop);
  if (busy || !session || videoEl.readyState < 2 || videoEl.paused || videoEl.ended) return;
  busy = true;
  processFrame()
    .catch((err) => console.error("inference error:", err))
    .finally(() => {
      busy = false;
    });
}

startBtn.addEventListener("click", async () => {
  startBtn.disabled = true;
  modeFileBtn.disabled = true;
  modeCameraBtn.disabled = true;
  startBtn.textContent = "Завантаження моделі…";
  try {
    if (!session) await loadSession();
    started = true;
    await startVideo();
    startBtn.textContent = "Йде трекінг";
    requestAnimationFrame(loop);
  } catch (err) {
    console.error(err);
    startBtn.textContent = "Помилка — див. консоль";
    startBtn.disabled = false;
    modeFileBtn.disabled = false;
    modeCameraBtn.disabled = false;
    started = false;
  }
});
