import { env, AutoModel, RawImage, AutoProcessor } from "./transformers.min.js";

env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = "./models/";
env.backends.onnx.wasm.wasmPaths = "./wasm/";

let model, processor;

async function loadModnet() {
  if (model && processor) return;
  model = await AutoModel.from_pretrained("modnet");
  processor = await AutoProcessor.from_pretrained("modnet");
}

window.addEventListener("DOMContentLoaded", () => {
  setUIState("init");
  loadModnet();
});

window.addEventListener("beforeunload", () => {
  model?.dispose?.();
  processor?.dispose?.();
});

let filename;
async function handleFile(file) {
  setUIState("loading");
  try {
    const image = await RawImage.read(file);
    const { pixel_values } = await processor(image);
    const { output } = await model({ input: pixel_values });

    const mask = await RawImage.fromTensor(
      output[0].mul(255).to("uint8")
    ).resize(image.width, image.height);

    image.putAlpha(mask);

    const imageCanvas = document.getElementById("imageCanvas");
    imageCanvas.width = image.width;
    imageCanvas.height = image.height;
    imageCanvas.getContext("2d").drawImage(image.toCanvas(), 0, 0);
    const maskCanvas = document.getElementById("maskCanvas");
    maskCanvas.width = image.width;
    maskCanvas.height = image.height;
    maskCanvas.getContext("2d").drawImage(mask.toCanvas(), 0, 0);

    filename = file.name.replace(/\.[^/.]+$/, "");

    renderImageWithBackground(document.getElementById("color_input").value);

    setUIState("done");
  } catch (e) {
    setUIState("failed", e);
  }
}

// Setup file input and drop
const fileInput = document.getElementById("fileInput");
fileInput.addEventListener("change", (e) => {
  handleFile(e.target.files[0]);
});

const fileDrop = document.getElementById("image");
fileDrop.addEventListener("click", () => fileInput.click());
fileDrop.addEventListener("dragover", (e) => {
  e.preventDefault();
  fileDrop.classList.add("dragover");
});

fileDrop.addEventListener("dragleave", () => {
  fileDrop.classList.remove("dragover");
});
fileDrop.addEventListener("drop", (e) => {
  e.preventDefault();
  fileDrop.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (file) {
    handleFile(file);
  }
});

// Setup rendering tools
function renderImageWithBackground(background) {
  const imageCanvas = document.getElementById("imageCanvas");
  var tmpCanvas = document.createElement("canvas");
  tmpCanvas.width = imageCanvas.width;
  tmpCanvas.height = imageCanvas.height;
  const tmpCtx = tmpCanvas.getContext("2d");
  tmpCtx.fillStyle = background;
  tmpCtx.fillRect(0, 0, imageCanvas.width, imageCanvas.height);
  tmpCtx.drawImage(imageCanvas, 0, 0);

  const jpegSrc = tmpCanvas.toDataURL("image/jpeg", 0.92);

  document.getElementById("image").src = jpegSrc;

  const downloadLink = document.getElementById("downloadLink");
  downloadLink.download = `${filename}_${background}.jpg`; // filename is global
  downloadLink.href = jpegSrc;
}
function renderImageTransparent() {
  const imageCanvas = document.getElementById("imageCanvas");
  const pngSrc = imageCanvas.toDataURL("image/png");
  document.getElementById("image").src = pngSrc;

  const downloadLink = document.getElementById("downloadLink");
  downloadLink.download = `${filename}_transparent.png`; // filename is global
  downloadLink.href = pngSrc;
}
function renderMask() {
  const maskCanvas = document.getElementById("maskCanvas");
  const pngSrc = maskCanvas.toDataURL("image/png");
  document.getElementById("image").src = pngSrc;

  const downloadLink = document.getElementById("downloadLink");
  downloadLink.download = `${filename}_mask.png`; // filename is global
  downloadLink.href = pngSrc;
}

const colorInput = document.getElementById("color_input");
colorInput.addEventListener("change", colorChange);
colorInput.addEventListener("click", colorChange);
function colorChange(e) {
  document.getElementById("color_selected").style.backgroundColor =
    e.target.value;
  renderImageWithBackground(e.target.value);
}

const transparentBtn = document.getElementById("transparent");
transparentBtn.addEventListener("click", renderImageTransparent);

const maskBtn = document.getElementById("mask");
maskBtn.addEventListener("click", renderMask);

// UI States
function setUIState(state, msg = null) {
  const show = (el) => (el.style.display = "");
  const hide = (el) => (el.style.display = "none");

  const status = document.getElementById("status");
  const tools = document.getElementById("tools");

  if (state === "init") {
    hide(status);
    hide(tools);
  }

  if (state === "loading") {
    status.textContent = "Loading...";
    show(status);
    hide(tools);
  }

  if (state === "failed") {
    status.textContent = msg || "Something went wrong.";
    show(status);
    hide(tools);
  }

  if (state === "done") {
    hide(status);
    show(tools);
  }
}
