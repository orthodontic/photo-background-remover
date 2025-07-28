import {
  env,
  AutoModel,
  RawImage,
  AutoProcessor,
} from "./lib/transformers.min.js";

env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = "/models/";
env.backends.onnx.wasm.wasmPaths = "/lib/wasm/";

let model, processor;

(async () => {
  model = await AutoModel.from_pretrained("modnet");
  processor = await AutoProcessor.from_pretrained("modnet");
})();

async function handleFile(file) {
  setUIState("loading");
  try {
    const image = await RawImage.read(file);

    const { pixel_values } = await processor(image);
    const { output } = await model({ input: pixel_values });

    const mask = await RawImage.fromTensor(
      output[0].mul(255).to("uint8")
    ).resize(image.width, image.height);

    const imageCanvas = image.toCanvas();
    const imageContext = imageCanvas.getContext("2d");
    const imageData = imageContext.getImageData(
      0,
      0,
      image.width,
      image.height
    );

    const maskCanvas = mask.toCanvas();
    const maskCtx = maskCanvas.getContext("2d");
    const maskData = maskCtx.getImageData(0, 0, image.width, image.height);

    for (let i = 0; i < imageData.data.length; i += 4) {
      imageData.data[i + 3] = maskData.data[i];
    }
    imageContext.putImageData(imageData, 0, 0);

    const outputCanvas = document.getElementById("outputCanvas");
    outputCanvas.width = image.width;
    outputCanvas.height = image.height;
    const outputCtx = outputCanvas.getContext("2d");
    outputCtx.fillStyle = "#3ccef6";
    outputCtx.fillRect(0, 0, image.width, image.height);
    outputCtx.drawImage(imageCanvas, 0, 0);
    console.log(file);

    const downloadLink = document.getElementById("downloadLink");
    downloadLink.download = file.name;
    downloadLink.href = outputCanvas.toDataURL("image/jpeg", 0.92);
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

const fileDrop = document.getElementById("fileDrop");
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

// UI States
function setUIState(state, msg = null) {
  const show = (el) => (el.style.display = "");
  const hide = (el) => (el.style.display = "none");

  const fileDrop = document.getElementById("fileDrop");
  const loader = document.getElementById("loader");
  const error = document.getElementById("error");
  const outputCanvas = document.getElementById("outputCanvas");
  const downloadLink = document.getElementById("downloadLink");

  if (state === "init") {
    show(fileDrop);
    hide(loader);
    hide(error);
    hide(outputCanvas);
    hide(downloadLink);
  }

  if (state === "loading") {
    show(fileDrop);
    show(loader);
    hide(error);
    hide(outputCanvas);
    hide(downloadLink);
  }

  if (state === "failed") {
    show(fileDrop);
    hide(loader);
    show(error);
    error.textContent = msg || "Something went wrong.";
    hide(outputCanvas);
    hide(downloadLink);
  }

  if (state === "done") {
    show(fileDrop);
    hide(loader);
    hide(error);
    show(outputCanvas);
    show(downloadLink);
  }
}
