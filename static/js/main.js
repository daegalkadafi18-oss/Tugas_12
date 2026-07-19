document.addEventListener("DOMContentLoaded", () => {
  const dropzone   = document.getElementById("dropzone");
  const fileInput  = document.getElementById("file-input");
  const previewWrap= document.getElementById("preview-wrap");
  const previewImg = document.getElementById("preview-img");
  const dzIcon     = document.getElementById("dz-icon");
  const dzHint     = document.getElementById("dz-hint");
  const analyzeBtn = document.getElementById("analyze-btn");
  const resetBtn   = document.getElementById("reset-btn");
  const statusText = document.getElementById("status-text");
  const form       = document.getElementById("upload-form");
  const errorBox   = document.getElementById("error-box");
  const resultPanel= document.getElementById("result-panel");

  if (!dropzone) return; // halaman lain (about.html) tidak punya elemen ini

  let selectedFile = null;

  function showPreview(file){
    selectedFile = file;
    const url = URL.createObjectURL(file);
    previewImg.src = url;
    previewWrap.classList.add("active");
    dzIcon.style.display = "none";
    dzHint.style.display = "none";
    analyzeBtn.disabled = false;
    statusText.textContent = `Siap dianalisis: ${file.name}`;
    resultPanel.classList.remove("active");
    errorBox.classList.remove("active");
  }

  function resetDropzone(){
    selectedFile = null;
    fileInput.value = "";
    previewWrap.classList.remove("active", "scanning");
    dzIcon.style.display = "";
    dzHint.style.display = "";
    analyzeBtn.disabled = true;
    statusText.textContent = "Belum ada gambar dipilih";
    resultPanel.classList.remove("active");
    errorBox.classList.remove("active");
  }

  dropzone.addEventListener("click", (e) => {
    if (e.target === resetBtn) return;
    fileInput.click();
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files[0]) showPreview(fileInput.files[0]);
  });

  ["dragenter", "dragover"].forEach(evt => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach(evt => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    });
  });
  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) showPreview(file);
  });

  resetBtn.addEventListener("click", (e) => {
    e.preventDefault();
    resetDropzone();
  });

  function renderResults(data){
    const label = document.getElementById("res-label");
    const conf  = document.getElementById("res-confidence");
    const info  = document.getElementById("res-info");
    const list  = document.getElementById("prob-list");

    label.textContent = data.prediction;
    conf.textContent = `${(data.confidence * 100).toFixed(1)}% yakin`;
    info.textContent = data.info || "";

    list.innerHTML = "";
    data.all_probabilities.forEach((row, i) => {
      const el = document.createElement("div");
      el.className = "prob-row" + (i === 0 ? " top" : "");
      el.innerHTML = `
        <div class="name">${row.label}</div>
        <div class="grain-meter"><div class="fill" data-pct="${(row.probability*100).toFixed(1)}"></div></div>
        <div class="value">${(row.probability*100).toFixed(1)}%</div>
      `;
      list.appendChild(el);
    });

    resultPanel.classList.add("active");

    // animate meters after insertion
    requestAnimationFrame(() => {
      document.querySelectorAll(".grain-meter .fill").forEach(fill => {
        fill.style.width = fill.dataset.pct + "%";
      });
    });
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!selectedFile) return;

    errorBox.classList.remove("active");
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = "Menganalisis...";
    previewWrap.classList.add("scanning");
    statusText.textContent = "Model sedang memindai tekstur & bentuk butir beras...";

    const formData = new FormData();
    formData.append("image", selectedFile);

    try{
      const res = await fetch("/predict", { method: "POST", body: formData });
      const data = await res.json();

      previewWrap.classList.remove("scanning");

      if (!res.ok || data.error){
        errorBox.textContent = data.error || "Terjadi kesalahan saat memproses gambar.";
        errorBox.classList.add("active");
        statusText.textContent = "Analisis gagal.";
      } else {
        renderResults(data);
        statusText.textContent = "Analisis selesai.";
      }
    } catch(err){
      previewWrap.classList.remove("scanning");
      errorBox.textContent = "Tidak dapat terhubung ke server. Pastikan 'python app.py' sedang berjalan.";
      errorBox.classList.add("active");
      statusText.textContent = "Analisis gagal.";
    } finally {
      analyzeBtn.disabled = false;
      analyzeBtn.textContent = "Analisis Beras";
    }
  });
});
