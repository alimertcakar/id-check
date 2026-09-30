/*
 * ID name check: reads the ID image with Tesseract.js, in the browser (the
 * image is never uploaded), and checks that the typed name appears on it.
 */
(() => {
  'use strict';

  const DEFAULT_IMAGE = 'id.png';
  // Tesseract language data. 'eng+tur' reads Turkish letters (Ç, Ş, İ, ...) better.
  const OCR_LANGS = 'eng';

  const form = document.getElementById('check-form');
  const fields = document.getElementById('fields');
  const nameInput = document.getElementById('name');
  const imageInput = document.getElementById('image');
  const imageNote = document.getElementById('image-note');
  const preview = document.getElementById('preview');
  const submitButton = document.getElementById('submit');
  const statusEl = document.getElementById('status');
  const resultEl = document.getElementById('result');
  const ocrOutput = document.getElementById('ocr-output');
  const ocrText = document.getElementById('ocr-text');

  let previewUrl = null;
  let workerPromise = null;
  let onProgress = () => {};

  nameInput.addEventListener('input', () => {
    nameInput.setCustomValidity('');
    clearResult();
  });
  imageInput.addEventListener('change', () => {
    showPreview(imageInput.files[0]);
    clearResult();
  });
  form.addEventListener('submit', onSubmit);

  loadDefaultImage();
  getWorker().catch(() => {}); // start loading the OCR engine while the user types

  // ------------------------------------------------------------------ form

  async function loadDefaultImage() {
    try {
      const res = await fetch(DEFAULT_IMAGE);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const blob = await res.blob();
      if (imageInput.files.length) return; // the user already picked a file
      const files = new DataTransfer();
      files.items.add(new File([blob], DEFAULT_IMAGE, { type: blob.type || 'image/png' }));
      imageInput.files = files.files;
      showPreview(imageInput.files[0]);
    } catch {
      // Pages opened straight from disk (file://) may not read other files.
      imageNote.textContent = `Couldn't load ${DEFAULT_IMAGE} as the default image. ` +
        'Open this page through a web server to use it, or choose an image.';
      imageNote.hidden = false;
    }
  }

  function showPreview(file) {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = file ? URL.createObjectURL(file) : null;
    preview.hidden = !file;
    if (file) {
      preview.src = previewUrl;
      imageNote.hidden = true;
    } else {
      preview.removeAttribute('src');
    }
  }

  async function onSubmit(event) {
    event.preventDefault();
    const name = nameInput.value;
    if (!words(name).length) {
      nameInput.setCustomValidity('Please enter a name.');
      nameInput.reportValidity();
      return;
    }
    clearResult();
    setBusy(true);
    try {
      const { match, text } = await readCard(imageInput.files[0], name);
      showResult(match, text);
    } catch (err) {
      console.error(err);
      showError(err);
    } finally {
      setBusy(false);
    }
  }

  function setBusy(busy) {
    fields.disabled = busy;
    submitButton.textContent = busy ? 'Checking…' : 'Check name';
    if (!busy) setStatus('');
  }

  function setStatus(message) {
    statusEl.textContent = message;
  }

  function clearResult() {
    resultEl.hidden = true;
    ocrOutput.hidden = true;
  }

  function showResult(match, text) {
    const icon = document.createElement('span');
    icon.setAttribute('aria-hidden', 'true');
    icon.textContent = match ? '✓ ' : '✗ ';
    resultEl.replaceChildren(icon, match ? 'Name matches' : 'Name did not match the ID card');
    resultEl.className = `result ${match ? 'ok' : 'fail'}`;
    resultEl.hidden = false;
    ocrText.textContent = text || '(no text found)';
    ocrOutput.hidden = false;
  }

  function showError(err) {
    resultEl.textContent = `Couldn't check the image: ${err && err.message ? err.message : err}`;
    resultEl.className = 'result error';
    resultEl.hidden = false;
  }

  // ------------------------------------------------------------------- OCR

  // Reads the cleaned-up versions of the image in turn and stops as soon as
  // the name is found. Without a match, returns the most confident reading.
  async function readCard(file, name) {
    setStatus('Preparing the image…');
    const versions = await cleanedVersions(file);
    setStatus('Loading the OCR engine…');
    const worker = await getWorker();
    let best = null;
    for (let i = 0; i < versions.length; i++) {
      const label = i === 0 ? 'Reading the ID card' : 'Taking a second look';
      setStatus(`${label}…`);
      onProgress = ({ status, progress }) => {
        if (status === 'recognizing text') setStatus(`${label}… ${Math.round(progress * 100)}%`);
      };
      const { data } = await worker.recognize(versions[i], { rotateAuto: true });
      const reading = { text: data.text.trim(), confidence: data.confidence, match: nameMatches(name, data.text) };
      if (reading.match) return reading;
      if (!best || reading.confidence > best.confidence) best = reading;
    }
    return best;
  }

  function getWorker() {
    if (!workerPromise) {
      workerPromise = typeof Tesseract === 'undefined'
        ? Promise.reject(new Error('the OCR library (Tesseract.js) did not load. Check your internet connection and reload the page.'))
        : Tesseract.createWorker(OCR_LANGS, 1, { logger: (m) => onProgress(m) });
      workerPromise.catch(() => { workerPromise = null; }); // allow a retry
    }
    return workerPromise;
  }

  // --------------------------------------------------------- preprocessing
  // Tesseract reads dark, straight text on a plain light background best. ID
  // photos are often small, tilted, unevenly lit or light-on-dark (like
  // id.png), so it gets black-and-white versions made with a local (Sauvola)
  // threshold: one keeping dark text and one keeping light text, the more
  // likely first. rotateAuto then lets Tesseract straighten the text.

  async function cleanedVersions(file) {
    const bitmap = await createImageBitmap(file);
    const image = grayscale(bitmap);
    bitmap.close();
    const integral = integralImages(image);
    const lightFirst = medianGray(image) < 100; // mostly dark picture: expect light text
    return [lightFirst, !lightFirst].map((light) => toCanvas(image, threshold(image, integral, light)));
  }

  // Grayscale copy scaled so the long side is 1000-1600 px: small text needs
  // more pixels for Tesseract, and huge photos are only slower.
  function grayscale(bitmap) {
    const long = Math.max(bitmap.width, bitmap.height);
    const scale = Math.min(Math.max(long, 1000), 1600) / long;
    const w = Math.round(bitmap.width * scale);
    const h = Math.round(bitmap.height * scale);
    const ctx = canvas2d(w, h);
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, w, h);
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(bitmap, 0, 0, w, h);
    const rgba = ctx.getImageData(0, 0, w, h).data;
    const gray = new Uint8ClampedArray(w * h);
    for (let i = 0; i < gray.length; i++) {
      gray[i] = 0.299 * rgba[i * 4] + 0.587 * rgba[i * 4 + 1] + 0.114 * rgba[i * 4 + 2];
    }
    return { w, h, gray };
  }

  function medianGray({ gray }) {
    const hist = new Uint32Array(256);
    for (const v of gray) hist[v]++;
    let seen = 0;
    for (let v = 0; v < 256; v++) {
      seen += hist[v];
      if (seen * 2 >= gray.length) return v;
    }
    return 255;
  }

  // Summed-area tables of the pixels and their squares: the mean and standard
  // deviation of any window then cost four lookups each.
  function integralImages({ w, h, gray }) {
    const W = w + 1;
    const sum = new Float64Array(W * (h + 1));
    const sq = new Float64Array(W * (h + 1));
    for (let y = 0; y < h; y++) {
      let rowSum = 0, rowSq = 0;
      for (let x = 0; x < w; x++) {
        const v = gray[y * w + x];
        rowSum += v;
        rowSq += v * v;
        sum[(y + 1) * W + x + 1] = sum[y * W + x + 1] + rowSum;
        sq[(y + 1) * W + x + 1] = sq[y * W + x + 1] + rowSq;
      }
    }
    return { sum, sq };
  }

  // Sauvola threshold: a pixel is ink when it is clearly darker (or, for light
  // text, lighter) than its surroundings.
  function threshold({ w, h, gray }, { sum, sq }, lightText) {
    const k = 0.3, minContrast = 12;
    const r = Math.round(Math.max(w, h) / 30);
    const W = w + 1;
    const ink = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) {
      const y0 = Math.max(0, y - r), y1 = Math.min(h, y + r + 1);
      for (let x = 0; x < w; x++) {
        const x0 = Math.max(0, x - r), x1 = Math.min(w, x + r + 1);
        const n = (x1 - x0) * (y1 - y0);
        const s = sum[y1 * W + x1] - sum[y0 * W + x1] - sum[y1 * W + x0] + sum[y0 * W + x0];
        const q = sq[y1 * W + x1] - sq[y0 * W + x1] - sq[y1 * W + x0] + sq[y0 * W + x0];
        const mean = s / n;
        const sd = Math.sqrt(Math.max(0, q / n - mean * mean));
        // flip light text to dark so one formula covers both cases
        const m = lightText ? 255 - mean : mean;
        const v = lightText ? 255 - gray[y * w + x] : gray[y * w + x];
        if (v < m * (1 + k * (sd / 128 - 1)) && m - v > minContrast) ink[y * w + x] = 1;
      }
    }
    return ink;
  }

  // Black ink on white, as a canvas Tesseract can read.
  function toCanvas({ w, h }, ink) {
    const ctx = canvas2d(w, h);
    const out = ctx.createImageData(w, h);
    for (let i = 0; i < ink.length; i++) {
      const v = ink[i] ? 0 : 255;
      out.data[i * 4] = out.data[i * 4 + 1] = out.data[i * 4 + 2] = v;
      out.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(out, 0, 0);
    return ctx.canvas;
  }

  function canvas2d(w, h) {
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    return canvas.getContext('2d', { willReadFrequently: true });
  }

  // -------------------------------------------------------------- matching
  // Every word of the typed name must appear in the text read from the card,
  // in any order (cards often print the surname first or on its own line).
  // Case, accents and common OCR look-alikes are ignored, and longer words may
  // be off by a letter or two because OCR still slips.

  const LETTER_MAP = {
    'ß': 'SS', 'Æ': 'AE', 'æ': 'ae', 'Œ': 'OE', 'œ': 'oe', 'Ø': 'O', 'ø': 'o',
    'Đ': 'D', 'đ': 'd', 'Ł': 'L', 'ł': 'l', 'Þ': 'TH', 'þ': 'th', 'ı': 'i',
  };
  const LOOKALIKES = { '0': 'O', '1': 'I', '|': 'I', '!': 'I', '2': 'Z', '5': 'S', '$': 'S', '6': 'G', '8': 'B' };

  function nameMatches(name, text) {
    const wanted = words(name);
    const candidates = [];
    for (const line of text.split('\n')) {
      const found = words(line);
      found.forEach((word, i) => {
        candidates.push(word);
        if (i + 1 < found.length) candidates.push(word + found[i + 1]); // a word OCR split in two
      });
    }
    const onCard = (word) => candidates.some((c) => editDistance(word, c) <= allowedTypos(word));
    // also accept the whole name run together, in case OCR dropped the spaces
    return wanted.every(onCard) || (wanted.length > 1 && onCard(wanted.join('')));
  }

  // Up to 4 letters must match exactly, 5-7 may have one wrong letter, 8+ two.
  function allowedTypos(word) {
    return word.length >= 8 ? 2 : word.length >= 5 ? 1 : 0;
  }

  function words(text) {
    return text.split(/\s+/).map(normalizeWord).filter(Boolean);
  }

  function normalizeWord(word) {
    let w = word.replace(/[ßÆæŒœØøĐđŁłÞþı]/g, (c) => LETTER_MAP[c])
      .normalize('NFKD')
      .replace(/\p{M}/gu, '');
    // OCR often reads a capital I as a lowercase l inside upper-case words
    if ((w.match(/[A-Z]/g) || []).length > (w.match(/[a-z]/g) || []).length) w = w.replace(/l/g, 'I');
    return w.toUpperCase().replace(/[^A-Z]/g, (c) => LOOKALIKES[c] || '');
  }

  function editDistance(a, b) {
    let prev = Array.from({ length: b.length + 1 }, (_, j) => j);
    for (let i = 1; i <= a.length; i++) {
      const cur = [i];
      for (let j = 1; j <= b.length; j++) {
        cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
      }
      prev = cur;
    }
    return prev[b.length];
  }
})();
