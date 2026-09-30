/*
 * ocr.js: dependency-free check of whether a name is printed on an image.
 *
 * This is not a general-purpose OCR engine. The typed name is known, so the
 * job is to verify it, not to recognise arbitrary text:
 *
 *   1. grayscale + rescale the image
 *   2. Sauvola binarisation (light-on-dark and dark-on-light, two window sizes)
 *   3. connected components -> glyph candidates
 *   4. chain glyphs into text lines (rotation tolerant) and deskew each line
 *   5. score every glyph against letter templates rendered from system fonts
 *   6. look for each word of the name as a run of glyphs whose shapes agree
 *      with the expected letters
 *
 * Exposes window.NameOCR.findName(imageSource, name).
 */
(function (global) {
  'use strict';

  const GRID = 16;                 // glyph feature resolution (GRID x GRID)
  const SUB = 3;                   // supersamples per feature cell, per axis
  const CLASSES = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
  const FONTS = [
    'Arial', 'Helvetica', 'Verdana', 'Tahoma', '"Trebuchet MS"', '"Segoe UI"',
    'Calibri', 'Bahnschrift', '"Century Gothic"', 'Futura', 'Avenir',
    '"Gill Sans"', '"DejaVu Sans"', '"Liberation Sans"', 'Roboto',
    '"Times New Roman"', 'Georgia', '"Courier New"',
    'sans-serif', 'serif', 'monospace',
  ];

  // ------------------------------------------------------------------ image

  function toGray(source) {
    const long = Math.max(source.width, source.height);
    const target = Math.min(Math.max(long, 1000), 1600);
    const scale = target / long;
    const w = Math.max(1, Math.round(source.width * scale));
    const h = Math.max(1, Math.round(source.height * scale));
    const ctx = makeCanvas(w, h);
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, w, h);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(source, 0, 0, w, h);
    const rgba = ctx.getImageData(0, 0, w, h).data;
    const gray = new Uint8Array(w * h);
    for (let i = 0, j = 0; i < gray.length; i++, j += 4) {
      gray[i] = (rgba[j] * 299 + rgba[j + 1] * 587 + rgba[j + 2] * 114) / 1000;
    }
    return { w, h, gray, scale };
  }

  function makeCanvas(w, h) {
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    return canvas.getContext('2d', { willReadFrequently: true });
  }

  function integrals(img) {
    const { w, h, gray } = img;
    const W = w + 1;
    const sum = new Float64Array(W * (h + 1));
    const sq = new Float64Array(W * (h + 1));
    for (let y = 0; y < h; y++) {
      let rs = 0, rq = 0;
      for (let x = 0; x < w; x++) {
        const v = gray[y * w + x];
        rs += v;
        rq += v * v;
        sum[(y + 1) * W + x + 1] = sum[y * W + x + 1] + rs;
        sq[(y + 1) * W + x + 1] = sq[y * W + x + 1] + rq;
      }
    }
    return { sum, sq };
  }

  // Sauvola thresholding. lightText selects the polarity we are looking for.
  function binarize(img, integ, win, lightText, k, minContrast) {
    const { w, h, gray } = img;
    const { sum, sq } = integ;
    const W = w + 1, r = win >> 1;
    const mask = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) {
      const y0 = Math.max(0, y - r), y1 = Math.min(h, y + r + 1);
      for (let x = 0; x < w; x++) {
        const x0 = Math.max(0, x - r), x1 = Math.min(w, x + r + 1);
        const n = (x1 - x0) * (y1 - y0);
        const s = sum[y1 * W + x1] - sum[y0 * W + x1] - sum[y1 * W + x0] + sum[y0 * W + x0];
        const q = sq[y1 * W + x1] - sq[y0 * W + x1] - sq[y1 * W + x0] + sq[y0 * W + x0];
        const m = s / n;
        const sd = Math.sqrt(Math.max(0, q / n - m * m));
        const v = gray[y * w + x];
        // work in "ink is dark" space so one formula covers both polarities
        const mi = lightText ? 255 - m : m;
        const vi = lightText ? 255 - v : v;
        if (vi < mi * (1 + k * (sd / 128 - 1)) && mi - vi > minContrast) mask[y * w + x] = 1;
      }
    }
    return mask;
  }

  // 8-connected components via flood fill.
  function components(mask, w, h) {
    const labels = new Int32Array(w * h);
    const stack = new Int32Array(w * h);
    const comps = [null];
    let next = 1;
    for (let i = 0; i < mask.length; i++) {
      if (!mask[i] || labels[i]) continue;
      let sp = 0;
      stack[sp++] = i;
      labels[i] = next;
      let minX = w, minY = h, maxX = -1, maxY = -1, area = 0;
      while (sp) {
        const p = stack[--sp];
        const px = p % w, py = (p - px) / w;
        area++;
        if (px < minX) minX = px;
        if (px > maxX) maxX = px;
        if (py < minY) minY = py;
        if (py > maxY) maxY = py;
        for (let dy = -1; dy <= 1; dy++) {
          const qy = py + dy;
          if (qy < 0 || qy >= h) continue;
          for (let dx = -1; dx <= 1; dx++) {
            const qx = px + dx;
            if (qx < 0 || qx >= w) continue;
            const q = qy * w + qx;
            if (mask[q] && !labels[q]) {
              labels[q] = next;
              stack[sp++] = q;
            }
          }
        }
      }
      comps.push({ id: next, minX, minY, maxX, maxY, area, w: maxX - minX + 1, h: maxY - minY + 1 });
      next++;
    }
    return { labels, comps };
  }

  // ------------------------------------------------------------- features

  // Bilinear "is this pixel part of the glyph" sampler over a label image.
  function makeSampler(labels, w, h, ids) {
    const set = new Set(ids);
    const at = (x, y) => (x >= 0 && y >= 0 && x < w && y < h && set.has(labels[y * w + x]) ? 1 : 0);
    return (x, y) => {
      const x0 = Math.floor(x), y0 = Math.floor(y);
      const fx = x - x0, fy = y - y0;
      return (at(x0, y0) * (1 - fx) + at(x0 + 1, y0) * fx) * (1 - fy) +
             (at(x0, y0 + 1) * (1 - fx) + at(x0 + 1, y0 + 1) * fx) * fy;
    };
  }

  // Resample a glyph box (in the line's rotated u/v frame) onto a GRID x GRID
  // grid and return it zero-mean / unit-length, ready for dot products.
  function featureVector(sample, frame, box) {
    let u0 = box.u0 - 0.5, u1 = box.u1 + 0.5, v0 = box.v0 - 0.5, v1 = box.v1 + 0.5;
    let bw = u1 - u0, bh = v1 - v0;
    // keep thin glyphs (I, l, 1) thin and flat ones flat instead of
    // stretching them into a solid block
    if (bw < 0.5 * bh) { const e = (0.5 * bh - bw) / 2; u0 -= e; u1 += e; bw = u1 - u0; }
    if (bh < 0.5 * bw) { const e = (0.5 * bw - bh) / 2; v0 -= e; v1 += e; bh = v1 - v0; }
    const { ox, oy, dx, dy, sh } = frame;
    const f = new Float32Array(GRID * GRID);
    const cu = bw / GRID, cv = bh / GRID;
    let mean = 0;
    for (let j = 0; j < GRID; j++) {
      for (let i = 0; i < GRID; i++) {
        let acc = 0;
        for (let sj = 0; sj < SUB; sj++) {
          const v = v0 + (j + (sj + 0.5) / SUB) * cv;
          for (let si = 0; si < SUB; si++) {
            const u = u0 + (i + (si + 0.5) / SUB) * cu + sh * v;
            acc += sample(ox + u * dx - v * dy, oy + u * dy + v * dx);
          }
        }
        f[j * GRID + i] = acc / (SUB * SUB);
        mean += f[j * GRID + i];
      }
    }
    mean /= f.length;
    let norm = 0;
    for (let i = 0; i < f.length; i++) { f[i] -= mean; norm += f[i] * f[i]; }
    norm = Math.sqrt(norm);
    if (norm < 1e-6) return null;
    for (let i = 0; i < f.length; i++) f[i] /= norm;
    return f;
  }

  // ------------------------------------------------------------ templates

  let templates = null;

  function buildTemplates() {
    if (templates) return templates;
    const chars = [];
    for (const ch of CLASSES) chars.push({ ch, cls: CLASSES.indexOf(ch), lower: false });
    for (const ch of 'abcdefghijklmnopqrstuvwxyz') chars.push({ ch, cls: CLASSES.indexOf(ch.toUpperCase()), lower: true });
    // letters whose mark can touch the body (other marks are dropped as
    // separate components, so e.g. Ü is read as U without its own template)
    for (const [ch, base] of [['Ç', 'C'], ['Ş', 'S'], ['ç', 'C'], ['ş', 'S']]) {
      chars.push({ ch, cls: CLASSES.indexOf(base), lower: ch !== ch.toUpperCase() });
    }

    const size = 64, cell = 112, cols = 16;
    const rows = Math.ceil(chars.length / cols);
    const cw = cols * cell, ch = rows * cell;
    const ctx = makeCanvas(cw, ch);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const vecs = [], meta = [], seen = new Set();
    const mask = new Uint8Array(cell * cell);
    const identity = { ox: 0, oy: 0, dx: 1, dy: 0 };

    for (const family of FONTS) {
      for (const weight of [400, 700]) {
        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, cw, ch);
        ctx.fillStyle = '#fff';
        ctx.font = `${weight} ${size}px ${family}`;
        chars.forEach((c, i) => ctx.fillText(c.ch, (i % cols + 0.5) * cell, (Math.floor(i / cols) + 0.5) * cell));
        const px = ctx.getImageData(0, 0, cw, ch).data;
        chars.forEach((c, i) => {
          const x0 = (i % cols) * cell, y0 = Math.floor(i / cols) * cell;
          for (let y = 0; y < cell; y++) {
            for (let x = 0; x < cell; x++) mask[y * cell + x] = px[((y0 + y) * cw + x0 + x) * 4] > 127 ? 1 : 0;
          }
          const { labels, comps } = components(mask, cell, cell);
          let big = null;
          for (let k = 1; k < comps.length; k++) if (!big || comps[k].area > big.area) big = comps[k];
          if (!big) return;
          const f = featureVector(makeSampler(labels, cell, cell, [big.id]), identity,
            { u0: big.minX, u1: big.maxX, v0: big.minY, v1: big.maxY });
          if (!f) return;
          // unavailable fonts fall back to the same face: skip duplicates
          let key = c.cls + ':';
          for (let k = 0; k < f.length; k += 3) key += Math.round(f[k] * 50) + ',';
          if (seen.has(key)) return;
          seen.add(key);
          vecs.push(f);
          meta.push({ cls: c.cls, lower: c.lower, ch: c.ch, font: `${weight} ${family}`,
            aspect: (big.maxX - big.minX + 1) / (big.maxY - big.minY + 1) });
        });
      }
    }
    const dim = GRID * GRID;
    const matrix = new Float32Array(vecs.length * dim);
    vecs.forEach((v, i) => matrix.set(v, i * dim));
    templates = { matrix, meta, count: vecs.length };
    return templates;
  }

  // Best score per class, once over upper-case/digit templates only and once
  // over all templates.
  function classify(f) {
    const { matrix, meta, count } = buildTemplates();
    const dim = GRID * GRID;
    const upper = new Float32Array(CLASSES.length).fill(-1);
    const all = new Float32Array(CLASSES.length).fill(-1);
    for (let t = 0; t < count; t++) {
      let s = 0;
      const off = t * dim;
      for (let k = 0; k < dim; k++) s += matrix[off + k] * f[k];
      const { cls, lower } = meta[t];
      if (s > all[cls]) all[cls] = s;
      if (!lower && s > upper[cls]) upper[cls] = s;
    }
    return { upper, all };
  }

  // ---------------------------------------------------------------- lines

  function median(arr) {
    if (!arr.length) return 0;
    const s = arr.slice().sort((a, b) => a - b);
    const m = s.length >> 1;
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  }

  // Dominant text direction from nearest-neighbour pairs (docstrum style).
  function dominantAngle(cands) {
    const bins = new Float64Array(91); // -45..45 degrees
    for (const a of cands) {
      let best = null, bestD = Infinity;
      for (const b of cands) {
        if (a === b) continue;
        const hr = Math.min(a.h, b.h) / Math.max(a.h, b.h);
        if (hr < 0.6) continue;
        const d = Math.hypot(b.bx - a.bx, b.by - a.by);
        if (d < bestD && d < 2.5 * Math.max(a.h, b.h)) { bestD = d; best = b; }
      }
      if (!best) continue;
      let dx = best.bx - a.bx, dy = best.by - a.by;
      if (dx < 0) { dx = -dx; dy = -dy; }
      const deg = Math.atan2(dy, dx) * 180 / Math.PI;
      if (deg < -45 || deg > 45) continue;
      bins[Math.round(deg) + 45] += 1;
    }
    let bestBin = 45, bestVal = -1;
    for (let i = 0; i < bins.length; i++) {
      let v = 0;
      for (let k = -2; k <= 2; k++) if (bins[i + k] !== undefined) v += bins[i + k] * (3 - Math.abs(k));
      if (v > bestVal) { bestVal = v; bestBin = i; }
    }
    return (bestBin - 45) * Math.PI / 180;
  }

  function findLines(cands, angle) {
    const dx = Math.cos(angle), dy = Math.sin(angle);
    for (const c of cands) {
      c.u = c.bx * dx + c.by * dy;
      c.v = -c.bx * dy + c.by * dx;
      c.next = c.prev = null;
      c.prevCost = Infinity;
    }
    const sorted = cands.slice().sort((a, b) => a.u - b.u);
    const maxH = cands.reduce((m, c) => Math.max(m, c.h), 0);
    for (let i = 0; i < sorted.length; i++) {
      const a = sorted[i];
      let best = null, bestCost = Infinity;
      for (let j = i + 1; j < sorted.length; j++) {
        const b = sorted[j];
        const du = b.u - a.u;
        if (du > 2.2 * maxH) break;
        const hMax = Math.max(a.h, b.h), hMin = Math.min(a.h, b.h);
        if (du > 2.2 * hMax || du <= 0 || hMin < 0.5 * hMax) continue;
        const dv = Math.abs(b.v - a.v);
        if (dv > 0.35 * hMax) continue;
        const cost = (du + 3 * dv) / hMax;
        if (cost < bestCost) { bestCost = cost; best = b; }
      }
      if (best && bestCost < best.prevCost) {
        if (best.prev) best.prev.next = null;
        best.prev = a;
        best.prevCost = bestCost;
        a.next = best;
      }
    }
    const lines = [];
    for (const c of sorted) {
      if (c.prev) continue;
      const line = [];
      for (let n = c; n; n = n.next) line.push(n);
      if (line.length >= 2) lines.push(line);
    }
    return lines;
  }

  // Theil-Sen slope of the glyph bottoms, clamped near the dominant angle.
  function lineAngle(line, fallback) {
    if (line.length < 4) return fallback;
    const slopes = [];
    for (let i = 0; i < line.length; i++) {
      for (let j = i + 1; j < line.length; j++) {
        const ddx = line[j].bx - line[i].bx;
        if (Math.abs(ddx) > 1) slopes.push((line[j].by - line[i].by) / ddx);
      }
    }
    const a = Math.atan(median(slopes));
    return Math.abs(a - fallback) < 0.14 ? a : fallback;
  }

  // Turn a chain of components into deskewed, classified glyphs.
  function buildLine(chain, labels, w, h, angle) {
    const frame = { ox: chain[0].bx, oy: chain[0].by, dx: Math.cos(angle), dy: Math.sin(angle) };
    // extents of each component in the rotated frame
    let glyphs = chain.map((c) => {
      let u0 = Infinity, u1 = -Infinity, v0 = Infinity, v1 = -Infinity;
      for (let y = c.minY; y <= c.maxY; y++) {
        for (let x = c.minX; x <= c.maxX; x++) {
          if (labels[y * w + x] !== c.id) continue;
          const rx = x - frame.ox, ry = y - frame.oy;
          const u = rx * frame.dx + ry * frame.dy, v = -rx * frame.dy + ry * frame.dx;
          if (u < u0) u0 = u;
          if (u > u1) u1 = u;
          if (v < v0) v0 = v;
          if (v > v1) v1 = v;
        }
      }
      return { ids: [c.id], u0, u1, v0, v1 };
    }).sort((a, b) => a.u0 - b.u0);

    // merge fragments of one letter that overlap along the line
    const merged = [];
    for (const g of glyphs) {
      const prev = merged[merged.length - 1];
      if (prev) {
        const overlap = Math.min(prev.u1, g.u1) - Math.max(prev.u0, g.u0);
        if (overlap > 0.5 * Math.min(prev.u1 - prev.u0, g.u1 - g.u0)) {
          prev.ids.push(...g.ids);
          prev.u0 = Math.min(prev.u0, g.u0);
          prev.u1 = Math.max(prev.u1, g.u1);
          prev.v0 = Math.min(prev.v0, g.v0);
          prev.v1 = Math.max(prev.v1, g.v1);
          continue;
        }
      }
      merged.push(g);
    }
    glyphs = merged;

    for (const g of glyphs) {
      g.h = g.v1 - g.v0 + 1;
      g.f = featureVector(makeSampler(labels, w, h, g.ids), frame, g);
      if (g.f) {
        const s = classify(g.f);
        g.upper = s.upper;
        g.all = s.all;
        g.bestUpper = Math.max(...s.upper);
        g.bestAll = Math.max(...s.all);
      } else {
        g.bestUpper = g.bestAll = -1;
      }
    }
    const heights = glyphs.map((g) => g.h);
    const gaps = [];
    for (let i = 1; i < glyphs.length; i++) gaps.push(glyphs[i].u0 - glyphs[i - 1].u1);
    return { frame, glyphs, height: median(heights), gap: median(gaps) };
  }

  // Run the whole detection pipeline and return every text line found.
  function readLines(img) {
    const integ = integrals(img);
    const short = Math.min(img.w, img.h);
    const minH = 10, maxH = 0.3 * short;
    const lines = [];
    const long = Math.max(img.w, img.h);
    for (const win of [Math.round(long / 60) | 1, Math.round(long / 20) | 1]) {
      for (const light of [true, false]) {
        const mask = binarize(img, integ, win, light, 0.3, 12);
        const { labels, comps } = components(mask, img.w, img.h);
        const cands = [];
        for (let k = 1; k < comps.length; k++) {
          const c = comps[k];
          if (c.h < minH || c.h > maxH || c.w > 3 * c.h) continue;
          if (c.area / (c.w * c.h) < 0.08) continue;
          c.bx = (c.minX + c.maxX) / 2;
          c.by = c.maxY;
          cands.push(c);
        }
        if (cands.length < 2) continue;
        const angle = dominantAngle(cands);
        for (const chain of findLines(cands, angle)) {
          const line = buildLine(chain, labels, img.w, img.h, lineAngle(chain, angle));
          line.light = light;
          line.win = win;
          lines.push(line);
        }
      }
    }
    return lines;
  }

  // ------------------------------------------------------------- matching

  const SPECIAL = { 'Ø': 'O', 'ø': 'o', 'Æ': 'AE', 'æ': 'ae', 'Œ': 'OE', 'œ': 'oe', 'Đ': 'D', 'đ': 'd',
    'Ł': 'L', 'ł': 'l', 'Þ': 'TH', 'þ': 'th', 'ı': 'i', 'ß': 'ss' };

  function nameTokens(name) {
    return name.replace(/[ØøÆæŒœĐđŁłÞþıß]/g, (c) => SPECIAL[c])
      .normalize('NFKD').replace(/[̀-ͯ]/g, '')
      .toUpperCase()
      .replace(/[^A-Z0-9]+/g, ' ')
      .trim()
      .split(' ')
      .filter(Boolean);
  }

  const MIN_SCORE = 0.6;    // expected letter must fit the glyph at least this well
  const MAX_MARGIN = 0.1;   // ...and be at most this far behind the best letter

  function isSeparator(line, i, side) {
    const glyphs = line.glyphs;
    const j = side < 0 ? i - 1 : i + 1;
    if (j < 0 || j >= glyphs.length) return true;
    const gap = side < 0 ? glyphs[i].u0 - glyphs[j].u1 : glyphs[j].u0 - glyphs[i].u1;
    if (gap >= Math.max(0.25 * line.height, 2 * line.gap)) return true;
    const n = glyphs[j];
    return n.bestAll < MIN_SCORE || n.h < 0.55 * line.height;
  }

  function matchToken(token, lines) {
    const n = token.length;
    const want = [...token].map((c) => CLASSES.indexOf(c));
    let best = null;
    for (const line of lines) {
      const g = line.glyphs;
      for (let i = 0; i + n <= g.length; i++) {
        if (!isSeparator(line, i, -1) || !isSeparator(line, i + n - 1, 1)) continue;
        const span = g.slice(i, i + n);
        if (span.some((x) => !x.f)) continue;
        const hs = span.map((x) => x.h);
        const hm = median(hs);
        const uniform = hs.filter((x) => Math.abs(x - hm) > 0.15 * hm).length <= (n > 3 ? 1 : 0);
        let total = 0, ok = true, worstMargin = 0;
        for (let k = 0; k < n && ok; k++) {
          const scores = uniform ? span[k].upper : span[k].all;
          const bestScore = uniform ? span[k].bestUpper : span[k].bestAll;
          const s = scores[want[k]];
          const margin = bestScore - s;
          if (s < MIN_SCORE || margin > MAX_MARGIN) ok = false;
          total += s;
          worstMargin = Math.max(worstMargin, margin);
        }
        if (!ok) continue;
        const score = total / n;
        if (!best || score > best.score) best = { token, score, worstMargin, line, start: i, end: i + n };
      }
    }
    return best;
  }

  function spanPolygon(m, scale) {
    const { frame, glyphs } = m.line;
    const span = glyphs.slice(m.start, m.end);
    const u0 = span[0].u0 - 2, u1 = span[span.length - 1].u1 + 2;
    const v0 = Math.min(...span.map((g) => g.v0)) - 2, v1 = Math.max(...span.map((g) => g.v1)) + 2;
    return [[u0, v0], [u1, v0], [u1, v1], [u0, v1]].map(([u, v]) => [
      (frame.ox + u * frame.dx - v * frame.dy) / scale,
      (frame.oy + u * frame.dy + v * frame.dx) / scale,
    ]);
  }

  async function findName(source, name) {
    const tokens = nameTokens(name);
    if (!tokens.length) return { match: false, tokens, found: [] };
    const bitmap = source instanceof ImageBitmap ? source : await createImageBitmap(source);
    const img = toGray(bitmap);
    const lines = readLines(img);
    const found = tokens.map((t) => matchToken(t, lines));
    return {
      match: found.every(Boolean),
      tokens,
      found: found.map((m, i) => m && { token: tokens[i], score: m.score, polygon: spanPolygon(m, img.scale) }),
      debug: { img, lines },
    };
  }

  global.NameOCR = { findName, nameTokens, _internals: { toGray, readLines, buildTemplates, CLASSES, integrals, binarize, components, featureVector, makeSampler, classify } };
})(window);
