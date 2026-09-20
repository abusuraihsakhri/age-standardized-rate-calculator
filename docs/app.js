"use strict";

const AGE_GROUPS = ["0-4","5-9","10-14","15-19","20-24","25-29","30-34","35-39","40-44","45-49","50-54","55-59","60-64","65-69","70-74","75-79","80-84","85+"];
const UNDER_75 = new Set(AGE_GROUPS.slice(0, 15));
const STANDARDS = {
  who2000: { label: "WHO World 2000-2025", weights: [8860,8690,8600,8470,8220,7930,7610,7150,6590,6040,5370,4550,3720,2960,2210,1520,910,635] },
  us2000: { label: "US 2000", weights: [18987000,19920000,20202000,20092000,19928000,19703000,20437000,22842000,22933000,19983000,17562000,13434000,10733000,9480000,8857000,7415000,4945000,4259000] },
  segi1960: { label: "Segi 1960", weights: [12000,10000,9000,8000,8000,6000,6000,6000,6000,6000,5000,4000,4000,3000,2000,1000,500,500] },
  european2013: { label: "European 2013", weights: [5000,5500,5500,5500,6000,6000,6500,7000,7000,7000,7000,6500,6000,5500,5000,4000,2500,1500] }
};
const EXAMPLE = [
  [1,52000],[1,51000],[2,50000],[3,49500],[5,49000],[8,48500],
  [14,48000],[22,47000],[38,46000],[61,44500],[95,42000],[140,38000],
  [190,32000],[230,26000],[250,20000],[240,14500],[190,9000],[120,5000]
];

const els = {
  rows: document.getElementById("dataRows"),
  standard: document.getElementById("standardSelect"),
  status: document.getElementById("inputStatus"),
  error: document.getElementById("errorBox"),
  empty: document.getElementById("emptyState"),
  results: document.getElementById("results"),
  download: document.getElementById("downloadButton"),
  theme: document.getElementById("themeToggle")
};

let lastResult = null;

function renderRows(values) {
  els.rows.replaceChildren();
  AGE_GROUPS.forEach(function(age, index) {
    const tr = document.createElement("tr");
    const ageCell = document.createElement("td");
    ageCell.textContent = age;

    const eventCell = document.createElement("td");
    const eventInput = document.createElement("input");
    eventInput.type = "number";
    eventInput.min = "0";
    eventInput.step = "any";
    eventInput.inputMode = "decimal";
    eventInput.dataset.kind = "events";
    eventInput.dataset.index = String(index);
    eventInput.setAttribute("aria-label", age + " events");
    eventInput.value = values ? String(values[index][0]) : "";
    eventCell.appendChild(eventInput);

    const pyCell = document.createElement("td");
    const pyInput = document.createElement("input");
    pyInput.type = "number";
    pyInput.min = "0.000001";
    pyInput.step = "any";
    pyInput.inputMode = "decimal";
    pyInput.dataset.kind = "py";
    pyInput.dataset.index = String(index);
    pyInput.setAttribute("aria-label", age + " person-years");
    pyInput.value = values ? String(values[index][1]) : "";
    pyCell.appendChild(pyInput);

    tr.append(ageCell, eventCell, pyCell);
    els.rows.appendChild(tr);
  });
}

function inputValue(kind, index) {
  const input = els.rows.querySelector('input[data-kind="' + kind + '"][data-index="' + index + '"]');
  return Number(input.value);
}

function readData() {
  return AGE_GROUPS.map(function(age, index) {
    const events = inputValue("events", index);
    const py = inputValue("py", index);
    if (!Number.isFinite(events) || events < 0) {
      throw new Error("Events must be finite and non-negative for " + age + ".");
    }
    if (!Number.isFinite(py) || py <= 0) {
      throw new Error("Person-years must be finite and greater than zero for " + age + ".");
    }
    return { age: age, events: events, py: py };
  });
}

function normalCdf(x) {
  return 0.5 * (1 + erf(x / Math.SQRT2));
}

function erf(x) {
  const sign = x < 0 ? -1 : 1;
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const ax = Math.abs(x);
  const t = 1 / (1 + p * ax);
  const y = 1 - (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t) * Math.exp(-ax * ax);
  return sign * y;
}

function normalPpf(p) {
  if (!(p > 0 && p < 1)) throw new Error("Probability must be between 0 and 1.");
  const a = [-39.69683028665376,220.9460984245205,-275.9285104469687,138.357751867269,-30.66479806614716,2.506628277459239];
  const b = [-54.47609879822406,161.5858368580409,-155.6989798598866,66.80131188771972,-13.28068155288572];
  const c = [-0.007784894002430293,-0.3223964580411365,-2.400758277161838,-2.549732539343734,4.374664141464968,2.938163982698783];
  const d = [0.007784695709041462,0.3224671290700398,2.445134137142996,3.754408661907416];
  const low = 0.02425, high = 1 - low;
  let q, r;
  if (p < low) {
    q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
  }
  if (p <= high) {
    q = p - 0.5;
    r = q*q;
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1);
  }
  q = Math.sqrt(-2 * Math.log(1-p));
  return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
}

function logGamma(z) {
  const p = [676.5203681218851,-1259.1392167224028,771.32342877765313,-176.61502916214059,12.507343278686905,-0.13857109526572012,9.9843695780195716e-6,1.5056327351493116e-7];
  if (z < 0.5) return Math.log(Math.PI) - Math.log(Math.sin(Math.PI*z)) - logGamma(1-z);
  z -= 1;
  let x = 0.99999999999980993;
  for (let i=0;i<p.length;i++) x += p[i]/(z+i+1);
  const t = z + p.length - 0.5;
  return 0.5*Math.log(2*Math.PI) + (z+0.5)*Math.log(t) - t + Math.log(x);
}

function gammaIncLower(a, x) {
  if (x <= 0) return 0;
  if (a <= 0) return 1;
  if (x < a + 1) {
    let ap = a, sum = 1/a, delta = sum;
    for (let i=0;i<100;i++) {
      ap += 1;
      delta *= x/ap;
      sum += delta;
      if (Math.abs(delta) < Math.abs(sum)*1e-15) break;
    }
    return sum * Math.exp(-x + a*Math.log(x) - logGamma(a));
  }
  let b = x + 1 - a, c = 1e30, d = 1/b, h = d;
  for (let i=1;i<100;i++) {
    const an = -i*(i-a);
    b += 2;
    d = an*d+b;
    if (Math.abs(d) < 1e-30) d = 1e-30;
    c = b + an/c;
    if (Math.abs(c) < 1e-30) c = 1e-30;
    d = 1/d;
    const delta = d*c;
    h *= delta;
    if (Math.abs(delta-1) < 1e-15) break;
  }
  const q = Math.exp(-x + a*Math.log(x) - logGamma(a))*h;
  return Math.max(0, Math.min(1, 1-q));
}

function chi2Cdf(x, df) {
  if (x <= 0) return 0;
  return gammaIncLower(df/2, x/2);
}

function chi2Ppf(p, df) {
  if (p <= 0) return 0;
  if (p >= 1) return Infinity;
  if (df <= 0) return 0;
  const z = normalPpf(p);
  const w = 2/(9*df);
  const term = 1-w+z*Math.sqrt(w);
  let x = term > 0 ? df*Math.pow(term,3) : df*Math.exp(z*Math.sqrt(2/df));
  x = Math.max(1e-8, x);
  for (let i=0;i<12;i++) {
    const err = chi2Cdf(x,df)-p;
    if (Math.abs(err) < 1e-12) break;
    const logPdf = -(df/2)*Math.log(2)-logGamma(df/2)+(df/2-1)*Math.log(x)-x/2;
    const pdf = Math.exp(logPdf);
    if (pdf > 1e-15) x = Math.max(1e-10, x-err/pdf);
  }
  return x;
}

function calculate() {
  const data = readData();
  const standard = STANDARDS[els.standard.value];
  const totalWeight = standard.weights.reduce(function(a,b){ return a+b; },0);
  const weights = standard.weights.map(function(w){ return w/totalWeight; });

  let asr = 0, variance = 0, totalEvents = 0, totalPy = 0, cumulativeRate = 0;
  data.forEach(function(row,index) {
    const rate = row.events/row.py;
    const w = weights[index];
    asr += w*rate;
    variance += w*w*row.events/(row.py*row.py);
    totalEvents += row.events;
    totalPy += row.py;
    if (UNDER_75.has(row.age)) cumulativeRate += 5*rate;
  });

  const se = Math.sqrt(variance);
  const z = normalPpf(0.975);
  const crude = totalEvents/totalPy;
  const wm = Math.max.apply(null, weights.map(function(w,index){ return w/data[index].py; }));
  let ffLow = 0, ffHigh = 0;
  if (asr === 0 || variance === 0) {
    ffHigh = (wm/2)*chi2Ppf(0.975,2);
  } else {
    const dfLow = 2*asr*asr/variance;
    ffLow = (variance/(2*asr))*chi2Ppf(0.025,dfLow);
    const dfHigh = 2*Math.pow(asr+wm,2)/(variance+wm*wm);
    ffHigh = ((variance+wm*wm)/(2*(asr+wm)))*chi2Ppf(0.975,dfHigh);
  }
  const waldLow = Math.max(0, asr-z*se);
  const waldHigh = asr+z*se;
  let logLow = 0, logHigh = 0;
  if (asr > 0) {
    const factor = Math.exp(z*se/asr);
    logLow = asr/factor;
    logHigh = asr*factor;
  }

  return {
    standard: standard.label,
    asr: asr*100000,
    ffLow: ffLow*100000,
    ffHigh: ffHigh*100000,
    crude: crude*100000,
    se: se*100000,
    cumulativeRisk: (1-Math.exp(-cumulativeRate))*100,
    totalEvents: totalEvents,
    totalPy: totalPy,
    waldLow: waldLow*100000,
    waldHigh: waldHigh*100000,
    logLow: logLow*100000,
    logHigh: logHigh*100000
  };
}

function formatNumber(value, digits) {
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits }).format(value);
}

function showResult(result) {
  lastResult = result;
  els.error.hidden = true;
  els.empty.hidden = true;
  els.results.hidden = false;
  els.download.disabled = false;
  document.getElementById("asrValue").textContent = formatNumber(result.asr,2);
  document.getElementById("standardName").textContent = result.standard;
  document.getElementById("ciValue").textContent = formatNumber(result.ffLow,2) + " – " + formatNumber(result.ffHigh,2);
  document.getElementById("crudeValue").textContent = formatNumber(result.crude,2);
  document.getElementById("seValue").textContent = formatNumber(result.se,2);
  document.getElementById("riskValue").textContent = formatNumber(result.cumulativeRisk,2) + "%";
  document.getElementById("eventsValue").textContent = formatNumber(result.totalEvents,0);
  document.getElementById("pyValue").textContent = formatNumber(result.totalPy,0);
  document.getElementById("waldValue").textContent = formatNumber(result.waldLow,2) + " – " + formatNumber(result.waldHigh,2);
  document.getElementById("logValue").textContent = formatNumber(result.logLow,2) + " – " + formatNumber(result.logHigh,2);
  els.status.textContent = "Calculation complete.";
}

function showError(error) {
  lastResult = null;
  els.download.disabled = true;
  els.error.textContent = error instanceof Error ? error.message : String(error);
  els.error.hidden = false;
  els.results.hidden = true;
  els.empty.hidden = false;
  els.status.textContent = "Check the highlighted input requirements.";
}

function parseCsvLine(line) {
  const out = [];
  let current = "", quoted = false;
  for (let i=0;i<line.length;i++) {
    const ch = line[i];
    if (ch === '"') {
      if (quoted && line[i+1] === '"') { current += '"'; i++; }
      else quoted = !quoted;
    } else if (ch === "," && !quoted) {
      out.push(current.trim()); current = "";
    } else current += ch;
  }
  out.push(current.trim());
  return out;
}

function importCsv(text) {
  const lines = text.replace(/^\uFEFF/,"").split(/\r?\n/).filter(function(line){ return line.trim(); });
  if (lines.length < 2) throw new Error("CSV must include a header and at least one data row.");
  const headers = parseCsvLine(lines[0]).map(function(v){ return v.toLowerCase(); });
  const ageIndex = headers.findIndex(function(v){ return v === "age_group" || v === "age"; });
  const countIndex = headers.findIndex(function(v){ return v === "count" || v === "cases" || v === "events"; });
  const pyIndex = headers.findIndex(function(v){ return v === "person_years" || v === "population" || v === "py"; });
  if (ageIndex < 0 || countIndex < 0 || pyIndex < 0) throw new Error("CSV needs age, event-count, and person-years columns.");

  const map = new Map();
  lines.slice(1).forEach(function(line) {
    const fields = parseCsvLine(line);
    const age = (fields[ageIndex] || "").trim();
    if (!AGE_GROUPS.includes(age)) throw new Error("Unsupported or unexpected age group: " + age);
    if (map.has(age)) throw new Error("Duplicate age group in CSV: " + age);
    const events = Number(fields[countIndex]);
    const py = Number(fields[pyIndex]);
    if (!Number.isFinite(events) || events < 0 || !Number.isFinite(py) || py <= 0) {
      throw new Error("Invalid numeric data for " + age + ".");
    }
    map.set(age,[events,py]);
  });
  AGE_GROUPS.forEach(function(age){ if (!map.has(age)) throw new Error("CSV is missing age group " + age + "."); });
  renderRows(AGE_GROUPS.map(function(age){ return map.get(age); }));
  els.status.textContent = "CSV loaded locally.";
}

function downloadResult() {
  if (!lastResult) return;
  const rows = [
    ["metric","value"],
    ["standard",lastResult.standard],
    ["asr_per_100k",lastResult.asr],
    ["fay_feuer_ci_lower",lastResult.ffLow],
    ["fay_feuer_ci_upper",lastResult.ffHigh],
    ["crude_rate_per_100k",lastResult.crude],
    ["standard_error_per_100k",lastResult.se],
    ["cumulative_risk_0_74_pct",lastResult.cumulativeRisk],
    ["total_events",lastResult.totalEvents],
    ["total_person_years",lastResult.totalPy]
  ];
  const csv = rows.map(function(row){ return row.map(function(v){ return '"' + String(v).replace(/"/g,'""') + '"'; }).join(","); }).join("\n");
  const blob = new Blob([csv],{type:"text/csv;charset=utf-8"});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "asr-result.csv";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  els.theme.textContent = theme === "dark" ? "Light" : "Dark";
  els.theme.setAttribute("aria-label", theme === "dark" ? "Switch to light theme" : "Switch to dark theme");
  try { localStorage.setItem("asr-theme",theme); } catch (_) {}
}

document.getElementById("exampleButton").addEventListener("click", function() {
  renderRows(EXAMPLE);
  els.status.textContent = "Example data loaded.";
  els.error.hidden = true;
});
document.getElementById("clearButton").addEventListener("click", function() {
  renderRows(null);
  lastResult = null;
  els.results.hidden = true;
  els.empty.hidden = false;
  els.error.hidden = true;
  els.download.disabled = true;
  els.status.textContent = "Cleared.";
});
document.getElementById("calculateButton").addEventListener("click", function() {
  try { showResult(calculate()); } catch (error) { showError(error); }
});
document.getElementById("downloadButton").addEventListener("click", downloadResult);
document.getElementById("csvInput").addEventListener("change", async function(event) {
  const file = event.target.files && event.target.files[0];
  if (!file) return;
  if (file.size > 2 * 1024 * 1024) { showError(new Error("CSV file is larger than 2 MB.")); return; }
  try { importCsv(await file.text()); els.error.hidden = true; } catch (error) { showError(error); }
  event.target.value = "";
});
els.theme.addEventListener("click", function() {
  setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
});

let savedTheme = "light";
try { savedTheme = localStorage.getItem("asr-theme") || "light"; } catch (_) {}
setTheme(savedTheme === "dark" ? "dark" : "light");
renderRows(null);
