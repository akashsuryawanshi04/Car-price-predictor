/**
 * main.js — CarValue AI
 * Populates dropdowns from /api/meta and handles price prediction.
 */

"use strict";

// ── State ─────────────────────────────────────────────────────────────────────
let META = null;

// ── Helpers ───────────────────────────────────────────────────────────────────
const $  = (id) => document.getElementById(id);
const el = (tag, cls, txt) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (txt) e.textContent = txt;
  return e;
};

function populateSelect(selectId, options, placeholder) {
  const select = $(selectId);
  select.innerHTML = `<option value="">${placeholder}</option>`;
  options.forEach(opt => {
    const o = document.createElement("option");
    o.value = opt;
    o.textContent = opt;
    select.appendChild(o);
  });
}

function formatINR(n) {
  n = Math.round(n);
  const s = String(n);
  if (s.length <= 3) return "₹ " + s;
  const last3 = s.slice(-3);
  let rest = s.slice(0, -3);
  const parts = [];
  while (rest.length > 2) { parts.unshift(rest.slice(-2)); rest = rest.slice(0, -2); }
  if (rest) parts.unshift(rest);
  return "₹ " + parts.join(",") + "," + last3;
}

// ── Load metadata & populate UI ───────────────────────────────────────────────
async function loadMeta() {
  try {
    const resp = await fetch("/api/meta");
    META = await resp.json();

    // Stats row
    $("statR2").textContent    = META.metrics.R2.toFixed(2);
    $("statMAE").textContent   = formatINR(META.metrics.MAE);
    $("statTrain").textContent = META.train_samples;
    $("statTest").textContent  = META.test_samples + " (20%)";

    // Dropdowns
    populateSelect("carName",  META.car_names,  "— Select car model —");
    populateSelect("company",  META.companies,  "— Select brand —");
    populateSelect("fuelType", META.fuel_types, "— Select fuel type —");

    // Year hints
    const [yMin, yMax] = META.year_range;
    $("yearHint").textContent = `Range: ${yMin} – ${yMax}`;
    $("year").min = yMin;
    $("year").max = yMax;
    $("year").placeholder = `e.g. ${yMax - 3}`;

    // KMs hints
    const [kMin, kMax] = META.kms_range;
    $("kmsHint").textContent = `Range: ${kMin.toLocaleString()} – ${kMax.toLocaleString()} kms`;
    $("kmsDriven").min = kMin;

  } catch (err) {
    console.error("[loadMeta] Error:", err);
  }
}

// ── Predict ───────────────────────────────────────────────────────────────────
async function predict() {
  hideResult();
  hideError();

  // Gather inputs
  const name       = $("carName").value.trim();
  const company    = $("company").value.trim();
  const fuel_type  = $("fuelType").value.trim();
  const year       = parseInt($("year").value);
  const kms_driven = parseInt($("kmsDriven").value);

  // Validation
  const missing = [];
  if (!name)        missing.push("Car Model");
  if (!company)     missing.push("Brand");
  if (!fuel_type)   missing.push("Fuel Type");
  if (!year)        missing.push("Year");
  if (!kms_driven && kms_driven !== 0) missing.push("KMs Driven");

  if (missing.length) {
    showError(`Please fill in: ${missing.join(", ")}`);
    return;
  }

  // Validate ranges
  if (META) {
    const [yMin, yMax] = META.year_range;
    if (year < yMin || year > yMax) {
      showError(`Year must be between ${yMin} and ${yMax}.`);
      return;
    }
  }

  // Loading state
  setLoading(true);

  try {
    const resp = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, company, year, kms_driven, fuel_type }),
    });
    const data = await resp.json();

    if (!resp.ok || data.error) {
      showError(data.error || "Prediction failed. Please try again.");
      return;
    }

    showResult(data.formatted, { name, company, year, kms_driven, fuel_type });

  } catch (err) {
    showError("Network error — make sure the Flask server is running.");
  } finally {
    setLoading(false);
  }
}

// ── UI state helpers ──────────────────────────────────────────────────────────
function setLoading(on) {
  const btn    = $("predictBtn");
  const loader = $("btnLoader");
  const text   = btn.querySelector(".btn-text");
  const icon   = btn.querySelector(".btn-icon");

  btn.disabled  = on;
  loader.style.display = on ? "block" : "none";
  text.textContent     = on ? "Predicting…" : "Estimate Price";
  icon.style.display   = on ? "none" : "inline";
}

function showResult(formattedPrice, inputs) {
  const panel = $("resultPanel");
  const price = $("resultPrice");
  const tags  = $("resultTags");

  price.textContent = formattedPrice;

  // Build tags from inputs
  tags.innerHTML = "";
  const tagData = [
    inputs.company,
    inputs.fuel_type + " Engine",
    inputs.year + " Model",
    Number(inputs.kms_driven).toLocaleString() + " kms",
  ];
  tagData.forEach(t => {
    const span = el("span", "tag", t);
    tags.appendChild(span);
  });

  panel.style.display = "block";
  panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function hideResult() { $("resultPanel").style.display = "none"; }

function showError(msg) {
  const panel = $("errorPanel");
  $("errorMsg").textContent = msg;
  panel.style.display = "flex";
}

function hideError() { $("errorPanel").style.display = "none"; }

// ── Enter key shortcut ────────────────────────────────────────────────────────
document.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !$("predictBtn").disabled) predict();
});

// ── Input event — hide result on change ───────────────────────────────────────
["carName","company","fuelType","year","kmsDriven"].forEach(id => {
  const el = $(id);
  if (el) el.addEventListener("change", () => { hideResult(); hideError(); });
});

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", loadMeta);
