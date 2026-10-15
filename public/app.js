const $ = id => document.getElementById(id);
const money = value => new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 2 }).format(value);
let sample = [];

function status(message, error = false) {
  $('status').textContent = message;
  $('status').className = error ? 'error' : '';
}

// Small SVG chart, without a charting library or build step.
function drawChart(history, predictions = []) {
  const past = history.slice(-30), all = [...past, ...predictions];
  const w = 720, h = 220, left = 38, right = 12, top = 15, bottom = 30;
  const max = Math.max(1, ...all.map(p => p.kwh)) * 1.15;
  const x = i => left + i * (w - left - right) / Math.max(1, all.length - 1);
  const y = value => h - bottom - value / max * (h - top - bottom);
  let svg = '';
  for (let i = 0; i <= 4; i++) {
    const value = max * i / 4, yy = y(value);
    svg += `<line x1="${left}" x2="${w-right}" y1="${yy}" y2="${yy}" stroke="#e9eee6"/><text x="${left-9}" y="${yy+3}" fill="#7b8b80" font-size="9" text-anchor="end">${value.toFixed(0)}</text>`;
  }
  const points = (rows, offset) => rows.map((p, i) => `${x(i + offset)},${y(p.kwh)}`).join(' ');
  svg += `<polyline points="${points(past, 0)}" fill="none" stroke="#34725b" stroke-width="2.5" stroke-linejoin="round"/>`;
  if (predictions.length) {
    const boundary = x(past.length - 1);
    svg += `<line x1="${boundary}" x2="${boundary}" y1="${top}" y2="${h-bottom}" stroke="#b9c9b3" stroke-dasharray="4 4"/><polyline points="${points([past.at(-1), ...predictions], past.length - 1)}" fill="none" stroke="#b28b20" stroke-width="2.5" stroke-dasharray="6 4" stroke-linejoin="round"/>`;
  }
  for (const index of [...new Set([0, Math.floor((all.length-1)/2), all.length-1])]) {
    svg += `<text x="${x(index)}" y="${h-7}" fill="#7b8b80" font-size="9" text-anchor="middle">${all[index].date.slice(5)}</text>`;
  }
  $('chart').innerHTML = `<svg viewBox="0 0 ${w} ${h}" aria-hidden="true">${svg}</svg>`;
  $('chart').setAttribute('aria-label', `Daily kWh: ${past.length} historical days and ${predictions.length} forecast days.`);
}

function restore() {
  $('last-date').value = sample.at(-1).date;
  $('readings').value = sample.slice(-28).map(p => p.kwh.toFixed(3)).join(', ');
}

async function predict(event) {
  event?.preventDefault();
  $('predict').disabled = true;
  status('Calculating your forecast…');
  try {
    const tokens = $('readings').value.trim().split(/[,\n]+/);
    if (tokens.some(v => !v.trim())) throw new Error('Remove blank readings and use one number per day.');
    const history = tokens.map(v => Number(v.trim()));
    const lastDate = $('last-date').value;
    const response = await fetch('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ history, lastDate, days: Number($('days').value), rate: Number($('rate').value), fixedCharge: Number($('fixed').value) }) });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error);
    $('total-kwh').textContent = result.totalKwh.toFixed(1);
    $('bill-total').textContent = money(result.bill.total);
    $('period-note').textContent = `${result.predictions.length} days · ${result.predictions[0].date} to ${result.predictions.at(-1).date}`;
    $('bill-note').textContent = `${money(result.bill.energyCharge)} energy + ${money(result.bill.fixedCharge)} fixed`;
    const daily = history.map((kwh, i) => {
      const date = new Date(`${lastDate}T00:00:00Z`);
      date.setUTCDate(date.getUTCDate() - history.length + 1 + i);
      return { date: date.toISOString().slice(0, 10), kwh };
    });
    drawChart(daily, result.predictions);
    $('chart-note').textContent = `Daily kWh · Forecast starts after ${lastDate}. Forecasts are estimates, not guaranteed consumption.`;
    status('Forecast ready. Edit the inputs to explore another estimate.');
  } catch (error) {
    $('total-kwh').textContent = '—'; $('bill-total').textContent = '—';
    $('period-note').textContent = 'Forecast unavailable';
    $('bill-note').textContent = 'Check your inputs and try again';
    $('chart').replaceChildren();
    status(error.message, true);
  } finally { $('predict').disabled = false; }
}

async function init() {
  $('predict').disabled = true;
  try {
    const response = await fetch('/api/data');
    const data = await response.json();
    if (!response.ok) throw new Error(data.error);
    sample = data.history;
    const r = data.report, s = r.statistics;
    const bestMse = r.models.reduce((best, model) => model.test.mse < best.test.mse ? model : best);
    const bestR2 = r.models.reduce((best, model) => model.test.r2 > best.test.r2 ? model : best);
    $('comparison-summary').textContent = `Lowest test MSE: ${bestMse.name} (${bestMse.test.mse.toFixed(2)} kWh²). Highest test R²: ${bestR2.name} (${bestR2.test.r2.toFixed(3)}). Production model selection still uses validation MAE.`;
    const classification = r.high_usage_classification;
    $('classification-note').textContent = `High usage means daily kWh ≥ ${classification.threshold_kwh.toFixed(2)}, the median observed training consumption. Accuracy: correct high/normal labels. Precision: fraction of predicted high days that were high. Recall: fraction of actual high days detected. ${classification.test_positive_days} of ${classification.test_total_days} test days were high. These are derived classification scores, not regression accuracy.`;
    restore();
    $('selected-model').textContent = `Selected: ${r.selected_model}`;
    // Names are generated by our local training script, not uploaded content.
    $('model-rows').innerHTML = r.models.map(m => `<tr class="${m.name === r.selected_model ? 'selected' : ''}"><td>${m.name}${m.name === r.selected_model ? '<span class="best">Selected</span>' : ''}</td><td>${m.validation_before.mae.toFixed(2)}</td><td>${m.validation_after.mae.toFixed(2)}</td><td>${m.test.mae.toFixed(2)}</td><td>${m.test.rmse.toFixed(2)}</td><td>${m.test.r2.toFixed(3)}</td></tr>`).join('') + `<tr><td>Previous-day baseline</td><td>—</td><td>—</td><td>${r.baseline.test.mae.toFixed(2)}</td><td>${r.baseline.test.rmse.toFixed(2)}</td><td>${r.baseline.test.r2.toFixed(3)}</td></tr>`;
    const horizon = r.recursive_30_day_test;
    $('evaluation-note').textContent = `30-day recursive test MAE: ${horizon.metrics?.mae.toFixed(2) ?? 'N/A'} kWh across ${horizon.windows.length} windows. Next-day metrics use actual past readings. Multi-day forecasts reuse predictions. Untuned settings are retained if tuning worsens validation MAE.`;
    $('data-note').textContent = `Dataset: daily.csv · ${s.daily_rows.toLocaleString()} daily records (${s.start} to ${s.end}). ${s.imputed_days} filled days are flagged and excluded as training/evaluation targets. Average: ${s.mean_daily_kwh} kWh/day.`;
    $('split-note').textContent = `Chronological 70/15/15 split: training ends ${r.split.train_end}; validation ends ${r.split.validation_end}; test ${r.split.test_start} to ${r.split.test_end}. Tuning uses 4-fold time-series cross-validation on training data.`;
    $('preprocessing-note').textContent = r.preprocessing_note;
    await predict();
  } catch (error) { status(error.message, true); }
}

$('forecast-form').addEventListener('submit', predict);
$('reset').addEventListener('click', () => { restore(); predict(); });
init();
