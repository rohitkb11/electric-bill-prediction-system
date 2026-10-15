const $ = id => document.getElementById(id);
const money = value => new Intl.NumberFormat('en-BD', { style: 'currency', currency: 'BDT', maximumFractionDigits: 2 }).format(value);
let report;

function status(message, error = false) {
  $('status').textContent = message;
  $('status').className = error ? 'error' : '';
}

function buildForm() {
  const fields = [...report.input_schema].sort((a, b) => (b.type === 'select') - (a.type === 'select'));
  for (const field of fields) {
    const group = document.createElement('div');
    const label = document.createElement('label');
    label.htmlFor = field.name;
    label.textContent = field.label;
    const input = document.createElement(field.type === 'select' ? 'select' : 'input');
    input.id = field.name;
    input.required = true;
    if (field.type === 'select') {
      for (const option of field.options) input.add(new Option(option, option));
    } else {
      input.type = 'number'; input.min = field.min; input.max = field.max; input.step = 1;
    }
    group.append(label, input);
    $('household-fields').append(group);
  }
  restore();
}

function restore() {
  for (const [name, value] of Object.entries(report.sample_input)) $(name).value = value;
}

async function predict(event) {
  event?.preventDefault();
  $('predict').disabled = true;
  status('Calculating your monthly estimate…');
  try {
    const household = Object.fromEntries(report.input_schema.map(field =>
      [field.name, field.type === 'number' ? Number($(field.name).value) : $(field.name).value]));
    const response = await fetch('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ household, rate: Number($('rate').value), fixedCharge: Number($('fixed').value) }) });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error);
    $('total-kwh').textContent = result.monthlyKwh.toFixed(1);
    $('bill-total').textContent = money(result.bill.total);
    $('bill-note').textContent = `${money(result.bill.energyCharge)} energy + ${money(result.bill.fixedCharge)} fixed`;
    status('Estimate ready. Change the household details or tariff and try again.');
  } catch (error) {
    $('total-kwh').textContent = '—'; $('bill-total').textContent = '—';
    $('bill-note').textContent = 'Check your inputs and try again';
    status(error.message, true);
  } finally { $('predict').disabled = false; }
}

async function init() {
  $('predict').disabled = true;
  try {
    const response = await fetch('/api/data');
    const data = await response.json();
    if (!response.ok) throw new Error(data.error);
    report = data.report;
    buildForm();
    $('selected-model').textContent = report.selected_variant;
    $('model-rows').innerHTML = report.models.map(m => `<tr class="${m.name === report.selected_variant ? 'selected' : ''}"><td>${m.name}</td><td>${m.cv_mae.toFixed(2)}</td><td>${m.test.mae.toFixed(2)}</td><td>${m.test.mse.toFixed(2)}</td><td>${m.test.rmse.toFixed(2)}</td><td>${m.test.r2.toFixed(3)}</td></tr>`).join('');
    $('optimization-note').textContent = report.optimization.conclusion;
    $('data-note').textContent = `${report.dataset} · ${report.statistics.usable_rows} household records. Inputs: location, area type, family members, rooms and appliances. Target: monthly kWh.`;
    $('split-note').textContent = `Training: ${report.split.training_rows} households. Test: ${report.split.test_rows} households. The saved model was trained only on training households.`;
    $('preprocessing-note').textContent = report.preprocessing_note;
    $('classification-note').textContent = `High usage means monthly kWh ≥ ${report.high_usage_classification.threshold_kwh.toFixed(2)} (training median). Accuracy, precision and recall describe these high/normal labels, not regression accuracy.`;
    await predict();
  } catch (error) { status(error.message, true); }
}

$('forecast-form').addEventListener('submit', predict);
$('reset').addEventListener('click', () => { restore(); predict(); });
init();
