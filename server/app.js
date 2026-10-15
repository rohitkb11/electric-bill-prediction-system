import express from 'express';
import { readFileSync, existsSync } from 'node:fs';
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { calculateBill } from './bill.js';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const app = express();
app.disable('x-powered-by');
app.use(express.json({ limit: '32kb' }));
app.use(express.static(path.join(root, 'public')));

export function validateForecast(body) {
  const { history, lastDate, days, rate, fixedCharge } = body ?? {};
  if (!Array.isArray(history) || history.length < 28 || history.length > 3660 ||
      history.some(n => typeof n !== 'number' || !Number.isFinite(n) || n < 0 || n > 10000)) {
    throw new Error('Enter 28–3660 consecutive daily readings, each between 0 and 10,000 kWh.');
  }
  const date = new Date(lastDate);
  if (typeof lastDate !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(lastDate) ||
      !Number.isFinite(date.getTime()) || date.toISOString().slice(0, 10) !== lastDate ||
      lastDate < '1900-01-01' || lastDate > '2099-12-31') throw new Error('Enter a valid final reading date (1900–2099).');
  if (!Number.isInteger(days) || days < 1 || days > 30) throw new Error('Forecast length must be 1–30 days.');
  if (typeof rate !== 'number' || !Number.isFinite(rate) || rate < 0 || rate > 1000 ||
      typeof fixedCharge !== 'number' || !Number.isFinite(fixedCharge) || fixedCharge < 0 || fixedCharge > 100000) {
    throw new Error('Enter a rate of ₹0–1,000 and a fixed charge of ₹0–100,000.');
  }
  return { history, lastDate, days, rate, fixedCharge };
}

app.get('/api/data', (req, res, next) => {
  try {
    const report = JSON.parse(readFileSync(path.join(root, 'models/report.json'), 'utf8'));
    const rows = readFileSync(path.join(root, 'data/daily.csv'), 'utf8').trim().split(/\r?\n/).slice(1);
    const history = rows.slice(-90).map(line => {
      const [date, kwh, imputed] = line.split(',');
      return { date, kwh: Number(kwh), imputed: imputed === 'True' };
    });
    res.json({ report, history });
  } catch (error) { next(error); }
});

let busy = false;
app.post('/api/predict', (req, res) => {
  let input;
  try { input = validateForecast(req.body); }
  catch (error) { return res.status(400).json({ error: error.message }); }
  if (busy) return res.status(503).json({ error: 'A forecast is running. Please try again shortly.' });
  const venv = path.join(root, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
  const python = process.env.PYTHON_PATH || (existsSync(venv) ? venv : 'python');
  busy = true;
  const child = spawn(python, [path.join(root, 'ml/predict.py')], { cwd: root, windowsHide: true });
  let output = '', errors = '', timedOut = false;
  const timer = setTimeout(() => { timedOut = true; child.kill(); }, 30000);
  child.stdout.on('data', chunk => { output += chunk; });
  child.stderr.on('data', chunk => { errors += chunk; });
  child.stdin.on('error', () => {});
  child.on('error', error => {
    clearTimeout(timer); busy = false;
    console.error(error.message);
    if (!res.headersSent) res.status(500).json({ error: 'Python could not start. Install requirements and check PYTHON_PATH.' });
  });
  child.on('close', code => {
    clearTimeout(timer); busy = false;
    if (res.headersSent) return;
    if (code !== 0 || timedOut) {
      console.error(errors);
      return res.status(500).json({ error: 'Prediction failed. Check Python dependencies and train the model first.' });
    }
    try {
      const { predictions } = JSON.parse(output);
      const totalKwh = Math.round(predictions.reduce((sum, row) => sum + row.kwh, 0) * 100) / 100;
      res.json({ predictions, totalKwh, bill: calculateBill(totalKwh, input.rate, input.fixedCharge) });
    } catch { res.status(500).json({ error: 'The model returned an invalid response.' }); }
  });
  child.stdin.end(JSON.stringify({ history: input.history, last_date: input.lastDate, days: input.days }));
});

app.use((error, req, res, next) => {
  console.error(error.message);
  const status = error.status === 400 || error.status === 413 ? error.status : 500;
  res.status(status).json({ error: status === 500 ? 'Data unavailable. Check data/daily.csv and run training first.' : 'Invalid or oversized JSON request.' });
});
