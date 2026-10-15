import express from 'express';
import { readFileSync, existsSync } from 'node:fs';
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { calculateBill } from './bill.js';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const readReport = () => JSON.parse(readFileSync(path.join(root, 'models/report.json'), 'utf8'));
export const app = express();
app.disable('x-powered-by');
app.use(express.json({ limit: '16kb' }));
app.use(express.static(path.join(root, 'public')));

export function validateInput(body) {
  const report = readReport();
  const household = {};
  for (const field of report.input_schema) {
    const value = body?.household?.[field.name];
    if (field.type === 'select') {
      if (!field.options.includes(value)) throw new Error(`Choose a supported ${field.label.toLowerCase()}.`);
    } else if (!Number.isInteger(value) || value < field.min || value > field.max) {
      throw new Error(`${field.label} must be an integer between ${field.min} and ${field.max}.`);
    }
    household[field.name] = value;
  }
  const { rate, fixedCharge } = body ?? {};
  if (typeof rate !== 'number' || !Number.isFinite(rate) || rate < 0 || rate > 1000 ||
      typeof fixedCharge !== 'number' || !Number.isFinite(fixedCharge) || fixedCharge < 0 || fixedCharge > 100000) {
    throw new Error('Enter a rate of 0-1,000 BDT and a fixed charge of 0-100,000 BDT.');
  }
  return { household, rate, fixedCharge };
}

app.get('/api/data', (req, res, next) => {
  try { res.json({ report: readReport() }); }
  catch (error) { next(error); }
});

let busy = false;
app.post('/api/predict', (req, res) => {
  let input;
  try { input = validateInput(req.body); }
  catch (error) { return res.status(400).json({ error: error.message }); }
  if (busy) return res.status(503).json({ error: 'A prediction is running. Please try again shortly.' });
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
    if (!res.headersSent) res.status(500).json({ error: 'Python could not start. Check requirements and PYTHON_PATH.' });
  });
  child.on('close', code => {
    clearTimeout(timer); busy = false;
    if (res.headersSent) return;
    if (code !== 0 || timedOut) {
      console.error(errors);
      return res.status(500).json({ error: 'Prediction failed. Check Python dependencies and train the model first.' });
    }
    try {
      const { monthlyKwh } = JSON.parse(output);
      if (!Number.isFinite(monthlyKwh) || monthlyKwh < 0) throw new Error('Invalid prediction');
      res.json({ monthlyKwh, bill: calculateBill(monthlyKwh, input.rate, input.fixedCharge) });
    } catch { res.status(500).json({ error: 'The model returned an invalid response.' }); }
  });
  child.stdin.end(JSON.stringify(input.household));
});

app.use((error, req, res, next) => {
  console.error(error.message);
  const status = error.status === 400 || error.status === 413 ? error.status : 500;
  res.status(status).json({ error: status === 500 ? 'Model report unavailable. Run python ml/train.py first.' : 'Invalid or oversized JSON request.' });
});
