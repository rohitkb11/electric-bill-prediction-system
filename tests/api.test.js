import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { app, validateForecast } from '../server/app.js';
import { calculateBill } from '../server/bill.js';

const input = { history: Array(28).fill(12), lastDate: '2010-11-25', days: 7, rate: 7, fixedCharge: 100 };
let server, base;
before(async () => {
  server = app.listen(0, '127.0.0.1');
  await new Promise(resolve => server.once('listening', resolve));
  base = `http://127.0.0.1:${server.address().port}`;
});
after(() => new Promise(resolve => server.close(resolve)));

test('bill uses energy charge plus one fixed charge', () => {
  assert.deepEqual(calculateBill(100.25, 7, 100), { energyCharge: 701.75, fixedCharge: 100, total: 801.75 });
  assert.equal(calculateBill(0, 7, 100).total, 100);
});
test('validates daily history, dates, horizon and tariff', () => {
  assert.deepEqual(validateForecast(input), input);
  for (const bad of [{ history: Array(27).fill(1) }, { history: [...Array(27).fill(1), -1] },
    { history: [...Array(27).fill(1), null] }, { lastDate: '2010-02-30' }, { days: 31 },
    { days: 1.5 }, { rate: -1 }, { fixedCharge: '100' }]) {
    assert.throws(() => validateForecast({ ...input, ...bad }));
  }
});
test('serves dashboard and real dataset report', async () => {
  const page = await fetch(base);
  assert.equal(page.status, 200);
  assert.match(await page.text(), /Electric Bill Prediction System/);
  const res = await fetch(`${base}/api/data`);
  assert.equal(res.status, 200);
  const data = await res.json();
  assert.equal(data.history.length, 90);
  assert.ok(data.report.statistics.raw_rows > 2000000);
  assert.deepEqual(data.report.models.map(model => model.name), ['Linear Regression', 'Random Forest']);
});
test('rejects invalid prediction input with helpful JSON', async () => {
  const response = await fetch(`${base}/api/predict`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
  assert.equal(response.status, 400);
  assert.match((await response.json()).error, /28/);
});
test('Express calls Python model and returns a consistent bill', async () => {
  const data = await (await fetch(`${base}/api/data`)).json();
  const response = await fetch(`${base}/api/predict`, { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...input, history: data.history.slice(-28).map(r => r.kwh), lastDate: data.history.at(-1).date, days: 30 }) });
  assert.equal(response.status, 200);
  const result = await response.json();
  assert.equal(result.predictions.length, 30);
  assert.equal(result.predictions[0].date, '2010-11-26');
  assert.ok(result.predictions.every(r => Number.isFinite(r.kwh) && r.kwh >= 0));
  assert.equal(result.bill.total, calculateBill(result.totalKwh, input.rate, input.fixedCharge).total);
});
