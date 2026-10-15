import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { app, validateInput } from '../server/app.js';
import { calculateBill } from '../server/bill.js';

let server, base, report, input;
before(async () => {
  server = app.listen(0, '127.0.0.1');
  await new Promise(resolve => server.once('listening', resolve));
  base = `http://127.0.0.1:${server.address().port}`;
  report = (await (await fetch(`${base}/api/data`)).json()).report;
  input = { household: report.sample_input, rate: 8, fixedCharge: 100 };
});
after(() => new Promise(resolve => server.close(resolve)));

test('bill uses monthly consumption and one fixed charge', () => {
  assert.deepEqual(calculateBill(100.25, 8, 100), { energyCharge: 802, fixedCharge: 100, total: 902 });
  assert.equal(calculateBill(0, 8, 100).total, 100);
});
test('validates household features, supported locations and tariffs', () => {
  assert.deepEqual(validateInput(input), input);
  for (const change of [{ Household_Size: 0 }, { Fan_Count: -1 }, { Rooms: 1.5 },
    { Refrigerator: 2 }, { Location: 'Unknown city' }, { TV_Count: null }]) {
    assert.throws(() => validateInput({ ...input, household: { ...input.household, ...change } }));
  }
  assert.throws(() => validateInput({ ...input, rate: -1 }));
  assert.throws(() => validateInput({ ...input, fixedCharge: '100' }));
});
test('serves the household form and Linear Regression report', async () => {
  const page = await fetch(base);
  assert.equal(page.status, 200);
  assert.match(await page.text(), /Your household/);
  assert.equal(report.statistics.usable_rows, 1000);
  assert.equal(report.split.training_rows, 800);
  assert.equal(report.split.test_rows, 200);
  assert.equal(report.selected_model, 'Linear Regression');
  assert.equal(report.optimization.candidates.length, 4);
  assert.deepEqual(report.models.map(model => model.name), ['Base Linear Regression', 'Tuned Linear Regression']);
  for (const model of report.models) {
    assert.ok(Math.abs(model.test.mse - model.test.rmse ** 2) < 0.01);
    for (const score of Object.values(model.high_usage_test)) assert.ok(score >= 0 && score <= 1);
  }
});
test('serves three Matplotlib evaluation plots as PNG images', async () => {
  for (const name of ['regression-comparison', 'high-usage-comparison', 'actual-vs-predicted']) {
    const response = await fetch(`${base}/plots/${name}.png`);
    assert.equal(response.status, 200);
    assert.match(response.headers.get('content-type'), /image\/png/);
    const bytes = Buffer.from(await response.arrayBuffer());
    assert.deepEqual([...bytes.subarray(0, 8)], [137, 80, 78, 71, 13, 10, 26, 10]);
  }
});
test('rejects missing household inputs with helpful JSON', async () => {
  const response = await fetch(`${base}/api/predict`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
  assert.equal(response.status, 400);
  assert.match((await response.json()).error, /Family members/);
});
test('Express calls the saved Linear Regression pipeline and calculates the bill', async () => {
  const response = await fetch(`${base}/api/predict`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(input) });
  assert.equal(response.status, 200);
  const result = await response.json();
  assert.ok(Number.isFinite(result.monthlyKwh) && result.monthlyKwh >= 0);
  assert.equal(result.bill.total, calculateBill(result.monthlyKwh, input.rate, input.fixedCharge).total);
});
