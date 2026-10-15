// Illustrative flat tariff. Actual electricity-provider tariffs may differ.
export function calculateBill(kwh, rate, fixedCharge) {
  const energyCharge = Math.round(kwh * rate * 100) / 100;
  return { energyCharge, fixedCharge, total: Math.round((energyCharge + fixedCharge) * 100) / 100 };
}
