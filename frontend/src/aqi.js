// Mirrors backend/app/config.py's CPCB_BINS / CPCB_LABELS / CPCB_CATEGORY_COLORS,
// so a station's own AQI forecast (which the API doesn't pre-categorize, only the
// city-wide one does) can still get a category + color on the client.
const BINS = [0, 50, 100, 200, 300, 400, 500];
const LABELS = ["Good", "Satisfactory", "Moderate", "Poor", "Very poor", "Severe"];
const COLORS = {
  Good: "#4CAF7D",
  Satisfactory: "#A8C84C",
  Moderate: "#F2C94C",
  Poor: "#F2994A",
  "Very poor": "#EB5757",
  Severe: "#8B2E3C",
};

export function aqiCategory(value) {
  if (value == null || Number.isNaN(value)) return null;
  for (let i = 0; i < BINS.length - 1; i++) {
    if (value <= BINS[i + 1]) return LABELS[i];
  }
  return LABELS[LABELS.length - 1];
}

export function aqiColor(value) {
  const cat = aqiCategory(value);
  return cat ? COLORS[cat] : "#9AA5B1";
}

export { COLORS as AQI_CATEGORY_COLORS };
