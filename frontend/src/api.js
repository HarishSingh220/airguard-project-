const configuredBaseUrl = String(import.meta.env.VITE_API_BASE_URL || "").trim();
// Use localhost only while running Vite's development server. A production
// build must be configured with the deployed FastAPI origin; otherwise it
// would incorrectly try to call localhost on every visitor's own device.
const BASE_URL = (
  configuredBaseUrl || (import.meta.env.DEV ? "http://localhost:8000" : "")
).replace(/\/$/, "");

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function get(path, { signal } = {}) {
  if (!BASE_URL) {
    throw new ApiError(
      "The frontend API URL is not configured. Set VITE_API_BASE_URL to the deployed AirGuard FastAPI URL and rebuild the frontend.",
      0,
    );
  }

  let response;
  try {
    response = await fetch(`${BASE_URL}${path}`, {
      signal,
      headers: { Accept: "application/json" },
    });
  } catch (error) {
    if (error.name === "AbortError") throw error;
    throw new ApiError(`Could not connect to the AirGuard API at ${BASE_URL}.`, 0);
  }

  if (!response.ok) {
    let detail = response.statusText || `Request failed (${response.status})`;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch {
      // The API returned a non-JSON error response.
    }
    throw new ApiError(detail, response.status);
  }
  return response.json();
}

export const api = {
  health: (options) => get("/api/health", options),
  stations: (options) => get("/api/stations", options),
  targets: (options) => get("/api/targets", options),
  cityForecast: (target, options) => get(`/api/forecast/city?target=${encodeURIComponent(target)}`, options),
  stationForecast: (station, target, options) =>
    get(`/api/forecast/station/${encodeURIComponent(station)}?target=${encodeURIComponent(target)}`, options),
  stationSnapshot: (target, options) =>
    get(`/api/forecast/stations/latest?target=${encodeURIComponent(target)}`, options),
  cityHistory: (target, days = 30, options) =>
    get(`/api/history/city?target=${encodeURIComponent(target)}&days=${days}`, options),
  stationHistory: (station, target, days = 30, options) =>
    get(`/api/history/station/${encodeURIComponent(station)}?target=${encodeURIComponent(target)}&days=${days}`, options),
  modelWeights: (target, options) => get(`/api/model/weights?target=${encodeURIComponent(target)}`, options),
  modelInfo: (options) => get("/api/model/info", options),
  ingestionStatus: (options) => get("/api/ingestion/status/latest-per-job", options),
};
