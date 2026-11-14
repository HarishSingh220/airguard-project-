# AirGuard frontend

A React 18 + Vite dashboard for the AirGuard FastAPI service. The interface is organized into five pages:

- **Overview** — city-level AQI summary, headline readings and observed/forecast trends.
- **Forecast** — pollutant/target forecast selection, uncertainty range and horizon table.
- **Stations** — station-level forecast comparison for the five configured Delhi locations.
- **Model health** — artifact manifest, model training metadata and ingestion status.
- **Methodology** — source and feature pipeline explanation, pollutants and uncertainty notes.

## Local development

```powershell
Copy-Item .env.example .env.local
npm install
npm run dev
```

Vite opens at `http://localhost:5173`. During development, `src/api.js` defaults to `http://localhost:8000` if `VITE_API_BASE_URL` is not set. Run the backend in another terminal using the instructions in `../backend/README.md`.

## Production build

```powershell
npm run build
npm run preview
```

The Vite output is `dist/`. For Vercel, set the Root Directory to `frontend`, Build Command to `npm run build`, Output Directory to `dist`, and define `VITE_API_BASE_URL` as the deployed FastAPI origin. The variable is embedded at build time; redeploy after changing it. Production does not fall back to localhost.
