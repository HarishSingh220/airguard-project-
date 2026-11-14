import { useCallback, useEffect, useMemo, useState } from "react";
import { api, ApiError } from "./api.js";
import { aqiCategory, aqiColor } from "./aqi.js";

const TARGETS_FALLBACK = [
  { name: "AQI", label: "Air quality index", unit: "index" },
  { name: "PM25", label: "PM2.5", unit: "µg/m³" },
  { name: "O3", label: "Ozone · daily mean", unit: "µg/m³" },
  { name: "O3_8H", label: "Ozone · 8-hour max", unit: "µg/m³" },
  { name: "NOx", label: "Nitrogen oxides", unit: "µg/m³" },
];

const NAV = [
  { id: "overview", label: "Overview", icon: "grid" },
  { id: "forecast", label: "Forecast", icon: "activity" },
  { id: "stations", label: "Stations", icon: "map" },
  { id: "models", label: "Model health", icon: "layers" },
  { id: "methodology", label: "Methodology", icon: "book" },
];

function Icon({ name, size = 18 }) {
  const common = { width: size, height: size, viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: 1.8, strokeLinecap: "round", strokeLinejoin: "round", "aria-hidden": true };
  const paths = {
    grid: <><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/></>,
    activity: <><path d="M3 12h4l3-8 4 16 3-8h4"/><path d="M3 21h18" opacity=".25"/></>,
    map: <><path d="m3 6 6-3 6 3 6-3v15l-6 3-6-3-6 3z"/><path d="M9 3v15M15 6v15"/></>,
    layers: <><path d="m12 3 9 5-9 5-9-5 9-5Z"/><path d="m3 12 9 5 9-5M3 16l9 5 9-5"/></>,
    book: <><path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H20v17H6.5A2.5 2.5 0 0 1 4 17.5z"/><path d="M4 17.5A2.5 2.5 0 0 1 6.5 15H20M8 7h8M8 10h7"/></>,
    chevron: <path d="m9 18 6-6-6-6"/>,
    refresh: <><path d="M20 7v5h-5"/><path d="M4 17v-5h5"/><path d="M5.6 9a7 7 0 0 1 11.6-2L20 12M4 12l2.8 5a7 7 0 0 0 11.6-2"/></>,
    wind: <><path d="M3 8h12a3 3 0 1 0-3-3"/><path d="M2 12h17a2.5 2.5 0 1 1-2.5 2.5"/><path d="M4 16h7a2 2 0 1 1-2 2"/></>,
    pin: <><path d="M20 10c0 5-8 11-8 11S4 15 4 10a8 8 0 1 1 16 0Z"/><circle cx="12" cy="10" r="2.5"/></>,
    check: <path d="m5 12 4 4L19 6"/>,
    alert: <><path d="M10.3 3.9 2.5 17.4A1.8 1.8 0 0 0 4.1 20h15.8a1.8 1.8 0 0 0 1.6-2.6L13.7 3.9a2 2 0 0 0-3.4 0Z"/><path d="M12 9v4m0 3h.01"/></>,
    spark: <><path d="m12 3 1.4 5.6L19 10l-5.6 1.4L12 17l-1.4-5.6L5 10l5.6-1.4L12 3Z"/><path d="m19 16 .6 2.4L22 19l-2.4.6L19 22l-.6-2.4L16 19l2.4-.6L19 16Z"/></>,
    arrow: <><path d="M5 12h14"/><path d="m13 6 6 6-6 6"/></>,
    external: <><path d="M14 4h6v6M20 4l-9 9"/><path d="M18 13v6a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h6"/></>,
  };
  return <svg {...common}>{paths[name] || paths.spark}</svg>;
}

function formatDate(value, options = { day: "numeric", month: "short" }) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleDateString("en-IN", options);
}

function formatNumber(value, digits = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toLocaleString("en-IN", { maximumFractionDigits: digits, minimumFractionDigits: digits }) : "—";
}

function statusLabel(status) {
  return ({ ok: "System healthy", demo: "Demo data", degraded: "Needs attention", unverified: "Unverified", unreachable: "API offline" })[status] || "Checking system";
}

function StatusPill({ status = "unknown", children }) {
  const tone = status === "ok" ? "good" : status === "demo" ? "demo" : status === "degraded" || status === "unverified" || status === "unreachable" ? "warn" : "neutral";
  return <span className={`status-pill status-${tone}`}><span className="status-dot" />{children || statusLabel(status)}</span>;
}

function Sidebar({ page, setPage, health }) {
  return <aside className="sidebar">
    <div className="brand-lockup">
      <div className="brand-symbol"><Icon name="wind" size={22}/></div>
      <div><div className="brand-name">airguard<span>.</span></div><div className="brand-caption">ATMOSPHERIC INTELLIGENCE</div></div>
    </div>
    <div className="nav-caption">WORKSPACE</div>
    <nav className="side-nav" aria-label="Main navigation">
      {NAV.map((item) => <button key={item.id} className={`nav-item ${page === item.id ? "active" : ""}`} onClick={() => setPage(item.id)}>
        <Icon name={item.icon}/><span>{item.label}</span>{page === item.id && <span className="nav-active-mark"/>}
      </button>)}
    </nav>
    <div className="sidebar-bottom">
      <div className="sidebar-system"><span className="eyebrow">SYSTEM STATUS</span><StatusPill status={health?.status || "unknown"}/><p>{health?.forecast_origin_date ? `Forecast origin · ${formatDate(health.forecast_origin_date, { day: "numeric", month: "short", year: "numeric" })}` : "Waiting for API health check"}</p></div>
      <div className="sidebar-foot"><div className="mini-mark">AG</div><span>Delhi NCR · Air intelligence</span></div>
    </div>
  </aside>;
}

function Topbar({ page, refresh, refreshing }) {
  const title = NAV.find((item) => item.id === page)?.label || "Overview";
  return <header className="topbar">
    <div className="topbar-path"><span>AirGuard</span><Icon name="chevron" size={14}/><strong>{title}</strong></div>
    <div className="topbar-actions"><span className="topbar-date">{new Date().toLocaleDateString("en-IN", { weekday: "short", day: "numeric", month: "short", year: "numeric" })}</span><button className="icon-button" onClick={refresh} disabled={refreshing} aria-label="Refresh dashboard"><Icon name="refresh" size={17}/></button><div className="avatar">AG</div></div>
  </header>;
}

function PageHeading({ eyebrow, title, description, action }) {
  return <div className="page-heading"><div><div className="eyebrow">{eyebrow}</div><h1>{title}</h1><p>{description}</p></div>{action}</div>;
}

function AqiGauge({ value, category }) {
  const color = aqiColor(Number(value));
  const pct = Math.max(0, Math.min(100, Number(value || 0) / 500 * 100));
  return <div className="aqi-gauge" style={{ "--gauge-color": color, "--gauge-progress": `${pct}%` }}>
    <div className="gauge-ring"><div className="gauge-inner"><span className="gauge-value">{formatNumber(value)}</span><span className="gauge-label">AQI</span></div></div>
    <div className="gauge-category" style={{ color }}>{category || aqiCategory(Number(value)) || "Awaiting data"}</div>
  </div>;
}

function StatCard({ label, value, unit, note, icon = "activity", accent = "blue" }) {
  return <article className={`stat-card accent-${accent}`}><div className="stat-top"><span>{label}</span><span className="stat-icon"><Icon name={icon}/></span></div><div className="stat-value">{value}<small>{unit}</small></div><div className="stat-note">{note}</div></article>;
}

function TrendChart({ history = [], forecast = [], unit = "" }) {
  const hist = history.filter((point) => point.value != null).map((point) => ({ t: new Date(point.date).getTime(), y: Number(point.value) })).filter((point) => Number.isFinite(point.t) && Number.isFinite(point.y));
  const fc = [...forecast].sort((a, b) => Number(a.horizon) - Number(b.horizon)).map((point) => ({ t: new Date(point.forecast_date).getTime(), y: Number(point.prediction), low: Number(point.lower), high: Number(point.upper) })).filter((point) => Number.isFinite(point.t) && Number.isFinite(point.y));
  const all = [...hist, ...fc];
  if (!all.length) return <div className="empty-state"><div className="empty-icon"><Icon name="activity" size={22}/></div><strong>No trend data available</strong><span>Forecast and history files will appear here after the pipeline completes.</span></div>;
  const width = 760; const height = 260; const pad = { left: 46, right: 18, top: 20, bottom: 36 };
  const minT = Math.min(...all.map((p) => p.t)); const maxT = Math.max(...all.map((p) => p.t));
  const values = [...all.map((p) => p.y), ...fc.flatMap((p) => [p.low, p.high]).filter(Number.isFinite)];
  const minY = Math.max(0, Math.min(...values) - Math.max(8, (Math.max(...values) - Math.min(...values)) * .14));
  const maxY = Math.max(...values) + Math.max(8, (Math.max(...values) - Math.min(...values)) * .14);
  const x = (value) => pad.left + (maxT === minT ? .5 : (value - minT) / (maxT - minT)) * (width - pad.left - pad.right);
  const y = (value) => height - pad.bottom - (value - minY) / (maxY - minY || 1) * (height - pad.top - pad.bottom);
  const line = (points) => points.map((p) => `${x(p.t)},${y(p.y)}`).join(" ");
  const band = [...fc.map((p) => `${x(p.t)},${y(p.high)}`), ...fc.slice().reverse().map((p) => `${x(p.t)},${y(p.low)}`)].join(" ");
  const ticks = Array.from({ length: 5 }, (_, i) => minY + (maxY - minY) * i / 4);
  return <div className="chart-wrap"><div className="chart-legend"><span><i className="legend-line observed"/>Observed</span><span><i className="legend-line predicted"/>Forecast</span><span><i className="legend-band"/>Prediction range</span><span className="chart-unit">{unit}</span></div><svg className="trend-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Observed history and ten day forecast chart">
    {ticks.map((tick) => <g key={tick}><line x1={pad.left} x2={width - pad.right} y1={y(tick)} y2={y(tick)} className="chart-grid"/><text x={pad.left - 10} y={y(tick) + 4} textAnchor="end" className="chart-axis-label">{Math.round(tick)}</text></g>)}
    {fc.length > 1 && <polygon points={band} className="chart-confidence"/>}
    {hist.length > 1 && <polyline points={line(hist)} className="chart-observed"/>}
    {fc.length > 1 && <polyline points={line(fc)} className="chart-predicted"/>}
    {fc.map((point) => <circle key={point.t} cx={x(point.t)} cy={y(point.y)} r="3.2" className="chart-point"/>)}
    <text x={pad.left} y={height - 8} className="chart-axis-label">{formatDate(minT)}</text><text x={width - pad.right} y={height - 8} textAnchor="end" className="chart-axis-label">{formatDate(maxT)}</text>
  </svg></div>;
}

function ForecastTable({ rows, unit }) {
  if (!rows.length) return <div className="empty-inline">No forecast rows to display.</div>;
  return <div className="table-scroll"><table className="data-table"><thead><tr><th>Day</th><th>Forecast date</th><th>Predicted</th><th>Expected range</th><th>Confidence</th></tr></thead><tbody>{rows.map((row) => {
    const width = Math.max(0, Number(row.upper) - Number(row.lower));
    const confidence = width < 25 ? "Narrow" : width < 60 ? "Moderate" : "Wide";
    return <tr key={`${row.target || "city"}-${row.horizon}`}><td><span className="horizon-chip">D+{row.horizon}</span></td><td>{formatDate(row.forecast_date, { weekday: "short", day: "numeric", month: "short" })}</td><td className="table-strong">{formatNumber(row.prediction, 1)} <small>{unit}</small></td><td>{formatNumber(row.lower, 1)} – {formatNumber(row.upper, 1)}</td><td><span className={`range-pill range-${confidence.toLowerCase()}`}>{confidence}</span></td></tr>;
  })}</tbody></table></div>;
}

function OverviewPage({ data, selections, setTarget, loading, error, setPage }) {
  const { target, targets, forecast, history, health, stationSnapshot, modelInfo } = data;
  const targetInfo = targets.find((item) => item.name === target) || TARGETS_FALLBACK[0];
  const tomorrow = forecast.find((row) => Number(row.horizon) === 1);
  const aqiForecast = target === "AQI" ? forecast : data.cityAqiForecast;
  const aqiTomorrow = aqiForecast.find((row) => Number(row.horizon) === 1);
  const pmTomorrow = data.pmForecast.find((row) => Number(row.horizon) === 1);
  const stationCount = stationSnapshot.length;
  const avgStationAqi = stationSnapshot.length ? stationSnapshot.filter((row) => row.target === "AQI").reduce((sum, row) => sum + Number(row.prediction || 0), 0) / Math.max(1, stationSnapshot.filter((row) => row.target === "AQI").length) : null;
  return <>
    <PageHeading eyebrow="DELHI NCR · AIR QUALITY INTELLIGENCE" title="A clearer view of the air." description="Understand current conditions, explore the next ten days, and see how the atmosphere is expected to evolve." action={<StatusPill status={health?.status || "unknown"}/>}/>
    {error && <div className="notice notice-warning"><Icon name="alert"/><span>{error}</span></div>}
    <section className="overview-hero">
      <div className="hero-copy"><span className="hero-kicker"><span className="hero-kicker-dot"/> CITY OUTLOOK · TOMORROW</span><h2>Know what’s<br/><em>coming next.</em></h2><p>Forecast-led air quality intelligence across five Delhi monitoring stations, powered by weather, atmospheric and ground observations.</p><button className="button button-light" onClick={() => setPage("forecast")}>Explore forecast <Icon name="arrow" size={16}/></button></div>
      <div className="hero-reading"><div className="hero-reading-label">DELHI CITY · AQI</div><AqiGauge value={aqiTomorrow?.prediction} category={aqiTomorrow ? aqiCategory(Number(aqiTomorrow.prediction)) : null}/><div className="hero-reading-foot"><span>Tomorrow’s outlook</span><strong>{aqiTomorrow ? formatDate(aqiTomorrow.forecast_date, { day: "numeric", month: "short" }) : "No forecast"}</strong></div></div>
      <div className="hero-decor hero-decor-one"/><div className="hero-decor hero-decor-two"/>
    </section>
    <div className="section-row"><div><div className="eyebrow">AT A GLANCE</div><h2 className="section-title">City signals</h2></div><span className="muted-note">Forecast-led · next 24 hours</span></div>
    <div className="stats-grid">
      <StatCard label="Air quality index" value={formatNumber(aqiTomorrow?.prediction)} unit="AQI" note={aqiTomorrow ? aqiCategory(Number(aqiTomorrow.prediction)) : "Forecast unavailable"} icon="wind" accent="green"/>
      <StatCard label="Fine particles" value={formatNumber(pmTomorrow?.prediction, 1)} unit="µg/m³" note="PM2.5 · tomorrow" icon="activity" accent="amber"/>
      <StatCard label="Station coverage" value={stationCount || "—"} unit="stations" note="Monitoring locations" icon="pin" accent="blue"/>
      <StatCard label="Average station AQI" value={formatNumber(avgStationAqi, 0)} unit="AQI" note="Tomorrow · available stations" icon="layers" accent="violet"/>
    </div>
    <section className="panel chart-panel"><div className="panel-heading"><div><div className="eyebrow">AIR QUALITY TREND</div><h2>History meets forecast</h2><p>Observed conditions transition into the model’s ten-day outlook.</p></div><div className="target-select">{targets.map((item) => <button key={item.name} className={target === item.name ? "selected" : ""} onClick={() => setTarget(item.name)}>{item.name === "O3_8H" ? "O₃ · 8h" : item.name === "PM25" ? "PM₂.₅" : item.name}</button>)}</div></div>{loading ? <ChartSkeleton/> : <TrendChart history={target === "AQI" ? history : data.overviewHistory} forecast={forecast} unit={targetInfo.unit}/>}</section>
    <div className="bottom-grid"><section className="panel"><div className="panel-heading compact"><div><div className="eyebrow">NEXT TEN DAYS</div><h2>Forecast timeline</h2></div><button className="text-link" onClick={() => setPage("forecast")}>Full forecast <Icon name="chevron" size={14}/></button></div><ForecastTable rows={forecast.slice(0, 5)} unit={targetInfo.unit}/></section>
      <section className="panel insight-panel"><div className="insight-icon"><Icon name="spark" size={20}/></div><div className="eyebrow">MODEL NOTE</div><h2>Built for the days ahead.</h2><p>AirGuard blends tree-based and linear models across multiple forecast horizons. Predictions are accompanied by an estimated range, not a single-point promise.</p><div className="insight-meta"><span>Last training</span><strong>{modelInfo?.trained_at ? formatDate(modelInfo.trained_at, { day: "numeric", month: "short", year: "numeric" }) : "Not reported"}</strong></div><button className="text-link" onClick={() => setPage("models")}>Explore model health <Icon name="chevron" size={14}/></button></section></div>
  </>;
}

function ChartSkeleton() { return <div className="chart-skeleton"><div/><div/><div/></div>; }

function ForecastPage({ data, selections, setSelections, loading, error }) {
  const { targets, stations, forecast, history } = data;
  const targetInfo = targets.find((item) => item.name === selections.target) || TARGETS_FALLBACK[0];
  return <>
    <PageHeading eyebrow="PREDICTIVE OUTLOOK" title="Forecast explorer" description="Move between pollutants and monitoring locations. Compare recent observations with the model’s ten-day outlook."/>
    <section className="panel filter-panel"><div className="filter-group"><label>MEASURE</label><div className="filter-options">{targets.map((item) => <button key={item.name} className={selections.target === item.name ? "selected" : ""} onClick={() => setSelections((old) => ({ ...old, target: item.name }))}>{item.name === "O3_8H" ? "O₃ · 8h" : item.name === "PM25" ? "PM₂.₅" : item.name}</button>)}</div></div><div className="filter-divider"/><div className="filter-group"><label>LOCATION</label><select value={selections.station} onChange={(event) => setSelections((old) => ({ ...old, station: event.target.value }))}><option value="city">Delhi city-wide</option>{stations.map((item) => <option key={item.name} value={item.name}>{item.name}</option>)}</select></div></section>
    {error && <div className="notice notice-warning"><Icon name="alert"/><span>{error}</span></div>}
    <div className="forecast-summary"><div><span className="eyebrow">{selections.station === "city" ? "DELHI CITY" : selections.station.toUpperCase()} · {targetInfo.label.toUpperCase()}</span><h2>10-day outlook</h2><p>From {formatDate(forecast[0]?.forecast_date, { day: "numeric", month: "long" })} to {formatDate(forecast.at(-1)?.forecast_date, { day: "numeric", month: "long" })}</p></div><div className="summary-value"><strong>{formatNumber(forecast[0]?.prediction, 1)}</strong><span>{targetInfo.unit} · tomorrow</span></div></div>
    <section className="panel chart-panel"><div className="panel-heading"><div><div className="eyebrow">OBSERVED + PREDICTED</div><h2>Trend and uncertainty</h2><p>Shaded region represents the model’s estimated prediction range.</p></div></div>{loading ? <ChartSkeleton/> : <TrendChart history={history} forecast={forecast} unit={targetInfo.unit}/>}</section>
    <section className="panel"><div className="panel-heading compact"><div><div className="eyebrow">DAILY DETAIL</div><h2>Forecast by horizon</h2></div><span className="muted-note">10 forecast steps</span></div><ForecastTable rows={forecast} unit={targetInfo.unit}/></section>
  </>;
}

function StationsPage({ data, loading, setSelections, setPage }) {
  const aqiRows = data.stationSnapshot.filter((row) => row.target === "AQI");
  const sorted = [...aqiRows].sort((a, b) => Number(b.prediction) - Number(a.prediction));
  return <>
    <PageHeading eyebrow="MONITORING NETWORK" title="A city of local conditions." description="Explore tomorrow’s AQI outlook across the five monitoring locations used by AirGuard."/>
    <section className="station-intro"><div><span className="eyebrow">NETWORK COVERAGE</span><strong>{data.stations.length || aqiRows.length || 0} <small>monitoring locations</small></strong><p>Station-level predictions can differ from the city average. Use a location to open its complete forecast.</p></div><div className="station-pins"><span/><span/><span/><span/><span/></div></section>
    {loading && <ChartSkeleton/>}
    {!loading && <div className="station-grid">{data.stations.map((station) => {
      const row = aqiRows.find((item) => item.station === station.name);
      const rank = sorted.findIndex((item) => item.station === station.name) + 1;
      const color = row ? aqiColor(Number(row.prediction)) : "#9aa4a0";
      return <button className="station-card" key={station.name} onClick={() => { setSelections((old) => ({ ...old, station: station.name, target: "AQI" })); setPage("forecast"); }}><div className="station-card-top"><div className="station-pin"><Icon name="pin" size={17}/></div><span className="station-rank">NETWORK {String(rank || "—").padStart(2, "0")}</span></div><h3>{station.name}</h3><div className="station-coords">{Number(station.lat).toFixed(3)}° N · {Number(station.lon).toFixed(3)}° E</div><div className="station-aqi-row"><div><strong>{formatNumber(row?.prediction)}</strong><span>AQI tomorrow</span></div><span className="station-category" style={{ color }}>{row ? aqiCategory(Number(row.prediction)) : "No data"}</span></div><div className="station-card-foot"><span>{row ? `Range ${formatNumber(row.lower)}–${formatNumber(row.upper)}` : "Forecast unavailable"}</span><Icon name="arrow" size={16}/></div></button>;
    })}</div>}
    {!data.stations.length && !loading && <div className="empty-state"><strong>Station metadata unavailable</strong><span>The stations endpoint did not return monitoring locations.</span></div>}
  </>;
}

function ModelsPage({ data }) {
  const modelTargets = Object.keys(data.modelInfo?.targets || {});
  const runJobs = data.ingestionStatus || [];
  return <>
    <PageHeading eyebrow="MODEL OPERATIONS" title="Know what powers the forecast." description="A transparent view of the production model set, retraining record, and data-pipeline health." action={<StatusPill status={data.health?.status || "unknown"}/>}/>
    <div className="model-summary-grid"><article className="model-summary-card"><span className="eyebrow">PRODUCTION MODEL FAMILY</span><strong>Ensemble regression</strong><p>Random Forest · LightGBM · XGBoost · Ridge</p></article><article className="model-summary-card"><span className="eyebrow">FORECAST HORIZON</span><strong>10 days</strong><p>One separately fitted pipeline per active member and horizon.</p></article><article className="model-summary-card"><span className="eyebrow">PARAMETER POLICY</span><strong>Fixed production JSON</strong><p>Retraining reuses committed parameters and weights; it does not tune them.</p></article></div>
    <section className="panel"><div className="panel-heading compact"><div><div className="eyebrow">TRAINING MANIFEST</div><h2>Latest production fit</h2></div><StatusPill status={data.health?.model_artifacts_ready ? "ok" : "degraded"}>{data.health?.model_artifacts_ready ? "Model files detected" : "Model files incomplete"}</StatusPill></div><div className="manifest-grid"><div className="manifest-date"><span className="eyebrow">TRAINED AT</span><strong>{data.modelInfo?.trained_at ? formatDate(data.modelInfo.trained_at, { day: "numeric", month: "long", year: "numeric" }) : "Not available"}</strong><span>Values come from outputs/models/trained_at.json.</span></div>{(modelTargets.length ? modelTargets : ["AQI", "PM25", "O3", "O3_8H", "NOx"]).map((target) => <div className="manifest-target" key={target}><span>{target}</span><strong>{data.modelInfo?.targets?.[target]?.trained_up_to ? formatDate(data.modelInfo.targets[target].trained_up_to, { day: "numeric", month: "short", year: "numeric" }) : "—"}</strong><small>{(data.modelInfo?.targets?.[target]?.members || []).length || 0} members</small></div>)}</div></section>
    <section className="panel"><div className="panel-heading compact"><div><div className="eyebrow">DATA PIPELINE</div><h2>Latest ingestion activity</h2><p>Reported by the ingestion log when the API has database access.</p></div></div>{runJobs.length ? <div className="table-scroll"><table className="data-table"><thead><tr><th>Job</th><th>Last started</th><th>Status</th><th>Rows</th><th>Message</th></tr></thead><tbody>{runJobs.map((job) => <tr key={job.job}><td className="table-strong">{job.job}</td><td>{formatDate(job.started_at, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" })}</td><td><StatusPill status={job.status === "ok" ? "ok" : "degraded"}>{job.status || "unknown"}</StatusPill></td><td>{job.rows_written ?? "—"}</td><td className="job-message">{job.message || "—"}</td></tr>)}</tbody></table></div> : <div className="empty-inline">Ingestion status is not available. Configure DATABASE_URL on the API deployment to enable this view.</div>}</section>
    <div className="callout-note"><Icon name="alert"/><div><strong>Health is not the same as freshness.</strong><p>The API checks file structure, forecast origin age, feature schema and model artifact presence. Connector availability for CPCB, CAMS, Open-Meteo and FIRMS is recorded separately by ingestion jobs.</p></div></div>
  </>;
}

function MethodologyPage() {
  const steps = [
    ["01", "Collect", "Ground observations from CPCB stations are combined with atmospheric and weather context from supported data providers."],
    ["02", "Prepare", "Daily features align stations and dates, handle gaps, and construct target-specific labels and forecast covariates."],
    ["03", "Train", "A fixed ensemble of Random Forest, LightGBM, XGBoost and Ridge is refitted on the latest complete training window."],
    ["04", "Forecast", "Each model member predicts ten horizons. Stored ensemble weights combine the members, with residual bands describing uncertainty."],
  ];
  return <>
    <PageHeading eyebrow="HOW TO READ AIRGUARD" title="Evidence in. Insight out." description="AirGuard is a forecast system, not a live sensor feed. This page explains how observations become a practical ten-day outlook."/>
    <section className="method-hero"><div className="method-orbit orbit-a"/><div className="method-orbit orbit-b"/><div className="method-hero-content"><div className="eyebrow">THE AIRGUARD PIPELINE</div><h2>From fragmented signals<br/>to a shared picture.</h2><p>Five monitoring locations. Five air-quality targets. Ten forecast horizons. One consistent workflow.</p></div><div className="method-hero-mark"><Icon name="wind" size={52}/><span>AG / 10D</span></div></section>
    <div className="method-steps">{steps.map(([number, title, description]) => <article className="method-step" key={number}><div className="method-number">{number}</div><div><h3>{title}</h3><p>{description}</p></div></article>)}</div>
    <div className="bottom-grid"><section className="panel"><div className="eyebrow">POLLUTANTS COVERED</div><h2>Five views of air quality</h2><div className="pollutant-list"><div><strong>AQI</strong><span>Composite indicator</span></div><div><strong>PM₂.₅</strong><span>Fine particulate matter</span></div><div><strong>O₃</strong><span>Daily mean ozone</span></div><div><strong>O₃ · 8h</strong><span>Maximum 8-hour ozone</span></div><div><strong>NOx</strong><span>Nitrogen oxides</span></div></div></section><section className="panel"><div className="eyebrow">INTERPRETING THE RANGE</div><h2>Forecasts carry uncertainty.</h2><p className="method-copy">The predicted value is the ensemble estimate. The lower and upper bounds are derived from historical out-of-fold residuals. A wider interval means the model’s historical errors were larger for that horizon.</p><p className="method-copy">The forecast is not a regulatory reading or a health diagnosis. For immediate exposure decisions, consult official monitoring and public-health guidance.</p></section></div>
  </>;
}

function Footer({ health }) {
  return <footer className="app-footer"><span>© {new Date().getFullYear()} AirGuard · Delhi NCR atmospheric intelligence</span><span><span className={`footer-dot ${health?.status === "ok" ? "green" : "amber"}`}/>{health?.status === "ok" ? "Data pipeline healthy" : health?.status === "demo" ? "Demo dataset" : "Pipeline status requires attention"}</span></footer>;
}

const initialData = { health: null, stations: [], targets: TARGETS_FALLBACK, forecast: [], history: [], modelInfo: null, weights: null, stationSnapshot: [], cityAqiForecast: [], pmForecast: [], overviewHistory: [], ingestionStatus: [] };

export default function App() {
  const [page, setPage] = useState("overview");
  const [selections, setSelections] = useState({ station: "city", target: "AQI" });
  const [data, setData] = useState(initialData);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState("");
  const [lastUpdated, setLastUpdated] = useState(null);

  const loadData = useCallback(async ({ quiet = false, signal } = {}) => {
    if (quiet) setRefreshing(true); else setLoading(true);
    setError("");
    const safe = async (promise, fallback) => { try { return await promise; } catch (err) { if (err.name === "AbortError") throw err; return fallback; } };
    try {
      const [health, stations, targets, modelInfo, stationSnapshot, cityAqiForecast, pmForecast, ingestionStatus] = await Promise.all([
        safe(api.health({ signal: signal }), { status: "unreachable" }),
        safe(api.stations({ signal: signal }), []),
        safe(api.targets({ signal: signal }), TARGETS_FALLBACK),
        safe(api.modelInfo({ signal: signal }), null),
        safe(api.stationSnapshot("AQI", { signal: signal }), []),
        safe(api.cityForecast("AQI", { signal: signal }), []),
        safe(api.cityForecast("PM25", { signal: signal }), []),
        safe(api.ingestionStatus({ signal: signal }), []),
      ]);
      const target = selections.target;
      const forecastPromise = selections.station === "city" ? api.cityForecast(target, { signal: signal }) : api.stationForecast(selections.station, target, { signal: signal });
      const historyPromise = selections.station === "city" ? api.cityHistory(target, 30, { signal: signal }) : api.stationHistory(selections.station, target, 30, { signal: signal });
      const [forecast, history, weights] = await Promise.all([
        safe(forecastPromise, []), safe(historyPromise, []), safe(api.modelWeights(target, { signal: signal }), null),
      ]);
      const overviewHistory = target === "AQI" ? history : await safe(api.cityHistory("AQI", 30, { signal: signal }), []);
      setData({ health, stations, targets: targets?.length ? targets : TARGETS_FALLBACK, modelInfo, stationSnapshot, cityAqiForecast, pmForecast, ingestionStatus, forecast, history, weights, overviewHistory });
      setLastUpdated(new Date());
      if (!forecast.length && health.status === "unreachable") setError("The API is not reachable. Start the backend service and check VITE_API_BASE_URL.");
      else if (!forecast.length) setError("Forecast data is not available yet. Run the feature and forecast pipeline, then refresh this view.");
    } catch (err) {
      if (err.name !== "AbortError") setError(err.message || "Unable to refresh AirGuard data.");
    } finally {
      if (!signal?.aborted) {
        setLoading(false);
        setRefreshing(false);
      }
    }
  }, [selections.station, selections.target]);

  useEffect(() => {
    const controller = new AbortController();
    loadData({ signal: controller.signal });
    return () => controller.abort();
  }, [loadData]);

  const targetOptions = data.targets?.length ? data.targets : TARGETS_FALLBACK;
  const refresh = () => loadData({ quiet: true });
  const currentTitle = useMemo(() => NAV.find((item) => item.id === page)?.label || "Overview", [page]);

  return <div className="shell">
    <Sidebar page={page} setPage={setPage} health={data.health}/>
    <div className="workspace"><Topbar page={page} refresh={refresh} refreshing={refreshing}/><main className="page-content">
      {page === "overview" && <OverviewPage data={{ ...data, targets: targetOptions }} selections={selections} setTarget={(target) => setSelections((old) => ({ ...old, target }))} loading={loading} error={error} setPage={setPage}/>}
      {page === "forecast" && <ForecastPage data={{ ...data, targets: targetOptions, stations: data.stations }} selections={selections} setSelections={setSelections} loading={loading} error={error}/>}
      {page === "stations" && <StationsPage data={data} loading={loading} setSelections={setSelections} setPage={setPage}/>}
      {page === "models" && <ModelsPage data={data}/>}
      {page === "methodology" && <MethodologyPage/>}
      <Footer health={data.health}/>
    </main><div className="mobile-nav" aria-label="Mobile navigation">{NAV.map((item) => <button key={item.id} className={page === item.id ? "active" : ""} onClick={() => setPage(item.id)}><Icon name={item.icon} size={17}/><span>{item.label}</span></button>)}</div></div>
    <div className="sr-only" aria-live="polite">{loading ? `Loading ${currentTitle}` : lastUpdated ? `Updated ${lastUpdated.toLocaleTimeString()}` : ""}</div>
  </div>;
}
