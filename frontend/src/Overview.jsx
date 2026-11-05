import React, { useState } from "react";
import { STATUS_NAMES } from "./labels.js";

function AlertsChart({ days }) {
  const [hover, setHover] = useState(null);
  const [asTable, setAsTable] = useState(false);
  const W = 720, H = 220, L = 38, B = 26, T = 10;
  const max = Math.max(...days.map((d) => d.REVIEW + d.BLOCK), 1);
  const step = (W - L) / days.length;
  const bar = Math.max(3, Math.min(22, step - 2));
  const y = (v) => ((H - B - T) * v) / max;
  const ticks = [0, Math.round(max / 2), max];

  return (
    <figure className="chart">
      <div className="legend">
        <span><i className="dot review" /> Needs review</span>
        <span><i className="dot block" /> Would block</span>
        <button className="link" onClick={() => setAsTable(!asTable)}>{asTable ? "Show chart" : "Show as table"}</button>
      </div>
      {asTable ? (
        <table className="linked">
          <thead><tr><th>Day</th><th className="num">Needs review</th><th className="num">Would block</th></tr></thead>
          <tbody>{days.map((d) => <tr key={d.day}><td>{d.day}</td><td className="num">{d.REVIEW}</td><td className="num">{d.BLOCK}</td></tr>)}</tbody>
        </table>
      ) : (
        <div className="plot">
          <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Alerts per day, split by needs review and would block"
            onMouseLeave={() => setHover(null)}>
            {ticks.map((t) => (
              <g key={t}>
                <line x1={L} x2={W} y1={H - B - y(t)} y2={H - B - y(t)} className="grid" />
                <text x={L - 8} y={H - B - y(t) + 4} textAnchor="end" className="tick">{t}</text>
              </g>
            ))}
            {days.map((d, i) => {
              const x = L + i * step + (step - bar) / 2;
              const r = y(d.REVIEW), b = y(d.BLOCK);
              return (
                <g key={d.day} onMouseEnter={() => setHover({ d, x: x + bar / 2 })}>
                  <rect x={L + i * step} y={T} width={step} height={H - B - T} fill="transparent" />
                  <rect x={x} y={H - B - r} width={bar} height={r} className="mark review" />
                  {b > 0 && <rect x={x} y={H - B - r - b - (r ? 2 : 0)} width={bar} height={b} rx="2" className="mark block" />}
                </g>
              );
            })}
            {days.length > 0 && <text x={L} y={H - 6} className="tick">{days[0].day}</text>}
            {days.length > 1 && <text x={W} y={H - 6} className="tick" textAnchor="end">{days[days.length - 1].day}</text>}
          </svg>
          {hover && (
            <div className="tip" style={{ left: `${(hover.x / W) * 100}%` }}>
              <b>{hover.d.day}</b><br />{hover.d.REVIEW} need review<br />{hover.d.BLOCK} would block
            </div>
          )}
        </div>
      )}
    </figure>
  );
}

export default function Overview({ summary, token, stats }) {
  if (!summary) return <section className="overview"><p className="sub">Loading</p></section>;
  const drift = summary.drift || {};
  const total = Object.values(summary.by_status).reduce((a, b) => a + b, 0);
  return (
    <section className="overview">
      <h2>Desk overview</h2>
      <div className="numbers">
        <p><b>{total.toLocaleString()}</b> alerts received</p>
        <p><b>{(summary.by_status.open + summary.by_status.investigating + summary.by_status.escalated).toLocaleString()}</b> still need a decision</p>
        <p><b>{summary.precision === null ? "None yet" : `${(summary.precision * 100).toFixed(0)}%`}</b> of {summary.reviewed} reviewed alerts were fraud</p>
      </div>

      <h3>Alerts per transaction day</h3>
      {summary.per_day.length ? <AlertsChart days={summary.per_day} /> : <p className="sub">Alerts appear here once traffic is flowing.</p>}

      <div className="two">
        <div>
          <h3>Case status</h3>
          <dl className="facts">
            {Object.entries(summary.by_status).map(([k, v]) => (<React.Fragment key={k}><dt>{STATUS_NAMES[k]}</dt><dd>{v.toLocaleString()}</dd></React.Fragment>))}
          </dl>
        </div>
        <div>
          <h3>Data drift</h3>
          {drift.status === "waiting" || !drift.status ? <p className="sub">{drift.message}</p> : (
            <>
              <p className={`drift ${drift.status}`}>
                {drift.status === "drift" ? "Drift detected" : "Traffic matches training data"}
              </p>
              <dl className="facts">
                <dt>Columns drifted</dt><dd>{(drift.drift_share * 100).toFixed(0)}%</dd>
                <dt>Retraining threshold</dt><dd>{(drift.threshold * 100).toFixed(0)}%</dd>
                <dt>Rows checked</dt><dd>{drift.rows?.toLocaleString()}</dd>
              </dl>
              <a href={`/api/drift/report?token=${encodeURIComponent(token)}`} target="_blank" rel="noreferrer">Open the full drift report</a>
            </>
          )}
        </div>
      </div>
    </section>
  );
}
