import React, { useEffect, useState } from "react";
import { api } from "./api.js";
import { featureName, money, shortId, STATUS_NAMES, when } from "./labels.js";
import LinkMap from "./LinkMap.jsx";

export default function CaseFile({ id, token, onOpen, onChanged, notify }) {
  const [item, setItem] = useState(null);
  const [error, setError] = useState(null);
  const [note, setNote] = useState("");

  useEffect(() => {
    let live = true;
    setItem(null);
    setNote("");
    setError(null);
    api(`/api/cases/${id}`, { token })
      .then((x) => live && setItem(x))
      .catch((e) => live && setError(e.message));
    return () => { live = false; };
  }, [id, token]);

  if (error) return <section className="case empty"><p>{error}</p></section>;
  if (!item) return <section className="case empty"><p>Loading case {id}</p></section>;

  async function act(path, opts, done) {
    try {
      setItem(await api(path, { token, ...opts }));
      notify(done);
      onChanged();
    } catch (e) {
      notify(e.message);
    }
  }
  const setStatus = (status, done) => act(`/api/cases/${id}`, { method: "PATCH", body: { status } }, done);
  const label = (value, done) =>
    act(`/api/cases/${id}/feedback`, { method: "POST", body: { label: value, note: note || null } }, done);

  const maxImpact = Math.max(...item.reasons.map((r) => r.impact), 0.01);

  return (
    <section className="case" aria-label={`Case ${item.id}`}>
      <header className="case-head">
        <div>
          <h2>{money(item.amount)}</h2>
          <p className="sub">
            Case {item.id}, payment {item.transaction_id}, {when(item.event_ts)} UTC
          </p>
        </div>
        <div className="verdict">
          <strong className={`pill ${item.decision.toLowerCase()}`}>
            {item.decision === "BLOCK" ? "Would block" : "Needs review"}
          </strong>
          <span>{item.label ? (item.label === "fraud" ? "Confirmed fraud" : "Cleared as legitimate") : STATUS_NAMES[item.status]}</span>
        </div>
      </header>

      <div className="actions">
        <button className="primary danger" onClick={() => label("fraud", "Marked as fraud")}>Confirm fraud</button>
        <button className="primary" onClick={() => label("legit", "Marked as legitimate")}>Mark legitimate</button>
        {item.status === "open" && <button onClick={() => setStatus("investigating", "Moved to investigating")}>Start investigating</button>}
        {item.status !== "escalated" && item.status !== "closed" && <button onClick={() => setStatus("escalated", "Escalated")}>Escalate</button>}
        {item.status === "closed" && <button onClick={() => setStatus("open", "Case reopened")}>Reopen</button>}
        <input aria-label="Note for this decision" placeholder="Add a note to the decision" value={note}
          onChange={(e) => setNote(e.target.value)} />
      </div>

      <div className="case-grid">
        <div>
          <h3>Why it was flagged</h3>
          {item.reasons.length === 0 && <p className="sub">No single signal stands out. The connection score raised this alert.</p>}
          <ol className="reasons">
            {item.reasons.map((r) => (
              <li key={r.feature}>
                <span>{featureName(r.feature)}</span>
                <span className="value">{r.value ?? "missing"}</span>
                <span className="meter"><i style={{ width: `${(r.impact / maxImpact) * 100}%` }} /></span>
              </li>
            ))}
          </ol>

          <h3>Score</h3>
          <dl className="facts">
            <dt>Combined fraud probability</dt><dd>{(item.score * 100).toFixed(1)}%</dd>
            <dt>Payment pattern model</dt><dd>{(item.xgb_score * 100).toFixed(1)}%</dd>
            <dt>Connection model</dt><dd>{(item.gnn_score * 100).toFixed(1)}%</dd>
            <dt>Model version</dt><dd>{item.model_version}</dd>
          </dl>

          <h3>Identifiers</h3>
          <dl className="facts ids">
            <dt>Account</dt><dd>{item.user_id}</dd>
            <dt>Card</dt><dd>{item.card_id}</dd>
            <dt>Device</dt><dd>{item.device_id === "unknown" ? "Not captured" : item.device_id}</dd>
            <dt>Location</dt><dd>{item.location_id === "unknown" ? "Not captured" : item.location_id}</dd>
          </dl>
        </div>

        <div>
          <h3>Connected alerts: {item.linked.length}</h3>
          {item.linked.length === 0
            ? <p className="sub">No other alert shares this account, card or device.</p>
            : <LinkMap item={item} onOpen={onOpen} />}
          {item.linked.length > 0 && (
            <table className="linked">
              <thead><tr><th>Case</th><th>Shared</th><th className="num">Amount</th><th className="num">Score</th></tr></thead>
              <tbody>
                {item.linked.slice(0, 8).map((l) => (
                  <tr key={l.id}>
                    <td><button className="link" onClick={() => onOpen(l.id)}>{l.id}</button></td>
                    <td>{l.shared.join(", ")}</td>
                    <td className="num">{money(l.amount)}</td>
                    <td className="num">{(l.score * 100).toFixed(0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          <h3>History</h3>
          <ol className="history">
            {item.history.map((h, i) => (
              <li key={i}><b>{h.actor}</b> {h.action === "created" ? "raised this alert" : h.action}{h.detail ? `: ${shortId(h.detail)}` : ""}</li>
            ))}
          </ol>
        </div>
      </div>
    </section>
  );
}
