import React, { useCallback, useEffect, useState } from "react";
import { api, loadSession, saveSession } from "./api.js";
import { money, STATUS_NAMES, when } from "./labels.js";
import CaseFile from "./CaseFile.jsx";
import Overview from "./Overview.jsx";

function Login({ onLogin }) {
  const [form, setForm] = useState({ username: "", password: "" });
  const [error, setError] = useState(null);
  async function submit(e) {
    e.preventDefault();
    try {
      onLogin(await api("/api/login", { method: "POST", body: form }));
    } catch (err) {
      setError(err.message);
    }
  }
  return (
    <main className="login">
      <form onSubmit={submit}>
        <h1>Fraud case desk</h1>
        <label>Username<input autoFocus value={form.username} onChange={(e) => setForm({ ...form, username: e.target.value })} /></label>
        <label>Password<input type="password" value={form.password} onChange={(e) => setForm({ ...form, password: e.target.value })} /></label>
        {error && <p className="error" role="alert">{error}</p>}
        <button className="primary">Sign in</button>
      </form>
    </main>
  );
}

const FILTERS = [["open", "Open"], ["investigating", "Investigating"], ["escalated", "Escalated"], ["closed", "Closed"]];

export default function App() {
  const [session, setSession] = useState(loadSession);
  const [view, setView] = useState("queue");
  const [status, setStatus] = useState("open");
  const [q, setQ] = useState("");
  const [cases, setCases] = useState({ total: 0, items: [] });
  const [summary, setSummary] = useState(null);
  const [selected, setSelected] = useState(null);
  const [toast, setToast] = useState(null);
  const [offline, setOffline] = useState(false);
  const token = session?.token;

  const signOut = useCallback(() => { saveSession(null); setSession(null); }, []);

  const refresh = useCallback(async () => {
    if (!token) return;
    try {
      const params = new URLSearchParams({ status, limit: "100", ...(q ? { q } : {}) });
      const [list, sum] = await Promise.all([api(`/api/cases?${params}`, { token }), api("/api/summary", { token })]);
      setCases(list);
      setSummary(sum);
      setOffline(false);
    } catch (e) {
      if (e.status === 401) signOut();
      else setOffline(true);
    }
  }, [token, status, q, signOut]);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 4000);
    return () => clearInterval(t);
  }, [refresh]);

  useEffect(() => {
    if (!toast) return undefined;
    const t = setTimeout(() => setToast(null), 2800);
    return () => clearTimeout(t);
  }, [toast]);

  if (!session) return <Login onLogin={(s) => { saveSession(s); setSession(s); }} />;

  return (
    <div className="shell">
      <header className="top">
        <h1>Fraud case desk</h1>
        <nav aria-label="Views">
          <button className={view === "queue" ? "tab on" : "tab"} onClick={() => setView("queue")}>Alert queue</button>
          <button className={view === "overview" ? "tab on" : "tab"} onClick={() => setView("overview")}>Overview</button>
        </nav>
        <div className="who">
          {offline && <span role="status">Can't reach the server. Retrying</span>}
          <span>{session.user}</span>
          <button className="link" onClick={signOut}>Sign out</button>
        </div>
      </header>

      {view === "overview" ? <Overview summary={summary} token={token} /> : (
        <main className={selected ? "desk has-case" : "desk"}>
          <section className="queue" aria-label="Alert queue">
            <div className="filters">
              {FILTERS.map(([key, name]) => (
                <button key={key} className={status === key ? "chip on" : "chip"} onClick={() => setStatus(key)}>
                  {name} <span>{summary?.by_status[key]?.toLocaleString() ?? ""}</span>
                </button>
              ))}
              <input type="search" placeholder="Search payment, account, card or device" aria-label="Search alerts"
                value={q} onChange={(e) => setQ(e.target.value)} />
            </div>
            {cases.items.length === 0 ? (
              <p className="sub pad">{q ? "No alert matches this search." : `No ${STATUS_NAMES[status].toLowerCase()} alerts. New alerts appear here as traffic is scored.`}</p>
            ) : (
              <ul className="rows">
                {cases.items.map((c) => (
                  <li key={c.id}>
                    <button className={selected === c.id ? "row on" : "row"} onClick={() => setSelected(c.id)}>
                      <span className={`score ${c.decision.toLowerCase()}`}>{(c.score * 100).toFixed(0)}</span>
                      <span className="what"><b>{money(c.amount)}</b><small>{c.user_id}</small></span>
                      <span className="time">{when(c.event_ts)}</span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </section>
          {selected
            ? <div className="case-wrap">
                <button className="link back" onClick={() => setSelected(null)}>Back to queue</button>
                <CaseFile id={selected} token={token} onOpen={setSelected} onChanged={refresh} notify={setToast} />
              </div>
            : <section className="case empty"><p>Pick an alert to open its case file.</p></section>}
        </main>
      )}
      {toast && <div className="toast" role="status">{toast}</div>}
    </div>
  );
}
