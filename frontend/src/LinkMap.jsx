import React from "react";

// Radial map of one case. Centre: this payment. Middle ring: its account, card and device.
// Outer ring: other alerts sharing that account, card or device. A ring shows up as a crowded spoke.
const SIZE = 460;
const C = SIZE / 2;
const INNER = 92;
const OUTER = 186;
const ANGLES = { user: -90, card: 30, device: 150 };
const NAMES = { user: "Account", card: "Card", device: "Device" };
const rad = (d) => (d * Math.PI) / 180;
const pt = (angle, r) => [C + r * Math.cos(rad(angle)), C + r * Math.sin(rad(angle))];

function tone(c) {
  if (c.label === "fraud") return "confirmed";
  if (c.label === "legit") return "cleared";
  return c.decision === "BLOCK" ? "block" : "review";
}

export default function LinkMap({ item, onOpen }) {
  const groups = { user: [], card: [], device: [] };
  for (const l of item.linked) groups[l.shared[0]]?.push(l);
  const spokes = Object.keys(groups).filter((k) => k !== "device" || item.device_id !== "unknown");

  return (
    <figure className="linkmap">
      <svg viewBox={`0 0 ${SIZE} ${SIZE}`} role="img"
        aria-label={`This payment links to ${item.linked.length} other alerts through shared accounts, cards or devices`}>
        {spokes.map((k) => {
          const [x, y] = pt(ANGLES[k], INNER);
          const list = groups[k];
          const spread = Math.min(100, Math.max(0, (list.length - 1) * 12));
          return (
            <g key={k}>
              <line x1={C} y1={C} x2={x} y2={y} className="edge strong" />
              {list.map((l, i) => {
                const a = ANGLES[k] + (list.length > 1 ? -spread / 2 + (spread * i) / (list.length - 1) : 0);
                const r = OUTER - (i % 2) * 22;
                const [nx, ny] = pt(a, r);
                return (
                  <g key={l.id}>
                    <line x1={x} y1={y} x2={nx} y2={ny} className="edge" />
                    <circle cx={nx} cy={ny} r="9" tabIndex="0" role="button" className={`node ${tone(l)}`}
                      aria-label={`Open case ${l.id}, score ${(l.score * 100).toFixed(0)}`}
                      onClick={() => onOpen(l.id)}
                      onKeyDown={(e) => e.key === "Enter" && onOpen(l.id)}>
                      <title>{`Case ${l.id}: score ${(l.score * 100).toFixed(0)}`}</title>
                    </circle>
                  </g>
                );
              })}
              <circle cx={x} cy={y} r="26" className="hub" />
              <text x={x} y={y + 4} className="hub-label" textAnchor="middle">{NAMES[k]}</text>
            </g>
          );
        })}
        <circle cx={C} cy={C} r="34" className={`centre ${tone(item)}`} />
        <text x={C} y={C + 7} className="centre-label" textAnchor="middle">{(item.score * 100).toFixed(0)}</text>
      </svg>
      <figcaption>
        <span><i className="dot block" /> Would block</span>
        <span><i className="dot review" /> Needs review</span>
        <span><i className="dot confirmed" /> Confirmed fraud</span>
        <span><i className="dot cleared" /> Cleared</span>
      </figcaption>
    </figure>
  );
}
