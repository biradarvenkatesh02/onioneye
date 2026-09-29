// Small dependency-free SVG charts.

export function Gauge({ value, size = 112, label = 'Quality score' }: { value: number; size?: number; label?: string }) {
  const v = Math.max(0, Math.min(100, value))
  const r = 44, c = 2 * Math.PI * r, arc = c * 0.75
  const color = v >= 80 ? '#2e7d32' : v >= 55 ? '#b26a00' : '#c62828'
  return (
    <svg className="gauge" width={size} height={size} viewBox="0 0 110 110" role="img" aria-label={`${label} ${v} of 100`}>
      <circle cx="55" cy="55" r={r} fill="none" stroke="rgba(255,255,255,.25)" strokeWidth="10"
        strokeDasharray={`${arc} ${c}`} transform="rotate(135 55 55)" strokeLinecap="round" />
      <circle cx="55" cy="55" r={r} fill="none" stroke={color === '#2e7d32' ? '#a5e0a9' : color === '#b26a00' ? '#ffd48a' : '#ffb3b3'}
        strokeWidth="10" strokeDasharray={`${(arc * v) / 100} ${c}`} transform="rotate(135 55 55)" strokeLinecap="round" />
      <text x="55" y="58" textAnchor="middle" fontSize="28" fontWeight="800" fill="#fff">{Math.round(v)}</text>
      <text x="55" y="76" textAnchor="middle" fontSize="9.5" fill="rgba(255,255,255,.85)">{label}</text>
    </svg>
  )
}

/** Size histogram with the Grade A / URS bands shaded. */
export function SizeHistogram({ sizes, aMin = 45, aMax = 65, uMin = 35, uMax = 70 }:
  { sizes: number[]; aMin?: number; aMax?: number; uMin?: number; uMax?: number }) {
  const W = 520, H = 200, P = { l: 34, r: 10, t: 26, b: 30 }
  const lo = 20, hi = 90, bw = 5
  const bins = Array.from({ length: (hi - lo) / bw }, (_, i) => lo + i * bw)
  const counts = bins.map(b => sizes.filter(s => s >= b && s < b + bw).length)
  const max = Math.max(1, ...counts)
  const x = (mm: number) => P.l + ((mm - lo) / (hi - lo)) * (W - P.l - P.r)
  const y = (n: number) => H - P.b - (n / max) * (H - P.t - P.b)
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Onion size distribution">
      <rect x={x(uMin)} y={P.t} width={x(uMax) - x(uMin)} height={H - P.t - P.b} fill="#fdf1dc" />
      <text x={x(uMin) + 3} y={P.t - 8} fontSize="10.5" fill="#b26a00">URS</text>
      <rect x={x(aMin)} y={P.t} width={x(aMax) - x(aMin)} height={H - P.t - P.b} fill="#e6f2e7" />
      <text x={(x(aMin) + x(aMax)) / 2} y={P.t - 8} textAnchor="middle" fontSize="11" fill="#2e7d32" fontWeight="700">Grade A {aMin}–{aMax} mm</text>
      {bins.map((b, i) => counts[i] > 0 && (
        <rect key={b} x={x(b) + 1} y={y(counts[i])} width={x(b + bw) - x(b) - 2} height={H - P.b - y(counts[i])} rx="3"
          fill={b >= aMin && b + bw <= aMax ? '#2e7d32' : b >= uMin && b + bw <= uMax ? '#b26a00' : '#c62828'}>
          <title>{`${b}–${b + bw} mm: ${counts[i]}`}</title>
        </rect>
      ))}
      <line x1={P.l} x2={W - P.r} y1={H - P.b} y2={H - P.b} stroke="#cfc9cc" />
      {[20, 30, 40, 50, 60, 70, 80, 90].map(t => (
        <text key={t} x={x(t)} y={H - 10} textAnchor="middle" fontSize="11" fill="#6a6770">{t}</text>
      ))}
      <text x={W - P.r} y={H - 10} textAnchor="end" fontSize="11" fill="#6a6770" dx="-2" dy="-14">mm</text>
      {[0, Math.ceil(max / 2), max].map(t => (
        <text key={t} x={P.l - 6} y={y(t) + 4} textAnchor="end" fontSize="11" fill="#6a6770">{t}</text>
      ))}
    </svg>
  )
}

export function HBars({ rows }: { rows: { label: string; value: number; color: string; note?: string }[] }) {
  const max = Math.max(1, ...rows.map(r => r.value))
  return (
    <div>
      {rows.map(r => (
        <div className="hbar" key={r.label}>
          <span>{r.label}</span>
          <div className="track"><div className="fill" style={{ width: `${(100 * r.value) / max}%`, background: r.color }} /></div>
          <b style={{ textAlign: 'right' }}>{r.note ?? r.value}</b>
        </div>
      ))}
    </div>
  )
}

/** Stacked A / URS / Reject bar per lot, newest right. */
export function LotTrend({ lots }: { lots: { label: string; A: number; URS: number; Reject: number }[] }) {
  const W = 560, H = 180, P = { l: 40, r: 8, t: 8, b: 24 }
  const n = Math.max(1, lots.length)
  const bw = Math.min(34, (W - P.l - P.r) / n - 6)
  const y = (p: number) => P.t + ((100 - p) / 100) * (H - P.t - P.b)
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Grade mix per lot">
      {[0, 50, 100].map(t => (
        <g key={t}><line x1={P.l} x2={W - P.r} y1={y(t)} y2={y(t)} stroke="#eee" />
          <text x={P.l - 6} y={y(t) + 4} textAnchor="end" fontSize="10.5" fill="#6a6770">{t}%</text></g>
      ))}
      {lots.map((l, i) => {
        const x = P.l + 6 + i * ((W - P.l - P.r) / n)
        let acc = 0
        return (
          <g key={i}>
            {([['A', '#2e7d32'], ['URS', '#b26a00'], ['Reject', '#c62828']] as const).map(([k, c]) => {
              const v = l[k]; const top = y(acc + v); const h = y(acc) - top; acc += v
              return h > 0 ? <rect key={k} x={x} y={top} width={bw} height={h} fill={c}><title>{`${l.label} · ${k} ${v}%`}</title></rect> : null
            })}
            <text x={x + bw / 2} y={H - 8} textAnchor="middle" fontSize="10" fill="#6a6770">{i + 1}</text>
          </g>
        )
      })}
    </svg>
  )
}
