import { useMemo } from 'react'
import { LuLayers, LuCircleCheck, LuTriangleAlert, LuScale, LuMapPin, LuCamera } from 'react-icons/lu'
import { allHistory } from '../api'
import { HBars, LotTrend } from '../components/Charts'

export default function Dashboard() {
  const rows = useMemo(() => allHistory().slice().reverse(), [])   // oldest first

  if (!rows.length) return (
    <div className="card empty">
      <LuCamera />
      <h2 style={{ marginTop: 10 }}>No lots graded yet</h2>
      <p>Grade a lot and the centre dashboard fills in: grade mix per lot, defects, and centre-wise quality.</p>
      <a href="#/"><button className="primary">Grade the first lot</button></a>
    </div>
  )

  const onions = rows.reduce((s, r) => s + r.result.summary.onion_count, 0)
  const avg = (k: 'A' | 'URS' | 'Reject') => rows.reduce((s, r) => s + r.result.summary.pct_by_count[k], 0) / rows.length
  const decisions: Record<string, number> = {}
  rows.forEach(r => { decisions[r.result.summary.decision] = (decisions[r.result.summary.decision] || 0) + 1 })
  const defects: Record<string, number> = {}
  rows.forEach(r => r.result.onions.forEach(o => o.defects.forEach(d => { defects[d] = (defects[d] || 0) + 1 })))
  const centres: Record<string, { lots: number; a: number }> = {}
  rows.forEach(r => { const c = r.centre || 'Unspecified'; centres[c] = centres[c] || { lots: 0, a: 0 }; centres[c].lots++; centres[c].a += r.result.summary.pct_by_count.A })

  const tiles = [
    { icon: LuLayers, lbl: 'Lots inspected', val: rows.length },
    { icon: LuScale, lbl: 'Onions graded', val: onions.toLocaleString('en-IN') },
    { icon: LuCircleCheck, lbl: 'Average Grade A', val: `${avg('A').toFixed(1)}%` },
    { icon: LuTriangleAlert, lbl: 'Average reject', val: `${avg('Reject').toFixed(1)}%` },
  ]

  return (
    <>
      <div className="page-head">
        <div><div className="eyebrow">Centre dashboard</div><h1>Quality at a glance</h1>
          <div className="muted small">From the reports stored on this device. With the central server this becomes the DoCA-wide view.</div></div>
      </div>
      <div className="grid g4" style={{ marginBottom: 16 }}>
        {tiles.map(t => <div key={t.lbl} className="stat-tile"><div className="lbl"><t.icon />{t.lbl}</div><div className="val">{t.val}</div></div>)}
      </div>
      <div className="grid g2" style={{ alignItems: 'start' }}>
        <section className="card">
          <h2>Grade mix per lot</h2>
          <LotTrend lots={rows.slice(-16).map(r => ({ label: r.lot_ref || r.id, ...r.result.summary.pct_by_count }))} />
          <div className="row small muted"><span className="pill A">Grade A</span><span className="pill URS">URS</span><span className="pill Reject">Reject</span> last {Math.min(16, rows.length)} lots</div>
        </section>
        <section className="card">
          <h2>Lot decisions</h2>
          <HBars rows={Object.entries(decisions).map(([k, v]) => ({
            label: k.replace('Accept as ', ''), value: v, color: k.includes('Grade A') ? '#2e7d32' : k.includes('URS') ? '#b26a00' : '#c62828',
          }))} />
          <h2 style={{ marginTop: 18 }}>Defects found</h2>
          {Object.keys(defects).length ? <HBars rows={Object.entries(defects).sort((a, b) => b[1] - a[1]).map(([k, v]) => ({ label: k.replace('_', ' '), value: v, color: '#8e244d' }))} />
            : <p className="small muted">No defects recorded yet.</p>}
        </section>
      </div>
      <section className="card" style={{ marginTop: 16 }}>
        <h2><LuMapPin style={{ verticalAlign: '-3px' }} /> Centres</h2>
        <table className="onions">
          <thead><tr><th>Centre</th><th>Lots</th><th>Avg Grade A</th></tr></thead>
          <tbody>{Object.entries(centres).map(([c, v]) => <tr key={c}><td>{c}</td><td>{v.lots}</td><td>{(v.a / v.lots).toFixed(1)}%</td></tr>)}</tbody>
        </table>
      </section>
    </>
  )
}
