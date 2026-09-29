import { useEffect, useState } from 'react'
import { LuSearch, LuFileText } from 'react-icons/lu'
import { listInspections } from '../api'
import type { Inspection } from '../types'

export default function History() {
  const [rows, setRows] = useState<Inspection[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [q, setQ] = useState('')

  useEffect(() => { listInspections().then(setRows).catch(e => setError((e as Error).message)) }, [])

  if (error) return <p className="alert bad">{error}</p>
  if (!rows) return <div className="card"><div className="skeleton" /><div className="skeleton" /></div>
  const shown = rows.filter(r => `${r.farmer} ${r.lot_ref} ${r.centre} ${r.id}`.toLowerCase().includes(q.toLowerCase()))

  return (
    <>
      <div className="page-head">
        <div><div className="eyebrow">Reports</div><h1>Inspection reports</h1></div>
        <label style={{ minWidth: 260 }}><span className="row" style={{ gap: 6 }}><LuSearch /> Search farmer, lot, centre or ID</span>
          <input value={q} onChange={e => setQ(e.target.value)} placeholder="e.g. LSG-0142" /></label>
      </div>
      {!rows.length ? (
        <div className="card empty"><LuFileText /><p>No reports yet. <a href="#/">Grade your first lot</a>.</p></div>
      ) : (
        <section className="card" style={{ overflowX: 'auto' }}>
          <table className="onions">
            <thead><tr><th>When</th><th>Farmer</th><th>Lot</th><th>Centre</th><th>Onions</th><th>Grade A</th><th>Decision</th></tr></thead>
            <tbody>
              {shown.map(r => (
                <tr key={r.id} className="clickable" onClick={() => { window.location.hash = `#/r/${r.id}` }}>
                  <td>{new Date(r.created_at).toLocaleString('en-IN', { dateStyle: 'medium', timeStyle: 'short' })}</td>
                  <td>{r.farmer || '—'}</td><td>{r.lot_ref || '—'}</td><td>{r.centre || '—'}</td>
                  <td>{r.result.summary.onion_count}</td>
                  <td><span className={`pill ${r.result.summary.pct_by_count.A >= 90 ? 'A' : r.result.summary.pct_by_count.A >= 50 ? 'URS' : 'Reject'}`}>{r.result.summary.pct_by_count.A}%</span></td>
                  <td>{r.result.summary.decision}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
    </>
  )
}
