import { useEffect, useMemo, useState } from 'react'
import { LuPrinter, LuShieldCheck, LuTriangleAlert, LuIndianRupee, LuRuler, LuMicroscope, LuArrowLeft } from 'react-icons/lu'
import { getInspection, media } from '../api'
import type { Bucket, Inspection } from '../types'
import Advisor from '../components/Advisor'
import { Gauge, HBars, SizeHistogram } from '../components/Charts'

const BUCKETS: Bucket[] = ['A', 'URS', 'Reject']
const LABEL: Record<Bucket, string> = { A: 'Grade A', URS: 'URS', Reject: 'Reject' }
const DEFECT_LABEL: Record<string, string> = { good: 'Good', rotten: 'Rotten', black_mould: 'Black mould', sprouted: 'Sprouted', damaged: 'Damaged' }
const MODE_LABEL: Record<string, string> = {
  'ai': 'Finder + defect models + rules', 'finder-only': 'Finder model + rules (defects not checked)',
  'sheet+classifier': 'Sheet sizing + defect model + rules', 'fallback': 'Sheet sizing only (no AI models)',
}
function tone(d: string) { return d.includes('Grade A') ? 'good' : d.includes('URS') ? 'warn' : 'bad' }

export default function Result({ id }: { id: string }) {
  const [rec, setRec] = useState<Inspection | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [rateA, setRateA] = useState(2410)     // ₹/quintal, editable (buffer procurement price is set per season)
  const [rateU, setRateU] = useState(1800)
  const [qtl, setQtl] = useState(20)

  useEffect(() => { setRec(null); getInspection(id).then(setRec).catch(e => setError((e as Error).message)) }, [id])

  const stats = useMemo(() => {
    if (!rec) return null
    const on = rec.result.onions
    const sizes = on.map(o => o.diameter_mm).filter((x): x is number => !!x)
    const defects: Record<string, number> = {}
    on.forEach(o => { const k = o.defect_label || 'not checked'; defects[k] = (defects[k] || 0) + 1 })
    const s = rec.result.summary
    const spread = on.filter(o => o.defects.includes('rotten')).length * 2 + on.filter(o => o.defects.includes('black_mould') || o.defects.includes('sprouted')).length
    const score = Math.max(0, Math.min(100, Math.round(s.pct_by_count.A + 0.5 * s.pct_by_count.URS - (30 * spread) / Math.max(1, s.onion_count))))
    return { sizes, defects, score, mean: sizes.length ? sizes.reduce((a, b) => a + b, 0) / sizes.length : null }
  }, [rec])

  if (error) return <p className="alert bad"><LuTriangleAlert />{error}</p>
  if (!rec || !stats) return <div className="card">{[60, 90, 75].map((w, i) => <div key={i} className="skeleton" style={{ width: `${w}%` }} />)}</div>

  const { summary: s, onions, warnings, calibration, mode } = rec.result
  const pw = s.pct_by_weight.A != null ? s.pct_by_weight : s.pct_by_count
  const value = qtl * ((rateA * (pw.A ?? 0)) / 100 + (rateU * (pw.URS ?? 0)) / 100)
  const best = qtl * rateA
  const link = window.location.href

  return (
    <div className="grid" style={{ gap: 16 }}>
      <div className="page-head">
        <div>
          <a href="#/history" className="small no-print" style={{ textDecoration: 'none' }}><LuArrowLeft style={{ verticalAlign: '-2px' }} /> All reports</a>
          <h1>Quality report · {rec.lot_ref || rec.id}</h1>
          <div className="muted small">{new Date(rec.created_at).toLocaleString('en-IN')} · {rec.farmer || 'Farmer not entered'} · {rec.centre || 'Centre not entered'}</div>
        </div>
        <button className="no-print" onClick={() => window.print()}><LuPrinter /> Print / save PDF</button>
      </div>

      <div className={`decision ${tone(s.decision)}`}>
        <Gauge value={stats.score} size={96} />
        <div>
          <div className="small" style={{ opacity: .8, fontWeight: 600 }}>LOT DECISION</div>
          <div className="big">{s.decision}</div>
          <div className="sub">{s.onion_count} onions graded{stats.mean ? ` · average ${stats.mean.toFixed(1)} mm` : ''}{s.est_total_weight_g ? ` · sample ≈ ${(s.est_total_weight_g / 1000).toFixed(2)} kg` : ''}</div>
        </div>
      </div>

      <div className="grid g3">
        {BUCKETS.map(b => (
          <div key={b} className={`kpi ${b}`}>
            <div className="lbl">{LABEL[b]}</div>
            <div className="val">{s.pct_by_count[b]}%</div>
            <div className="sub">{s.count[b]} of {s.onion_count} onions{s.pct_by_weight[b] != null && <> · {s.pct_by_weight[b]}% by weight</>}</div>
            <div className="bar" style={{ width: `${s.pct_by_count[b]}%` }} />
          </div>
        ))}
      </div>

      {warnings.map(w => <p key={w} className="alert warn" style={{ margin: 0 }}><LuTriangleAlert />{w}</p>)}

      <div className="grid g2" style={{ alignItems: 'start' }}>
        <section className="card">
          <h2>Graded photo</h2>
          {rec.annotated ? <img className="graded" src={media(rec.annotated)} alt="Graded sample with each onion outlined" />
            : <p className="muted">Graded photo is not stored for older reports on this device.</p>}
          <p className="small muted" style={{ marginBottom: 0 }}>Green = Grade A · amber = URS · red = reject. Numbers match the table below.</p>
        </section>
        <div className="grid" style={{ gap: 16 }}>
          <section className="card">
            <h2><LuRuler style={{ verticalAlign: '-3px' }} /> Size distribution</h2>
            {stats.sizes.length ? <SizeHistogram sizes={stats.sizes} /> : <p className="muted small">Sizes not measured (no coin or sheet found).</p>}
          </section>
          <section className="card">
            <h2><LuMicroscope style={{ verticalAlign: '-3px' }} /> Condition</h2>
            <HBars rows={Object.entries(stats.defects).sort((a, b) => b[1] - a[1]).map(([k, v]) => ({
              label: DEFECT_LABEL[k] || k, value: v, color: k === 'good' ? '#2e7d32' : k === 'not checked' ? '#9aa1ad' : '#c62828',
            }))} />
          </section>
        </div>
      </div>

      <Advisor rec={rec} />

      <div className="grid g2" style={{ alignItems: 'start' }}>
        <section className="card">
          <h2><LuIndianRupee style={{ verticalAlign: '-3px' }} /> Lot value estimate</h2>
          <div className="grid g3" style={{ gap: 10 }}>
            <label>Lot size (quintal)<input type="number" min={0} value={qtl} onChange={e => setQtl(Number(e.target.value))} /></label>
            <label>Grade A ₹/qtl<input type="number" min={0} value={rateA} onChange={e => setRateA(Number(e.target.value))} /></label>
            <label>URS ₹/qtl<input type="number" min={0} value={rateU} onChange={e => setRateU(Number(e.target.value))} /></label>
          </div>
          <div className="row" style={{ marginTop: 12 }}>
            <div className="kpi" style={{ flex: 1 }}><div className="lbl">Estimated value as graded</div><div className="val" style={{ fontSize: 24 }}>₹{Math.round(value).toLocaleString('en-IN')}</div></div>
            <div className="kpi" style={{ flex: 1 }}><div className="lbl">If fully sorted to Grade A</div><div className="val" style={{ fontSize: 24, color: 'var(--good)' }}>₹{Math.round(best).toLocaleString('en-IN')}</div></div>
          </div>
          <p className="small muted" style={{ marginBottom: 0 }}>Example rates; enter this season's procurement rates. Uses the {s.pct_by_weight.A != null ? 'weight' : 'count'} share per grade; rejects valued at zero.</p>
        </section>
        <section className="card">
          <h2><LuShieldCheck style={{ verticalAlign: '-3px' }} /> Tamper-evident record</h2>
          <div className="row" style={{ alignItems: 'flex-start' }}>
            <img className="qr" alt="QR code linking to this report" src={`https://api.qrserver.com/v1/create-qr-code/?size=240x240&margin=0&data=${encodeURIComponent(link)}`} />
            <dl className="meta" style={{ flex: 1, minWidth: 200 }}>
              <dt>Report ID</dt><dd className="mono">{rec.id}</dd>
              <dt>Photo SHA-256</dt><dd className="mono">{rec.image_sha256.slice(0, 24)}…</dd>
              <dt>Result SHA-256</dt><dd className="mono">{rec.result_sha256.slice(0, 24)}…</dd>
              <dt>Size reference</dt><dd>{calibration.method === 'sheet' ? 'Calibration sheet' : calibration.method === 'coin' ? `Coin (${calibration.coin_mm} mm)` : 'None'}</dd>
              <dt>Grading</dt><dd>{MODE_LABEL[mode]} · rules {s.rules_version}</dd>
            </dl>
          </div>
          <p className="small muted" style={{ marginBottom: 0 }}>Any change to the photo or the grades changes these fingerprints, so a report can be verified in a dispute.</p>
        </section>
      </div>

      <section className="card">
        <h2>Every onion</h2>
        <div style={{ overflowX: 'auto' }}>
          <table className="onions">
            <thead><tr><th>#</th><th>Grade</th><th>Size (mm)</th><th>Weight (g)</th><th>Condition</th><th>Why</th></tr></thead>
            <tbody>
              {onions.map(o => (
                <tr key={o.id}>
                  <td>{o.id}</td>
                  <td><span className={`pill ${o.bucket}`}>{LABEL[o.bucket]}</span></td>
                  <td>{o.diameter_mm ?? '—'}</td>
                  <td>{o.weight_g ?? '—'}</td>
                  <td>{o.defect_label ? `${DEFECT_LABEL[o.defect_label] || o.defect_label} (${Math.round((o.defect_conf ?? 0) * 100)}%)` : (o.defects.length ? o.defects.join(', ') : 'not checked')}</td>
                  <td className="muted">{[...o.reasons, ...o.flags].join('; ') || '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  )
}
