import { useEffect, useMemo, useState } from 'react'
import { LuCamera, LuCircleCheck, LuTriangleAlert, LuSparkles, LuScanSearch } from 'react-icons/lu'
import { allHistory, createInspection, getHealth, shrinkImage } from '../api'
import type { Health } from '../types'

const REFERENCES = [
  { label: '₹10 coin (27 mm)', coin_mm: 27 },
  { label: '₹5 coin (23 mm)', coin_mm: 23 },
  { label: '₹2 coin (25 mm)', coin_mm: 25 },
  { label: '₹1 coin (21.93 mm)', coin_mm: 21.93 },
  { label: 'Printed calibration sheet', coin_mm: 27 },
]

const SAMPLES = [
  { src: '/samples/lot-coin.jpg', name: 'Mixed sizes + coin' },
  { src: '/samples/lot-uniform.jpg', name: 'Uniform lot' },
  { src: '/samples/lot-small.jpg', name: 'Small onions' },
]

const STAGES = ['Uploading photo', 'Finding every onion', 'Measuring with the coin', 'Checking defects', 'Applying DoCA grade rules']

export default function Inspect() {
  const [health, setHealth] = useState<Health | null>(null)
  const [backendDown, setBackendDown] = useState(false)
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [farmer, setFarmer] = useState('')
  const [lotRef, setLotRef] = useState('')
  const [centre, setCentre] = useState(() => localStorageGet('oe_centre'))
  const [ref, setRef] = useState(0)
  const [busy, setBusy] = useState(false)
  const [stage, setStage] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const hist = useMemo(() => allHistory(), [])

  useEffect(() => { getHealth().then(setHealth).catch(() => setBackendDown(true)) }, [])
  useEffect(() => {
    if (!file) { setPreview(null); return }
    const url = URL.createObjectURL(file); setPreview(url)
    return () => URL.revokeObjectURL(url)
  }, [file])
  useEffect(() => {
    if (!busy) return
    setStage(0)
    const t = setInterval(() => setStage(s => Math.min(s + 1, STAGES.length - 1)), 900)
    return () => clearInterval(t)
  }, [busy])

  async function pickSample(src: string) {
    const b = await fetch(src).then(r => r.blob())
    setFile(new File([b], src.split('/').pop() || 'sample.jpg', { type: 'image/jpeg' }))
    setLotRef(l => l || `DEMO-${Math.floor(100 + Math.random() * 900)}`)
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault()
    if (!file) return
    setBusy(true); setError(null)
    try { localStorage.setItem('oe_centre', centre) } catch { /* ignore */ }
    const form = new FormData()
    form.append('image', await shrinkImage(file), 'photo.jpg')
    form.append('farmer', farmer); form.append('lot_ref', lotRef); form.append('centre', centre)
    form.append('coin_mm', String(REFERENCES[ref].coin_mm))
    try {
      const rec = await createInspection(form)
      window.location.hash = `#/r/${rec.id}`
    } catch (err) {
      setError((err as Error).message)
    } finally { setBusy(false) }
  }

  const graded = hist.reduce((s, r) => s + r.result.summary.onion_count, 0)
  const avgA = hist.length ? Math.round(hist.reduce((s, r) => s + r.result.summary.pct_by_count.A, 0) / hist.length) : null

  return (
    <>
      <section className="hero">
        <div className="eyebrow" style={{ color: '#f3b7cf' }}>AI quality inspection</div>
        <h1>Grade an onion lot from one photo</h1>
        <p>Spread a sample of 30–50 onions with a coin, take one photo from above. OnionEye finds every onion, measures it in mm,
          checks for rot, mould, sprouting and damage, applies the DoCA grade rules and gives a tamper-evident report with AI advice.</p>
        <div className="hero-stats">
          <div><b>{hist.length}</b><span>lots on this device</span></div>
          <div><b>{graded}</b><span>onions graded</span></div>
          <div><b>{avgA === null ? '—' : `${avgA}%`}</b><span>average Grade A</span></div>
          <div><b>&lt; 10 s</b><span>photo → report</span></div>
        </div>
      </section>

      {backendDown && <p className="alert bad"><LuTriangleAlert />Can't reach the grading server. Check your connection and reload.</p>}
      {health && !health.classifier_loaded && (
        <p className="alert warn"><LuTriangleAlert />Defect model not loaded: rot, mould, sprouting and damage are not checked right now.</p>
      )}

      <div className="steps">
        {['Take photo', 'Lot details', 'AI grading', 'Report + advice'].map((s, i) => (
          <span key={s} className={`step ${(i === 0 && !file) || (i === 1 && file && !busy) || (i === 2 && busy) ? 'on' : ''}`}><i>{i + 1}</i>{s}</span>
        ))}
      </div>

      <form onSubmit={submit} className="grid g2">
        <div className="card">
          <label className="drop">
            {preview ? <img src={preview} alt="Selected sample" /> : (
              <div className="cta"><LuCamera /><b style={{ color: 'var(--ink)' }}>Tap to take a photo</b><span className="small">or choose one from the gallery</span></div>
            )}
            <input type="file" accept="image/*" capture="environment" onChange={e => setFile(e.target.files?.[0] ?? null)} />
          </label>
          <div className="samples">
            <span className="small muted">No onions handy? Try a sample:</span>
            {SAMPLES.map(s => (
              <button type="button" key={s.src} title={s.name} onClick={() => pickSample(s.src)}><img src={s.src} alt={s.name} /></button>
            ))}
          </div>
        </div>

        <div className="card form">
          <h2 style={{ margin: 0 }}>Lot details</h2>
          <label>Farmer name<input value={farmer} onChange={e => setFarmer(e.target.value)} placeholder="e.g. Ramesh Patil" /></label>
          <div className="grid g2" style={{ gap: 12 }}>
            <label>Lot / token no.<input value={lotRef} onChange={e => setLotRef(e.target.value)} placeholder="e.g. LSG-0142" /></label>
            <label>Procurement centre<input value={centre} onChange={e => setCentre(e.target.value)} placeholder="e.g. Lasalgaon" /></label>
          </div>
          <label>Size reference in the photo
            <select value={ref} onChange={e => setRef(Number(e.target.value))}>
              {REFERENCES.map((r, i) => <option key={r.label} value={i}>{r.label}</option>)}
            </select>
          </label>
          <ul className="tips">
            <li><LuCircleCheck />Onions in one layer, not touching; coin flat on the same surface.</li>
            <li><LuCircleCheck />Shoot straight from above in daylight or even light, no flash glare.</li>
            <li><LuCircleCheck />Only onions in the frame; other vegetables are ignored.</li>
          </ul>
          {error && <p className="alert bad"><LuTriangleAlert />{error}</p>}
          {busy ? (
            <div className="progress">
              {STAGES.map((s, i) => <div key={s} className={i < stage ? 'done' : i === stage ? 'cur' : ''}><span className="dot" />{s}</div>)}
            </div>
          ) : (
            <button className="primary" disabled={!file}><LuScanSearch /> Grade this lot</button>
          )}
          <p className="small muted" style={{ margin: 0 }}><LuSparkles style={{ verticalAlign: '-2px' }} /> After grading, the AI advisor explains the result and suggests what to do with the lot.</p>
        </div>
      </form>
    </>
  )
}

function localStorageGet(k: string) { try { return localStorage.getItem(k) || '' } catch { return '' } }
