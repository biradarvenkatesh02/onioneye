import { useEffect, useState } from 'react'
import { LuSparkles, LuRefreshCw, LuMessageCircle, LuCopy, LuWarehouse, LuStore, LuSprout, LuTriangleAlert } from 'react-icons/lu'
import { getAdvice, LANGUAGES } from '../api'
import type { Advice, Inspection } from '../types'

export default function Advisor({ rec }: { rec: Inspection }) {
  const [lang, setLang] = useState(() => { try { return localStorage.getItem('oe_lang') || 'hi' } catch { return 'hi' } })
  const [adv, setAdv] = useState<Advice | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [copied, setCopied] = useState(false)

  function load(l: string) {
    setLoading(true); setErr(null)
    getAdvice(rec, l).then(setAdv).catch(e => setErr((e as Error).message)).finally(() => setLoading(false))
  }
  useEffect(() => { load(lang) }, [rec.id]) // eslint-disable-line react-hooks/exhaustive-deps

  function pickLang(l: string) {
    setLang(l); try { localStorage.setItem('oe_lang', l) } catch { /* ignore */ }
    load(l)
  }

  const shareText = adv ? `${adv.farmer_message}\n\n${window.location.href}` : ''
  return (
    <section className="card advisor">
      <div className="adv-head">
        <div className="adv-icon"><LuSparkles /></div>
        <div>
          <h2 style={{ margin: 0 }}>AI quality advisor</h2>
          <div className="small muted">
            {adv ? (adv.source === 'genai'
              ? <><span className="pill ai">GenAI · {adv.model}</span> grounded on the measured facts of this lot</>
              : <><span className="pill rules">Offline rulebook</span> standard onion post-harvest practice</>) : 'Reading the lot…'}
          </div>
        </div>
        <span className="spacer" />
        <select value={lang} onChange={e => pickLang(e.target.value)} aria-label="Farmer message language" style={{ width: 'auto' }}>
          {LANGUAGES.map(([k, v]) => <option key={k} value={k}>{v}</option>)}
        </select>
        <button type="button" className="no-print" onClick={() => load(lang)} disabled={loading} title="Regenerate"><LuRefreshCw /></button>
      </div>

      {err && <p className="alert bad"><LuTriangleAlert />{err}</p>}
      {loading && !adv && <>{[80, 95, 70, 88].map((w, i) => <div key={i} className="skeleton" style={{ width: `${w}%` }} />)}</>}

      {adv && (
        <div style={{ opacity: loading ? 0.5 : 1 }}>
          <div className="headline">{adv.headline}</div>
          <div className="grid g2" style={{ alignItems: 'start' }}>
            <div>
              <h3>What to do with this lot</h3>
              {adv.actions.map((a, i) => (
                <div key={i} className={`action ${a.priority}`}><div className="sev" /><div><b>{a.title}</b><span>{a.detail}</span></div></div>
              ))}
            </div>
            <div className="grid" style={{ gap: 14 }}>
              <div><h3><LuStore style={{ verticalAlign: '-2px' }} /> Sell / sort decision</h3><ul className="bul">{adv.market.map((m, i) => <li key={i}>{m}</li>)}</ul></div>
              <div><h3><LuWarehouse style={{ verticalAlign: '-2px' }} /> Storage</h3><ul className="bul">{adv.storage.map((m, i) => <li key={i}>{m}</li>)}</ul></div>
              <div><h3><LuSprout style={{ verticalAlign: '-2px' }} /> Next season</h3><ul className="bul">{adv.prevention.map((m, i) => <li key={i}>{m}</li>)}</ul></div>
            </div>
          </div>
          <h3 style={{ marginTop: 16 }}>Message for the farmer</h3>
          <div className="farmer">{adv.farmer_message}</div>
          <div className="row no-print" style={{ marginTop: 10 }}>
            <a href={`https://wa.me/?text=${encodeURIComponent(shareText)}`} target="_blank" rel="noreferrer">
              <button type="button"><LuMessageCircle /> Send on WhatsApp</button></a>
            <button type="button" onClick={() => { navigator.clipboard?.writeText(shareText); setCopied(true); setTimeout(() => setCopied(false), 1500) }}>
              <LuCopy /> {copied ? 'Copied' : 'Copy message'}</button>
          </div>
          <details style={{ marginTop: 12 }} className="small muted">
            <summary>What the advisor was told</summary>
            <ul className="bul" style={{ marginTop: 8 }}>{adv.findings.map((f, i) => <li key={i}>{f}</li>)}</ul>
          </details>
        </div>
      )}
    </section>
  )
}
