import { useEffect, useState } from 'react'
import { LuScanSearch, LuMicroscope, LuRuler, LuScale, LuSparkles, LuDatabase, LuCircleCheck, LuCircleX } from 'react-icons/lu'
import { getHealth } from '../api'
import type { Health } from '../types'

const STAGES = [
  { icon: LuScanSearch, name: 'Finder', model: 'YOLO11n-seg · 2.8 M params · ONNX', what: 'Outlines every onion and the reference coin (instance masks).',
    metrics: [['Box mAP50, held-out test (812 photos)', '0.967'], ['Precision / recall', '0.972 / 0.938'], ['Unseen author / camera (2,798 photos)', '0.815'], ['False "onion" on other produce (567 photos)', '71.3% → 0.7%']] },
  { icon: LuRuler, name: 'Calibration & sizing', model: 'OpenCV · coin or ArUco sheet', what: 'Coin diameter (27 mm) gives mm per pixel; each mask gives diameter, length and weight.',
    metrics: [['Mean size error vs hand-measured onions', '≈ 1.3 mm'], ['Grade band width', '10–20 mm']] },
  { icon: LuMicroscope, name: 'Defect checker', model: 'YOLO11n-cls · 224 px · ONNX', what: 'Classifies each onion crop: good · rotten · black mould · sprouted · damaged (+ not-onion filter).',
    metrics: [['Accuracy, held-out test (1,428 crops, 6 classes)', '89.5%'], ['Not-onion filter (F1)', '0.96'], ['Good (F1)', '0.95'], ['Sprouted (F1)', '0.82'], ['Rotten (F1)', '0.78'], ['v1 accuracy (5 classes)', '88.1%']] },
  { icon: LuScale, name: 'Rule engine', model: 'grading_rules.json (versioned)', what: 'Grade A 45–65 mm, no defect · URS 35–70 mm, mould allowed · rot / sprout / damage → reject.',
    metrics: [['Lot accept as Grade A', '≥ 90% A, ≤ 2% reject'], ['Low-confidence onions', 'flagged for manual check']] },
  { icon: LuSparkles, name: 'GenAI advisor', model: 'Groq · gpt-oss-20b · offline rulebook fallback', what: 'Explains the grade, what to do with the lot, storage and market advice, farmer message in 8 Indian languages.',
    metrics: [['Grounding', 'only the measured facts of the lot'], ['Works offline', 'yes (rulebook)']] },
]

export default function Models() {
  const [h, setH] = useState<Health | null>(null)
  useEffect(() => { getHealth().then(setH).catch(() => setH(null)) }, [])
  const Status = ({ ok }: { ok?: boolean }) => ok ? <span className="pill A"><LuCircleCheck style={{ verticalAlign: '-2px' }} /> live</span> : <span className="pill Reject"><LuCircleX style={{ verticalAlign: '-2px' }} /> not loaded</span>

  return (
    <>
      <div className="page-head">
        <div><div className="eyebrow">Transparency</div><h1>How OnionEye grades</h1>
          <div className="muted small">Every number below is measured on photos the models never saw during training.</div></div>
        {h && <div className="row small">Finder <Status ok={h.finder_loaded} /> Defect model <Status ok={h.classifier_loaded} /> Rules <span className="chip">{h.rules_version}</span></div>}
      </div>
      <div className="grid g2">
        {STAGES.map((s, i) => (
          <section key={s.name} className="card model-card">
            <div className="row"><div className="adv-icon"><s.icon /></div><div><div className="small muted">Stage {i + 1}</div><h2 style={{ margin: 0 }}>{s.name}</h2></div></div>
            <div className="small"><span className="chip">{s.model}</span></div>
            <p className="small" style={{ margin: 0 }}>{s.what}</p>
            <div>{s.metrics.map(([k, v]) => <div key={k} className="metric"><span>{k}</span><b>{v}</b></div>)}</div>
          </section>
        ))}
        <section className="card model-card">
          <div className="row"><div className="adv-icon"><LuDatabase /></div><h2 style={{ margin: 0 }}>Training data</h2></div>
          <p className="small" style={{ margin: 0 }}>9,300+ labelled photos from 7 public onion datasets (Roboflow Universe, Mendeley), split by original photo so no image leaks between train and test.
            ~3,000 photos of ginger, garlic, tomato, potato and other vegetables teach the models what is <i>not</i> an onion.</p>
          <div>{['csrp onion-segmentation', 'instance-segmentation-wagk9', 'onionthesis1 (unseen test)', 'onion-disease-vcsiy', 'onions-ciqkj', 'onion-vi5f2', 'onion_sorting', 'Mendeley 42bcyncfhy'].map(d => <span key={d} className="chip">{d}</span>)}</div>
          <p className="small muted" style={{ margin: 0 }}>Limits: internal rot not visible on the skin; accuracy drops under very different lighting; officers can override any onion.</p>
        </section>
      </div>
    </>
  )
}
