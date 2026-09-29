import type { Advice, Health, Inspection } from './types'

// Empty = same origin (Vite dev proxy locally, or the /api function on Vercel).
export const API = import.meta.env.VITE_API_URL ?? ''

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`
    try { msg = (await res.json()).detail ?? msg } catch { /* not json */ }
    throw new Error(msg)
  }
  return res.json() as Promise<T>
}

// ---- results are also kept in this browser: serverless hosts don't keep files between requests
const HISTORY_KEY = 'onioneye_history_v1'
const memory = new Map<string, Inspection>()

function readHistory(): Inspection[] {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]') } catch { return [] }
}

function remember(rec: Inspection) {
  memory.set(rec.id, rec)
  try { sessionStorage.setItem('onioneye_' + rec.id, JSON.stringify(rec)) } catch { /* quota */ }
  const slim = { ...rec, image: '', annotated: rec.annotated.startsWith('data:') ? '' : rec.annotated }
  const list = [slim, ...readHistory().filter(r => r.id !== rec.id)].slice(0, 30)
  try { localStorage.setItem(HISTORY_KEY, JSON.stringify(list)) } catch { /* quota */ }
}

export const getHealth = () => fetch(`${API}/api/health`).then(r => json<Health>(r))

export async function getInspection(id: string): Promise<Inspection> {
  const hit = memory.get(id)
  if (hit) return hit
  try {
    const s = sessionStorage.getItem('onioneye_' + id)
    if (s) return JSON.parse(s)
  } catch { /* ignore */ }
  try {
    return await fetch(`${API}/api/inspections/${id}`).then(r => json<Inspection>(r))
  } catch (e) {
    const local = readHistory().find(r => r.id === id)
    if (local) return local
    throw e
  }
}

export async function listInspections(): Promise<Inspection[]> {
  const local = readHistory()
  if (local.length) return local
  return fetch(`${API}/api/inspections`).then(r => json<Inspection[]>(r))
}

// Phone photos are 3-8 MB; the grading needs ~2000 px. Shrinking here keeps uploads fast and under host limits.
export async function shrinkImage(file: File, maxSide = 2000, quality = 0.9): Promise<Blob> {
  try {
    const bmp = await createImageBitmap(file)
    const s = Math.min(1, maxSide / Math.max(bmp.width, bmp.height))
    const canvas = document.createElement('canvas')
    canvas.width = Math.round(bmp.width * s)
    canvas.height = Math.round(bmp.height * s)
    canvas.getContext('2d')!.drawImage(bmp, 0, 0, canvas.width, canvas.height)
    return await new Promise<Blob>((res, rej) =>
      canvas.toBlob(b => (b ? res(b) : rej(new Error('encode failed'))), 'image/jpeg', quality))
  } catch {
    return file   // old browser: send as is
  }
}

export async function createInspection(form: FormData): Promise<Inspection> {
  const rec = await fetch(`${API}/api/inspections`, { method: 'POST', body: form }).then(r => json<Inspection>(r))
  remember(rec)
  return rec
}

export const media = (path: string) => (!path || path.startsWith('data:') ? path : `${API}${path}`)

// ---- GenAI advisor (falls back to the built-in rulebook on the server when no LLM is reachable)
const adviceCache = new Map<string, Advice>()
export async function getAdvice(rec: Inspection, lang: string, useLlm = true): Promise<Advice> {
  const key = `${rec.id}:${lang}:${useLlm}`
  const hit = adviceCache.get(key)
  if (hit) return hit
  const res = await fetch(`${API}/api/advice`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ result: rec.result, meta: { farmer: rec.farmer, lot_ref: rec.lot_ref, centre: rec.centre }, lang, use_llm: useLlm }),
  }).then(r => json<Advice>(r))
  adviceCache.set(key, res)
  return res
}

export const LANGUAGES: [string, string][] = [
  ['en', 'English'], ['hi', 'हिन्दी'], ['mr', 'मराठी'], ['kn', 'ಕನ್ನಡ'], ['te', 'తెలుగు'], ['ta', 'தமிழ்'], ['gu', 'ગુજરાતી'], ['bn', 'বাংলা'],
]

export function allHistory(): Inspection[] { return readHistory() }
