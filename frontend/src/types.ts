export type Bucket = 'A' | 'URS' | 'Reject'

export interface Onion {
  id: number
  box: [number, number, number, number]
  bucket: Bucket
  reasons: string[]
  defects: string[]
  diameter_mm: number | null
  length_mm: number | null
  weight_g: number | null
  confidence: number | null
  defect_label: string | null
  defect_conf: number | null
  flags: string[]
}

export interface Summary {
  onion_count: number
  count: Record<Bucket, number>
  pct_by_count: Record<Bucket, number>
  pct_by_weight: Record<Bucket, number | null>
  est_total_weight_g: number | null
  decision: string
  rules_version: string
}

export interface Inspection {
  id: string
  created_at: string
  farmer: string
  lot_ref: string
  centre: string
  image: string
  annotated: string
  image_sha256: string
  result_sha256: string
  result: {
    mode: 'ai' | 'finder-only' | 'sheet+classifier' | 'fallback'
    calibration: { method: 'sheet' | 'coin' | null; coin_mm?: number; markers_found?: number[] }
    onions: Onion[]
    summary: Summary
    warnings: string[]
    ignored_objects?: number
  }
}

export interface Health {
  ok: boolean
  finder_loaded: boolean
  classifier_loaded: boolean
  rules_version: string
  inline_media?: boolean
}

export interface AdviceAction { priority: 'high' | 'medium' | 'low'; title: string; detail: string }

export interface Advice {
  source: 'genai' | 'rules'
  model: string
  language: string
  quality_score: number
  headline: string
  findings: string[]
  actions: AdviceAction[]
  storage: string[]
  market: string[]
  prevention: string[]
  farmer_message: string
  facts: {
    onions: number
    size_mm: { mean: number; min: number; max: number; sd: number } | null
    defects: Record<string, number>
    undersized: number
    needs_manual_check: number
  }
  llm_error?: string
}
