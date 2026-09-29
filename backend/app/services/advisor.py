"""GenAI quality advisor.

Turns a graded lot into practical advice: what to do with this lot now (sort / cure / sell / store),
how to avoid the defects next season, and a short message for the farmer in their language.

The numbers (quality score, defect counts, sizes) are computed here, deterministically. The LLM only writes
advice around those facts. If no LLM is reachable (no key, offline centre, quota), a rule-based advisor
built from standard onion post-harvest practice (NHRDF / ICAR-DOGR guidance) answers instead, so the
feature never breaks.

LLM access, first available: Groq (GROQ_API_KEY, openai/gpt-oss-20b), Google Gemini (GEMINI_API_KEY), Vercel AI Gateway (OpenAI-compatible). Auth = AI_GATEWAY_API_KEY env var, or the Vercel OIDC
token that Vercel passes to every function (header x-vercel-oidc-token / env VERCEL_OIDC_TOKEN).
"""
import json
import os
import statistics
import urllib.error
import urllib.request

GATEWAY_URL = "https://ai-gateway.vercel.sh/v1/chat/completions"
MODELS = [os.getenv("ONIONEYE_LLM", "google/gemini-2.5-flash"), "google/gemini-2.5-flash-lite", "openai/gpt-4o-mini"]

LANGS = {"en": "English", "hi": "Hindi", "mr": "Marathi", "kn": "Kannada", "te": "Telugu", "ta": "Tamil",
         "gu": "Gujarati", "bn": "Bengali"}

DEFECT_TEXT = {
    "rotten": "rot (soft rot / neck rot)",
    "black_mould": "black mould (Aspergillus)",
    "sprouted": "sprouting",
    "damaged": "mechanical damage / splits",
}


# ----------------------------------------------------------------------------- facts
def lot_facts(result, meta=None):
    s = result["summary"]
    onions = result.get("onions", [])
    sizes = [o["diameter_mm"] for o in onions if o.get("diameter_mm")]
    defects = {}
    for o in onions:
        for d in o.get("defects", []):
            defects[d] = defects.get(d, 0) + 1
    n = max(1, s["onion_count"])
    under = sum(1 for o in onions if any("undersized" in r for r in o.get("reasons", [])))
    over = sum(1 for o in onions if any("oversized" in r for r in o.get("reasons", [])))
    unsure = sum(1 for o in onions if o.get("flags"))
    # 0-100: Grade A share, minus a penalty for defects that spread in storage
    spread = defects.get("rotten", 0) * 2 + defects.get("black_mould", 0) + defects.get("sprouted", 0)
    score = max(0, min(100, round(s["pct_by_count"]["A"] + 0.5 * s["pct_by_count"]["URS"] - 30 * spread / n)))
    return {
        "onions": s["onion_count"], "decision": s["decision"],
        "pct": s["pct_by_count"], "pct_weight": s.get("pct_by_weight"),
        "est_weight_kg_sample": round(s["est_total_weight_g"] / 1000, 2) if s.get("est_total_weight_g") else None,
        "size_mm": ({"mean": round(statistics.mean(sizes), 1), "min": round(min(sizes), 1), "max": round(max(sizes), 1),
                     "sd": round(statistics.pstdev(sizes), 1)} if sizes else None),
        "defects": defects, "undersized": under, "oversized": over, "needs_manual_check": unsure,
        "quality_score": score,
        "calibration": result.get("calibration", {}).get("method"),
        "centre": (meta or {}).get("centre") or None,
    }


# ----------------------------------------------------------------------------- rule-based advisor
def rule_advice(f):
    n = max(1, f["onions"])
    d = f["defects"]
    actions, storage, market, findings = [], [], [], []

    findings.append(f"{f['pct']['A']}% Grade A, {f['pct']['URS']}% URS, {f['pct']['Reject']}% reject "
                    f"out of {f['onions']} onions. Lot decision: {f['decision']}.")
    if f["size_mm"]:
        z = f["size_mm"]
        findings.append(f"Average size {z['mean']} mm (range {z['min']}-{z['max']} mm); "
                        + ("sizes are uneven, size-sort before sale." if z["sd"] > 8 else "sizes are fairly uniform."))
    for k, c in sorted(d.items(), key=lambda kv: -kv[1]):
        findings.append(f"{c} onion(s) with {DEFECT_TEXT.get(k, k)} ({round(100 * c / n)}%).")

    if d.get("rotten"):
        actions.append({"priority": "high", "title": "Remove rotten bulbs today",
                        "detail": "Rot spreads to neighbouring bulbs in bags and heaps. Pick them out before weighing "
                                  "or storing and keep them away from the good lot."})
    if d.get("black_mould"):
        actions.append({"priority": "medium", "title": "Dry and cure mould-affected bulbs",
                        "detail": "Black mould grows on moist, bruised onions. Spread in shade with good airflow for "
                                  "a few days; sell these first (URS) rather than storing them."})
    if d.get("sprouted"):
        actions.append({"priority": "high", "title": "Sell sprouted stock first",
                        "detail": "Sprouting means dormancy has ended; weight and quality drop fast. Do not mix with "
                                  "storage stock."})
    if d.get("damaged"):
        actions.append({"priority": "medium", "title": "Handle gently",
                        "detail": "Cuts and bruises are entry points for rot. Use crates or padded bags, avoid "
                                  "throwing bags, and don't overfill."})
    if f["undersized"]:
        actions.append({"priority": "medium", "title": "Size-sort before selling",
                        "detail": f"{f['undersized']} small onion(s) pull the lot down. Separate them and sell as a "
                                  "separate small-grade lot so the main lot qualifies for a better grade."})
    if f["pct"]["A"] >= 90 and not d:
        actions.append({"priority": "low", "title": "Lot is ready for Grade A procurement",
                        "detail": "Keep the bags in shade and dry until weighing."})
    if f["needs_manual_check"]:
        actions.append({"priority": "low", "title": "Check flagged onions by hand",
                        "detail": f"{f['needs_manual_check']} onion(s) had low AI confidence; an officer should look at them."})

    storage += ["Cure properly before storage: field-dry 3-5 days, then shade-dry 2-3 weeks until necks are tight and dry.",
                "Store only sound, cured, sorted bulbs in a well-ventilated onion store (bottom and side ventilation).",
                "Keep relative humidity around 65-70%; avoid rain and floor moisture; check and remove rotting bulbs every 2 weeks."]
    if d.get("sprouted"):
        storage.append("For long storage, keep bulbs cool and dry; high warmth plus humidity triggers sprouting.")

    if f["pct"]["A"] >= 90:
        market.append("Qualifies for Grade A: offer the full lot to the procurement agency.")
    elif f["pct"]["A"] + f["pct"]["URS"] >= 90:
        market.append("Qualifies as URS. Sorting out the non-Grade-A onions could move part of the lot up to Grade A.")
    else:
        market.append("Re-sort before offering: remove rejects, then re-inspect; unsorted, the lot is likely to be rejected.")
    if d.get("sprouted") or d.get("black_mould"):
        market.append("Sell affected onions quickly in the local market rather than storing them for a price rise.")

    return {"findings": findings, "actions": actions, "storage": storage, "market": market,
            "prevention": ["Stop irrigation 10-15 days before harvest so bulbs mature and dry.",
                           "Harvest when 50-75% of tops have fallen; avoid harvesting in rain.",
                           "Avoid excess nitrogen late in the crop; balanced potash improves storage life."]}


FARMER_TPL = {
    "hi": "OnionEye जाँच: {A}% ग्रेड A, {U}% URS, {R}% रिजेक्ट। निर्णय: {D}। {X}इस रिपोर्ट को ग्रेड के प्रमाण के रूप में रखें।",
    "mr": "OnionEye तपासणी: {A}% ग्रेड A, {U}% URS, {R}% नाकारलेले. निर्णय: {D}. {X}ही अहवाल ग्रेडचा पुरावा म्हणून जपून ठेवा.",
    "kn": "OnionEye ಪರಿಶೀಲನೆ: {A}% ಗ್ರೇಡ್ A, {U}% URS, {R}% ತಿರಸ್ಕೃತ. ನಿರ್ಧಾರ: {D}. {X}ಈ ವರದಿಯನ್ನು ಗ್ರೇಡ್‌ನ ಪುರಾವೆಯಾಗಿ ಇಟ್ಟುಕೊಳ್ಳಿ.",
}
FARMER_X = {"hi": "तौल से पहले सड़े या खराब प्याज़ निकाल दें। ", "mr": "वजनापूर्वी सडलेले किंवा खराब कांदे काढून टाका. ",
            "kn": "ತೂಕಕ್ಕೆ ಮೊದಲು ಕೊಳೆತ ಅಥವಾ ಹಾಳಾದ ಈರುಳ್ಳಿಗಳನ್ನು ತೆಗೆದುಹಾಕಿ. "}


def farmer_message(f, lang):
    if lang in FARMER_TPL:
        return FARMER_TPL[lang].format(A=f["pct"]["A"], U=f["pct"]["URS"], R=f["pct"]["Reject"], D=f["decision"],
                                       X=FARMER_X[lang] if f["defects"] else "")
    return farmer_message_en(f)


def farmer_message_en(f):
    return (f"Your onions were checked by OnionEye: {f['pct']['A']}% Grade A, {f['pct']['URS']}% URS, "
            f"{f['pct']['Reject']}% reject. Decision: {f['decision']}. "
            + ("Please remove damaged or rotten onions before weighing. " if f["defects"] else "")
            + "Keep this report as proof of the grade.")


# ----------------------------------------------------------------------------- LLM
SYSTEM = """You are OnionEye Advisor, an onion post-harvest and quality expert for Indian government procurement
centres (NAFED/NCCF, Dept. of Consumer Affairs). You get the machine-graded facts of one onion lot.
Write practical, specific, safe advice. Never invent numbers beyond the facts. No pesticide names or doses;
for chemical treatment say "consult the local agriculture officer". Keep each item one or two sentences.
Return ONLY a JSON object with keys:
 headline (string, <= 14 words),
 findings (array of 2-4 strings: what the numbers mean),
 actions (array of 2-5 objects {priority: "high"|"medium"|"low", title, detail}: what to do with THIS lot now),
 storage (array of 2-4 strings), market (array of 1-3 strings: sell / sort / store decision),
 prevention (array of 2-3 strings: next season, field practice),
 farmer_message (string, 2-4 short sentences, in the requested language and its own script, simple words)."""


def _token(request_token=None):
    return os.getenv("AI_GATEWAY_API_KEY") or request_token or os.getenv("VERCEL_OIDC_TOKEN")


def _gemini(user, timeout):
    """Direct Google Gemini API (free tier key from aistudio.google.com), used when GEMINI_API_KEY is set."""
    key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    body = {"systemInstruction": {"parts": [{"text": SYSTEM}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 2048, "responseMimeType": "application/json"}}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read().decode())
    text = "".join(p.get("text", "") for p in out["candidates"][0]["content"]["parts"])
    return json.loads(text[text.find("{"): text.rfind("}") + 1]), f"google/{model}"


def _groq(user, timeout):
    """Groq (OpenAI-compatible), default model openai/gpt-oss-20b. Key: GROQ_API_KEY env var."""
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    body = {"model": model, "temperature": 0.3, "max_completion_tokens": 3000, "reasoning_effort": "low",
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]}
    for attempt in range(2):
        req = urllib.request.Request("https://api.groq.com/openai/v1/chat/completions", data=json.dumps(body).encode(),
                                     method="POST", headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
                                                             "Content-Type": "application/json", "User-Agent": "OnionEye/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out = json.loads(r.read().decode())
            text = out["choices"][0]["message"]["content"] or ""
            return json.loads(text[text.find("{"): text.rfind("}") + 1]), f"groq/{model}"
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="ignore")[:200]
            if e.code == 400 and attempt == 0:          # e.g. JSON mode not accepted: retry as plain text
                body.pop("response_format", None); body.pop("reasoning_effort", None)
                continue
            raise RuntimeError(f"groq {e.code}: {msg}")
    raise RuntimeError("groq failed")


def llm_advice(f, lang, request_token=None, timeout=25):
    user = (f"Lot facts (JSON): {json.dumps(f)}\nFarmer message language: {LANGS.get(lang, 'English')}.\n"
            "Grades: Grade A = 45-65 mm, no defects; URS = 35-70 mm, black mould allowed; reject = rot, sprout, damage.")
    errors = []
    if os.getenv("GROQ_API_KEY"):
        try:
            return _groq(user, timeout)
        except Exception as e:
            errors.append(f"groq: {e}")
    if os.getenv("GEMINI_API_KEY"):
        try:
            return _gemini(user, timeout)
        except Exception as e:
            errors.append(f"gemini: {e}")
    tok = _token(request_token)
    if not tok:
        raise RuntimeError("; ".join(errors) or "no LLM credentials")
    user = (f"Lot facts (JSON): {json.dumps(f)}\nFarmer message language: {LANGS.get(lang, 'English')}.\n"
            "Grades: Grade A = 45-65 mm, no defects; URS = 35-70 mm, black mould allowed; reject = rot, sprout, damage.")
    last = None
    for model in MODELS:
        body = {"model": model, "temperature": 0.3, "max_tokens": 1400,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]}
        req = urllib.request.Request(GATEWAY_URL, data=json.dumps(body).encode(), method="POST",
                                     headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out = json.loads(r.read().decode())
            text = out["choices"][0]["message"]["content"].strip()
            if text.startswith("```"):
                text = text.strip("`").split("\n", 1)[1].rsplit("```", 1)[0]
            data = json.loads(text[text.find("{"): text.rfind("}") + 1])
            return data, model
        except urllib.error.HTTPError as e:
            last = f"{e.code} {e.read().decode(errors='ignore')[:160]}"
            if e.code in (401, 402, 403):   # auth / billing problem: other models won't work either
                break
        except Exception as e:           # try the next model
            last = e
    raise RuntimeError("; ".join(errors + [f"gateway: {last}"]))


def advise(result, meta=None, lang="en", request_token=None, use_llm=True):
    f = lot_facts(result, meta)
    base = rule_advice(f)
    out = {"facts": f, "language": lang, "quality_score": f["quality_score"]}
    if use_llm:
        try:
            data, model = llm_advice(f, lang, request_token)
            merged = {**base, **{k: v for k, v in data.items() if v}}
            return {**out, **merged, "source": "genai", "model": model}
        except Exception as e:
            out["llm_error"] = str(e)[:200]
    return {**out, **base, "headline": f"Lot decision: {f['decision']}",
            "farmer_message": farmer_message(f, lang), "source": "rules", "model": "OnionEye rulebook"}
