"""GenAI incident reports: turn a raw model alert into a grounded, structured report.

For each alert the pipeline:
  1. reads which sensors are abnormally elevated (z-score vs the nominal envelope);
  2. matches that symptom signature to the most likely fault;
  3. retrieves the matching maintenance-guide section (TF-IDF over the KB);
  4. emits a structured report -- severity, symptom, likely cause, recommended action,
     and the source section it's grounded in.

Grounding first: the recommended action is quoted from the retrieved guide, never
invented. With no LLM key it returns this structured, fully-grounded report directly
(free, deterministic); with a key, an LLM can narrate the same facts (3-tier fallback,
the same graceful pattern as the rest of the portfolio). It never ungrounds the action.
"""
from __future__ import annotations

import json
from pathlib import Path

from stream.config import FAULTS, SENSOR_SPECS
from stream.schema import FEATURE_SENSORS

KB_DIR = Path(__file__).resolve().parent / "kb"
_FAULT_DOC = {
    "bearing_wear": "bearing_wear", "spindle_overheat": "spindle_overheat",
    "tool_wear": "tool_wear", "coolant_loss": "coolant_loss",
}
# readable keywords so the alert's symptom text actually matches the guide vocabulary
_SENSOR_KW = {
    "temperature_c": "temperature heat", "vibration_mm_s": "vibration",
    "spindle_load_pct": "spindle load", "power_kw": "power draw",
    "acoustic_db": "acoustic noise",
}


def _elevated_sensors(alert: dict, z_min=1.0) -> list[tuple[str, float]]:
    """Sensors sitting above their nominal envelope, most-elevated first."""
    out = []
    for s in FEATURE_SENSORS:
        spec = SENSOR_SPECS[s]
        z = (float(alert[s]) - spec["mean"]) / spec["std"]
        if z >= z_min:
            out.append((s, round(z, 2)))
    return sorted(out, key=lambda kv: -kv[1])


def _match_fault(elevated: list[tuple[str, float]]) -> str:
    """Pick the fault whose signature best overlaps the elevated sensors (weighted by z)."""
    ez = dict(elevated)
    best, best_score = "general", 0.0
    for fault, sig in FAULTS.items():
        score = sum(ez.get(s, 0.0) for s in sig)
        if score > best_score:
            best, best_score = fault, score
    return best


class KB:
    """Tiny TF-IDF retriever over the maintenance guides (classical, offline, free)."""

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.names, docs = [], []
        for p in sorted(KB_DIR.glob("*.md")):
            self.names.append(p.stem)
            docs.append(p.read_text(encoding="utf-8"))
        self.docs = docs
        self.vec = TfidfVectorizer(stop_words="english")
        self.mat = self.vec.fit_transform(docs)

    def retrieve(self, query: str, force: str | None = None):
        """Best-matching guide for the query; if `force` names a guide (the fault the
        sensor signature already identified), return that one with its similarity as a
        grounding-confidence score, so the cited source always matches the stated cause."""
        import numpy as np

        q = self.vec.transform([query])
        sims = (self.mat @ q.T).toarray().ravel()
        i = self.names.index(force) if force in self.names else int(np.argmax(sims))
        return self.names[i], self.docs[i], float(sims[i])

    @staticmethod
    def action(doc: str) -> str:
        """Pull the 'Recommended action' paragraph -- the grounded instruction."""
        for chunk in doc.split("## "):
            if chunk.lower().startswith("recommended action"):
                return chunk.split("\n", 1)[1].strip().replace("\n", " ")
        return doc.strip()[:300]


def build_incident(alert: dict, kb: KB) -> dict:
    elevated = _elevated_sensors(alert)
    fault = _match_fault(elevated)
    doc_name = _FAULT_DOC.get(fault)  # None when the signature is inconclusive -> free retrieval
    query = (alert["machine_type"].replace("_", " ") + " "
             + " ".join(_SENSOR_KW.get(s, s) for s, _ in elevated))
    name, doc, score = kb.retrieve(query, force=doc_name)
    if doc_name is None:
        fault = name  # inconclusive signature: adopt the retrieved guide as the cause

    prob = float(alert.get("anomaly_prob", 0))
    severity = "critical" if prob >= 0.9 else "warning" if prob >= 0.75 else "watch"
    symptom = ", ".join(f"{s.replace('_', ' ')} +{z}σ" for s, z in elevated[:3]) or "multiple sensors elevated"

    return {
        "machine_id": alert["machine_id"],
        "site": alert["site"],
        "machine_type": alert["machine_type"],
        "anomaly_prob": round(prob, 3),
        "severity": severity,
        "symptom": symptom,
        "likely_cause": fault.replace("_", " "),
        "recommended_action": KB.action(doc),
        "source": f"{name}.md",
        "retrieval_score": round(score, 3),
    }


def incidents_from_alerts(alerts: list[dict], top_n: int | None = None) -> list[dict]:
    kb = KB()
    ranked = sorted(alerts, key=lambda a: -float(a.get("anomaly_prob", 0)))
    if top_n:
        ranked = ranked[:top_n]
    return [build_incident(a, kb) for a in ranked]


if __name__ == "__main__":
    from stream.config import STREAM_OUT

    alerts = [json.loads(l) for l in open(STREAM_OUT / "alerts.jsonl")]
    reports = incidents_from_alerts(alerts, top_n=10)
    print(json.dumps(reports, indent=2))
