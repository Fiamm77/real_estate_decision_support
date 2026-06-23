"""Lightweight retrieval-augmented explanations for valuation results.

The module intentionally avoids an external LLM dependency for the assignment.
It retrieves short domain notes from versioned markdown files and combines
them with model outputs and optional SHAP rows.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
KNOWLEDGE_DIR = BASE_DIR / "docs" / "rag_knowledge"


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def _load_markdown_sections(knowledge_dir: Path = KNOWLEDGE_DIR) -> list[dict]:
    sections = []

    for path in sorted(knowledge_dir.glob("*.md")):
        current_heading = path.stem
        current_lines = []

        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                if current_lines:
                    sections.append(
                        {
                            "source": path.name,
                            "heading": current_heading,
                            "text": " ".join(current_lines).strip(),
                        }
                    )
                current_heading = line.replace("## ", "", 1).strip()
                current_lines = []
            elif line and not line.startswith("# "):
                current_lines.append(line.strip())

        if current_lines:
            sections.append(
                {
                    "source": path.name,
                    "heading": current_heading,
                    "text": " ".join(current_lines).strip(),
                }
            )

    return sections


def retrieve_context(query: str, top_k: int = 3) -> list[dict]:
    """Return the most relevant markdown knowledge sections for a query."""

    query_tokens = _tokenize(query)
    ranked_sections = []

    for section in _load_markdown_sections():
        section_tokens = _tokenize(section["heading"] + " " + section["text"])
        overlap = len(query_tokens & section_tokens)
        ranked_sections.append((overlap, section))

    ranked_sections.sort(key=lambda item: item[0], reverse=True)
    return [section for score, section in ranked_sections[:top_k] if score > 0]


def _format_huf(value) -> str:
    try:
        return f"{float(value):,.0f} Ft".replace(",", " ")
    except (TypeError, ValueError):
        return "nem elérhető"


def _format_signed_huf(value) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "nem elérhető"

    sign = "+" if numeric >= 0 else "-"
    return f"{sign}{abs(numeric):,.0f} Ft".replace(",", " ")


def _format_ratio(value) -> str:
    try:
        return f"{float(value):.2f}x"
    except (TypeError, ValueError):
        return "nem elérhető"


def _format_score(value) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "nem elérhető"


def _normalise_shap_rows(shap_rows: pd.DataFrame | Iterable[dict] | None):
    if shap_rows is None:
        return []

    if isinstance(shap_rows, pd.DataFrame):
        rows = shap_rows.to_dict("records")
    else:
        rows = list(shap_rows)

    return sorted(rows, key=lambda row: abs(float(row.get("shap_value", 0))), reverse=True)


def _estimate_shap_huf_effect(ksh_baseline, shap_value, adjustment_range=0.40):
    try:
        return float(ksh_baseline) * adjustment_range * float(shap_value)
    except (TypeError, ValueError):
        return None


def _context_summary(context: dict) -> str:
    heading = context.get("heading", "Szabály")
    summaries = {
        "KSH piaci benchmark": (
            "a KSH adja a piaci árszintet; hiányzó települési adatnál "
            "területi fallbacket használunk."
        ),
        "Ketkomponensu strukturalt score": (
            "a modell külön telek- és épületjelből becsül relatív "
            "strukturális pozíciót."
        ),
        "KSH korrekcios faktor": (
            "a strukturális score 0.80 és 1.20 közötti KSH-korrekciós "
            "faktorrá alakul."
        ),
        "Benchmark elteres": (
            "pozitív érték benchmark feletti, negatív érték benchmark alatti "
            "strukturális pozíciót jelez."
        ),
        "Felujitas utani szcenario": (
            "a célállapot mellett új score és új becsült piaci érték készül."
        ),
        "SHAP magyarazat": (
            "a legerősebb lokális feature-hatások mutatják, mi tolta fel vagy "
            "le a becslést."
        ),
        "Bizonytalansag": (
            "a becslés bizonytalanabb, ha csak fallback KSH adat vagy "
            "szokatlan bemenet áll rendelkezésre."
        ),
    }
    return summaries.get(heading, str(context.get("text", "")).strip())


def build_explanation(property_row: dict, shap_rows=None, top_k: int = 1) -> str:
    """Build a short Hungarian bullet-list explanation for one valuation."""

    query = " ".join(
        [
            "ingatlan értékbecslés KSH benchmark strukturális score korrekció",
            str(property_row.get("ksh_source_level", "")),
            str(property_row.get("renovated_ksh_source_level", "")),
            "SHAP telek score épület score állapot korrekció",
        ]
    )
    contexts = retrieve_context(query, top_k=top_k)
    shap_items = _normalise_shap_rows(shap_rows)[:3]

    market_value = _format_huf(property_row.get("predicted_market_value"))
    ksh_baseline = _format_huf(property_row.get("ksh_baseline_value_huf"))
    structural_score = _format_score(property_row.get("predicted_structural_score"))
    adjustment_factor = _format_ratio(property_row.get("adjustment_factor"))
    benchmark_delta = _format_signed_huf(property_row.get("benchmark_delta"))
    renovated_value = _format_huf(property_row.get("renovated_market_value"))
    uplift = _format_signed_huf(property_row.get("value_uplift"))
    source_level = property_row.get("ksh_source_level", "nem elérhető")
    ksh_baseline_raw = property_row.get("ksh_baseline_value_huf")

    lines = [
        f"- **Becsült piaci érték:** {market_value}.",
        f"- **KSH benchmark:** {ksh_baseline}; forrásszint: `{source_level}`.",
        (
            "- **Strukturális korrekció:** "
            f"score `{structural_score}`, faktor `{adjustment_factor}`, "
            f"benchmark eltérés {benchmark_delta}."
        ),
        (
            "- **Felújítás utáni scenario:** "
            f"becsült érték {renovated_value}, értéknövekmény {uplift}."
        ),
    ]

    if shap_items:
        lines.append("- **Lokális SHAP tényezők:**")
        for item in shap_items:
            shap_value = float(item.get("shap_value", 0))
            direction = "növelte" if shap_value >= 0 else "csökkentette"
            feature = item.get("feature", "ismeretlen feature")
            value = item.get("feature_value", "n/a")
            estimated_huf_effect = _format_signed_huf(
                _estimate_shap_huf_effect(ksh_baseline_raw, shap_value)
            )
            lines.append(
                f'  - "{feature}" = {value}: {direction} a becslést, '
                f"becsült hatása {estimated_huf_effect}."
            )

    if contexts:
        lines.append("- **RAG értelmezési szabály:**")
        for context in contexts:
            lines.append(f"  - {context['heading']}: {_context_summary(context)}")

    return "\n".join(lines)
