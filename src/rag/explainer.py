"""Lightweight retrieval-augmented explanations for valuation results.

The module intentionally avoids an external LLM dependency for the beadando.
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


def _format_ratio(value) -> str:
    try:
        return f"{float(value):.2f}x"
    except (TypeError, ValueError):
        return "nem elérhető"


def _format_signed_huf(value) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "nem elérhető"

    sign = "+" if numeric >= 0 else "-"
    return f"{sign}{abs(numeric):,.0f} Ft".replace(",", " ")


def _normalise_shap_rows(shap_rows: pd.DataFrame | Iterable[dict] | None):
    if shap_rows is None:
        return []

    if isinstance(shap_rows, pd.DataFrame):
        rows = shap_rows.to_dict("records")
    else:
        rows = list(shap_rows)

    return sorted(rows, key=lambda row: abs(float(row.get("shap_value", 0))), reverse=True)


def build_explanation(property_row: dict, shap_rows=None, top_k: int = 1) -> str:
    """Build a concise Hungarian explanation for one property valuation."""

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
    structural_score = _format_ratio(property_row.get("predicted_structural_score"))
    adjustment_factor = _format_ratio(property_row.get("adjustment_factor"))
    benchmark_delta = _format_huf(property_row.get("benchmark_delta"))
    renovated_value = _format_huf(property_row.get("renovated_market_value"))
    uplift = _format_huf(property_row.get("value_uplift"))
    source_level = property_row.get("ksh_source_level", "nem elérhető")
    ksh_baseline_raw = property_row.get("ksh_baseline_value_huf")
    adjustment_range = 0.40

    lines = [
        "A becslés két részből áll: a KSH benchmark adja az aktuális piaci árszintet, a modell pedig a DF-ből tanult strukturális score alapján korrigálja ezt.",
        f"A becsült piaci érték {market_value}, a KSH piaci benchmark {ksh_baseline}.",
        f"A prediktált strukturális score {structural_score}, a KSH korrekciós faktor {adjustment_factor}.",
        f"A benchmarkhoz képesti eltérés {benchmark_delta}, a KSH árforrás szintje: {source_level}.",
        f"A célállapot szerinti becsült piaci érték {renovated_value}, az értéknövekmény {uplift}.",
    ]

    if shap_items:
        lines.append("A lokális SHAP magyarázat legerősebb tényezői:")
        for item in shap_items:
            shap_value = float(item.get("shap_value", 0))
            direction = "növelte" if shap_value >= 0 else "csökkentette"
            feature = item.get("feature", "ismeretlen feature")
            value = item.get("feature_value", "n/a")
            estimated_huf_effect = _format_signed_huf(
                float(ksh_baseline_raw) * adjustment_range * shap_value
            )
            lines.append(
                f'- "{feature}" = {value}: {direction} a becslést, becsült hatása {estimated_huf_effect}.'
            )

    if contexts:
        lines.append("Rövid értelmezési háttér:")
        for context in contexts:
            lines.append(f"- {context['heading']}: {context['text']}")

    return "\n".join(lines)
