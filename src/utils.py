import json
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import streamlit as st


def inject_css(css_path: Path) -> None:
    """로컬 CSS 파일을 Streamlit 앱에 주입한다."""
    path = Path(css_path)
    if not path.exists():
        return

    st.markdown(f"<style>{path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def build_probability_frame(probabilities: Sequence[float], predicted_label: int) -> pd.DataFrame:
    """막대 차트에 사용할 확률 데이터프레임을 만든다."""
    frame = pd.DataFrame(
        {
            "digit": list(range(len(probabilities))),
            "probability": [float(probability) for probability in probabilities],
        }
    )
    frame["is_predicted"] = frame["digit"] == int(predicted_label)
    return frame


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def format_timestamp(value) -> str:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return "-"
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def summarize_top_candidates(probabilities, limit: int = 3) -> str:
    values = _normalize_probabilities(probabilities)
    top_candidates = sorted(enumerate(values), key=lambda item: item[1], reverse=True)[:limit]
    return ", ".join(f"{digit}: {probability * 100:.1f}%" for digit, probability in top_candidates)


def chunked(items: list[dict], size: int) -> Iterable[list[dict]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _normalize_probabilities(probabilities) -> list[float]:
    if isinstance(probabilities, str):
        return [float(item) for item in json.loads(probabilities)]
    return [float(item) for item in probabilities]
