import csv
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from PIL import Image

DATA_DIR = Path("data")
IMAGE_DIR = DATA_DIR / "saved_digits"
METADATA_PATH = DATA_DIR / "predictions.csv"
CSV_FIELDS = [
    "record_id",
    "created_at",
    "original_path",
    "preprocessed_path",
    "predicted_label",
    "confidence",
    "probabilities",
]


def save_prediction(
    original_image: Image.Image,
    preprocessed_image: Image.Image,
    predicted_label: int,
    confidence: float,
    probabilities: list[float],
) -> dict[str, object]:
    """추론 결과와 이미지를 로컬 저장소에 저장한다."""
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    record_id = datetime.now().strftime("%Y%m%d%H%M%S%f")
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")

    original_path = IMAGE_DIR / f"{record_id}_original.png"
    preprocessed_path = IMAGE_DIR / f"{record_id}_preprocessed.png"
    original_image.save(original_path)
    preprocessed_image.save(preprocessed_path)

    record = {
        "record_id": record_id,
        "created_at": created_at,
        "original_path": original_path.as_posix(),
        "preprocessed_path": preprocessed_path.as_posix(),
        "predicted_label": int(predicted_label),
        "confidence": float(confidence),
        "probabilities": json.dumps([float(probability) for probability in probabilities]),
    }

    with METADATA_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        if file.tell() == 0:
            writer.writeheader()
        writer.writerow(record)

    record["probabilities"] = [float(probability) for probability in probabilities]
    return record


def load_prediction_history(limit: int = 12) -> pd.DataFrame:
    """저장된 추론 이력을 최신 순으로 반환한다."""
    if not METADATA_PATH.exists():
        return pd.DataFrame(columns=CSV_FIELDS)

    df = pd.read_csv(METADATA_PATH)
    if df.empty:
        return df

    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["confidence"] = df["confidence"].astype(float)
    df["predicted_label"] = df["predicted_label"].astype(int)
    df["probabilities"] = df["probabilities"].apply(_deserialize_probabilities)
    df = df.sort_values("created_at", ascending=False).head(limit)
    return df


def resolve_asset_path(relative_path: str) -> Path:
    """CSV에 저장된 상대 경로를 현재 작업 디렉터리 기준 절대 경로로 바꾼다."""
    path = Path(relative_path)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _deserialize_probabilities(value: str) -> list[float]:
    if pd.isna(value):
        return []
    return [float(item) for item in json.loads(value)]
