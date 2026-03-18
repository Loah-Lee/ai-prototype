from pathlib import Path

import numpy as np
import onnxruntime as ort
import requests
import streamlit as st

MODEL_PATH = Path("models") / "mnist-12.onnx"
MODEL_CARD_URL = "https://huggingface.co/onnxmodelzoo/mnist-12"
MODEL_DOWNLOAD_URL = "https://huggingface.co/onnxmodelzoo/mnist-12/resolve/main/mnist-12.onnx"
FALLBACK_DOWNLOAD_URL = (
    "https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-12.onnx"
)


def ensure_model_file() -> Path:
    """MNIST ONNX 모델이 없으면 다운로드한 뒤 경로를 반환한다."""
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
        return MODEL_PATH

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    download_errors: list[str] = []

    for url in (MODEL_DOWNLOAD_URL, FALLBACK_DOWNLOAD_URL):
        try:
            _download_model(url, MODEL_PATH)
            return MODEL_PATH
        except requests.RequestException as exc:
            download_errors.append(f"{url}: {exc}")
            if MODEL_PATH.exists():
                MODEL_PATH.unlink()

    raise RuntimeError("모델 다운로드에 실패했습니다.\n" + "\n".join(download_errors))


def _download_model(url: str, destination: Path) -> None:
    response = requests.get(url, timeout=30, stream=True)
    response.raise_for_status()

    with destination.open("wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)


@st.cache_resource(show_spinner=False)
def load_session() -> ort.InferenceSession:
    """세션 간 재사용할 ONNX Runtime 세션을 캐싱한다."""
    model_path = ensure_model_file()
    return ort.InferenceSession(model_path.as_posix(), providers=["CPUExecutionProvider"])


def predict_digit(input_tensor: np.ndarray) -> dict[str, object]:
    """전처리된 숫자 텐서를 받아 0~9 확률과 예측 레이블을 반환한다."""
    session = load_session()
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    logits = session.run([output_name], {input_name: input_tensor.astype(np.float32)})[0][0]
    probabilities = _softmax(logits)
    predicted_label = int(np.argmax(probabilities))

    return {
        "label": predicted_label,
        "confidence": float(probabilities[predicted_label]),
        "probabilities": [float(probability) for probability in probabilities],
    }


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum()
