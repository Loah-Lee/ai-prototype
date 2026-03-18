from pathlib import Path

import altair as alt
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from src.image_utils import prepare_canvas_for_inference
from src.model import MODEL_CARD_URL, MODEL_PATH, predict_digit
from src.storage import load_prediction_history, resolve_asset_path, save_prediction
from src.utils import (
    build_probability_frame,
    chunked,
    format_percent,
    format_timestamp,
    inject_css,
    summarize_top_candidates,
)

st.set_page_config(page_title="MNIST Digit Canvas", page_icon="✍️", layout="wide")
inject_css(Path("assets/style.css"))

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

if "last_result" not in st.session_state:
    st.session_state.last_result = None

st.title("MNIST Digit Canvas")
st.caption("손으로 숫자를 그리면 ONNX MNIST 모델이 0부터 9까지의 확률을 예측합니다.")

with st.sidebar:
    st.subheader("앱 개요")
    st.write("기존 텍스트 분류 프로토타입을 숫자 인식 서비스로 전환한 버전입니다.")

    st.subheader("모델 정보")
    st.code(str(MODEL_PATH), language="text")
    st.caption("첫 추론 시 ONNX 모델을 다운로드하고, 이후에는 캐시된 세션을 재사용합니다.")
    st.markdown(f"[MNIST-12 모델 카드]({MODEL_CARD_URL})")

    st.subheader("사용 순서")
    st.markdown(
        "1. 캔버스에 숫자를 그립니다.\n"
        "2. `예측 후 저장`을 눌러 결과를 확인합니다.\n"
        "3. 하단 저장소에서 이전 추론 기록을 다시 확인합니다."
    )

canvas_col, result_col = st.columns([1, 1], gap="large")

with canvas_col:
    st.markdown("### 1. 입력 캔버스")
    stroke_width = st.slider("브러시 두께", min_value=10, max_value=36, value=20, step=2)
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=stroke_width,
        stroke_color="#FFFFFF",
        background_color="#000000",
        update_streamlit=False,
        height=320,
        width=320,
        drawing_mode="freedraw",
        display_toolbar=True,
        point_display_radius=0,
        key=f"mnist_canvas_{st.session_state.canvas_key}",
    )

    action_cols = st.columns(2)
    predict_clicked = action_cols[0].button("예측 후 저장", use_container_width=True, type="primary")
    clear_clicked = action_cols[1].button("캔버스 초기화", use_container_width=True)

if clear_clicked:
    st.session_state.canvas_key += 1
    st.session_state.last_result = None
    st.rerun()

if predict_clicked:
    try:
        prepared = prepare_canvas_for_inference(canvas_result.image_data)
        prediction = predict_digit(prepared["input_tensor"])
        record = save_prediction(
            original_image=prepared["original_image"],
            preprocessed_image=prepared["preprocessed_image"],
            predicted_label=prediction["label"],
            confidence=prediction["confidence"],
            probabilities=prediction["probabilities"],
        )
        st.session_state.last_result = {
            "prepared": prepared,
            "prediction": prediction,
            "record": record,
        }
        st.toast("숫자 추론과 저장을 완료했습니다.")
    except ValueError as exc:
        st.warning(str(exc))
    except Exception as exc:
        st.error(f"추론 중 오류가 발생했습니다: {exc}")

last_result = st.session_state.last_result

with result_col:
    st.markdown("### 2. 전처리 이미지 표시")
    if last_result is None:
        st.info("숫자를 그린 뒤 `예측 후 저장`을 눌러 전처리 결과를 확인하세요.")
    else:
        preview_cols = st.columns(2)
        preview_cols[0].image(
            last_result["prepared"]["original_image"],
            caption="캔버스 원본",
            use_container_width=True,
        )
        preview_cols[1].image(
            last_result["prepared"]["preprocessed_image"],
            caption="모델 입력(28x28)",
            use_container_width=True,
        )

    st.markdown("### 3. 모델 추론 결과")
    if last_result is None:
        st.info("추론 결과는 예측 후 여기에 표시됩니다.")
    else:
        metric_cols = st.columns(2)
        metric_cols[0].metric("예측 숫자", last_result["prediction"]["label"])
        metric_cols[1].metric("최고 확률", format_percent(last_result["prediction"]["confidence"]))

        st.caption(
            "상위 후보: "
            f"{summarize_top_candidates(last_result['prediction']['probabilities'])}"
        )
        chart_df = build_probability_frame(
            last_result["prediction"]["probabilities"],
            last_result["prediction"]["label"],
        )
        chart = (
            alt.Chart(chart_df)
            .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
            .encode(
                x=alt.X("digit:O", title="숫자 레이블"),
                y=alt.Y("probability:Q", title="확률", scale=alt.Scale(domain=[0, 1])),
                color=alt.condition(
                    alt.datum.is_predicted,
                    alt.value("#FF6B35"),
                    alt.value("#2D6CDF"),
                ),
                tooltip=[
                    alt.Tooltip("digit:O", title="레이블"),
                    alt.Tooltip("probability:Q", title="확률", format=".4f"),
                ],
            )
            .properties(height=280)
        )
        st.altair_chart(chart, use_container_width=True)
        st.caption(f"마지막 저장 시각: {format_timestamp(last_result['record']['created_at'])}")

st.markdown("---")
st.markdown("### 4. 이미지 저장소")

history_df = load_prediction_history()
if history_df.empty:
    st.info("아직 저장된 숫자 이미지가 없습니다.")
else:
    for row_group in chunked(history_df.to_dict("records"), 3):
        gallery_cols = st.columns(3)
        for col, row in zip(gallery_cols, row_group):
            with col:
                st.image(
                    str(resolve_asset_path(row["original_path"])),
                    caption=f"예측 {row['predicted_label']}",
                    use_container_width=True,
                )
                st.write(f"확률: {format_percent(float(row['confidence']))}")
                st.write(f"저장: {format_timestamp(row['created_at'])}")
                st.caption(summarize_top_candidates(row["probabilities"]))
