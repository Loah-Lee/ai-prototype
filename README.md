# MNIST Digit Canvas

MNIST ONNX 숫자 인식 Streamlit 서비스 프로젝트입니다.

## 프로젝트 개요

- 사용자가 캔버스에 손으로 숫자를 그리면 ONNX MNIST 모델이 0부터 9까지의 확률을 예측합니다.
- 전처리된 28x28 이미지와 확률 막대 차트를 함께 보여줍니다.
- 추론한 이미지를 로컬 저장소에 저장하고, 예측 라벨과 확률을 다시 조회할 수 있습니다.
- Docker 이미지로 빌드할 수 있도록 `Dockerfile`을 포함합니다.

## 로컬에서 먼저 할 작업

1. Python 3.10 이상과 Docker Desktop을 설치합니다.
2. 제출 폴더를 `mission17/{팀명}_{이름}` 형식으로 준비합니다.
3. 현재 폴더에서 가상환경을 만듭니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

4. 의존성을 설치합니다.

```bash
pip install -r requirements.txt
```

`streamlit-drawable-canvas`는 유지보수 포크 패키지(`streamlit-drawable-canvas-bogaczm`)로 설치되며, import 방식은 동일합니다.

5. 로컬 실행으로 UI와 추론이 정상 동작하는지 확인합니다.

```bash
python -m streamlit run app.py
```

6. Docker Hub 배포를 위해 미리 로그인합니다.

```bash
docker login
```

7. 기존 Streamlit Cloud 앱이 이 저장소를 보고 있다면, 변경 사항을 push 했을 때 재배포되도록 연결 상태를 확인합니다.

## 모델 소스

미션 설명은 GitHub의 ONNX 모델 저장소를 기준으로 작성되어 있지만, ONNX Model Zoo는 **2025-07-01**부터 GitHub LFS 기반 직접 다운로드를 중단했습니다.  
그래서 이 프로젝트는 같은 ONNX Model Zoo의 공식 Hugging Face 미러를 기본 다운로드 경로로 사용합니다.

- 모델 카드: https://huggingface.co/onnxmodelzoo/mnist-12
- 모델 파일: https://huggingface.co/onnxmodelzoo/mnist-12/resolve/main/mnist-12.onnx
- 원본 저장소 안내: https://github.com/onnx/models/tree/main/validated/vision/classification/mnist

## 미션 진행 순서

1. MNIST ONNX 추론 코드를 작성합니다.
2. `streamlit-drawable-canvas`로 숫자 입력 캔버스를 구현합니다.
3. 캔버스 이미지를 흑백 28x28로 전처리하고 모델 입력 텐서로 변환합니다.
4. 0~9 확률을 막대 차트로 시각화합니다.
5. 원본 입력 이미지, 전처리 이미지, 예측 결과를 로컬 저장소에 누적 저장합니다.
6. `Dockerfile`로 이미지를 빌드하고 Docker Hub에 push 합니다.
7. 기존 Streamlit Cloud 저장소에 push 하여 웹 URL을 갱신합니다.

## 실행 방법

```bash
python -m streamlit run app.py
```

첫 실행에서는 `models/mnist-12.onnx` 파일을 자동으로 다운로드합니다.

## Docker

이미지 빌드:

```bash
docker build -t <dockerhub-id>/mnist-digit-canvas:latest .
```

컨테이너 실행:

```bash
docker run -p 8502:8501 <dockerhub-id>/mnist-digit-canvas:latest
```

Docker Hub 업로드:

```bash
docker push <dockerhub-id>/mnist-digit-canvas:latest
```

## 제출물 체크리스트

- 보고서 PDF: 프로젝트 개요, 코드 설명, Docker Hub URL 포함
- 코드 ZIP: 소스 코드, `Dockerfile`, `requirements.txt` 포함
- 배포 링크: 기존 Streamlit Cloud URL 또는 갱신된 서비스 URL
