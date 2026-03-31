# app.py — Dogs vs Cats 분류 웹 서비스
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gdown
import os
MODEL_PATH = "best_model.pt"
@st.cache_resource
def load_model():
    try:
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)

        gdown.download(
            id="10ervIOr-N0k_IvjiCIcqUVstURr46o2e",
            output=MODEL_PATH,
            quiet=False,
            fuzzy=True
        )

        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 1)

        model.load_state_dict(
            torch.load(MODEL_PATH, map_location='cpu')
        )

        model.eval()
        return model

    except Exception as e:
        st.error(f"🔥 모델 로드 실패: {e}")
        return None

# ── 이미지 전처리 (학습 시 eval_transform과 동일) ──────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# ── 페이지 설정 ───────────────────────────────────────────
st.set_page_config(page_title="Dogs vs Cats 분류기", page_icon="🐾")
st.title("🐾 Dogs vs Cats 분류기")
st.caption("이미지를 업로드하면 개인지 고양이인지 분류합니다.")
model = load_model()

if model is None:
    st.stop()

# ── 이미지 업로드 ─────────────────────────────────────────
uploaded = st.file_uploader(
    "이미지를 선택하세요", type=["jpg", "jpeg", "png"]
)

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="업로드된 이미지", use_container_width=True)

    # ── 예측 ──────────────────────────────────────────────
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        logit = model(input_tensor)
        prob = torch.sigmoid(logit).item()

    # ── 결과 표시 ─────────────────────────────────────────
    is_dog = prob >= 0.5
    label = "🐶 Dog" if is_dog else "🐱 Cat"
    confidence = prob if is_dog else 1 - prob

    st.markdown(f"### 예측 결과: {label}")
    st.metric("확신도", f"{confidence:.1%}")

    # 확률 바 시각화
    col1, col2 = st.columns(2)
    with col1:
        st.write("🐱 Cat")
        st.progress(1 - prob)
    with col2:
        st.write("🐶 Dog")
        st.progress(prob)
else:
    st.info("왼쪽 위의 업로드 버튼을 눌러 이미지를 선택하세요.")


