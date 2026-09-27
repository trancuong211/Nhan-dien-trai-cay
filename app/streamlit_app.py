"""
Streamlit App — Nhận diện trái cây.
Hỗ trợ: Camera (webcam) hoặc Upload ảnh.

Chạy: streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import streamlit as st
from PIL import Image

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.predict import FruitPredictor


# ==================== Cấu hình trang ====================
st.set_page_config(
    page_title="Nhận diện trái cây",
    page_icon="🍎",
    layout="wide",
)


@st.cache_resource
def load_predictor():
    model_dir = project_root / "models"
    return FruitPredictor(str(model_dir))


def display_results(image, top_k, show_confidence):
    """Hiển thị ảnh và kết quả dự đoán."""
    col_img, col_result = st.columns([1, 1])

    with col_img:
        st.image(image, caption="Ảnh đầu vào", use_container_width=True)

    with col_result:
        st.markdown("### Ket qua du doan")
        try:
            predictor = load_predictor()
            results = predictor.predict_from_pil(image, top_k=top_k)

            top_label, top_conf = results[0]
            st.success(f"**{top_label}** ({top_conf*100:.1f}%)")

            st.markdown("---")
            if show_confidence:
                for rank, (label, conf) in enumerate(results, 1):
                    c1, c2 = st.columns([2, 3])
                    with c1:
                        st.write(f"**{rank}. {label}**")
                    with c2:
                        st.progress(conf, text=f"{conf*100:.1f}%")
            else:
                for rank, (label, conf) in enumerate(results, 1):
                    st.write(f"{rank}. **{label}** — {conf*100:.1f}%")

        except Exception as e:
            st.error(f"Loi du doan: {e}")
            st.info("Dam bao da huan luyen model truoc khi su dung.")


# ==================== Giao diện ====================
st.title("Nhan dien trai cay")
st.markdown("Chup anh hoac tai anh len de nhan dien loai trai cay.")

# Sidebar
with st.sidebar:
    st.header("Cai dat")
    top_k = st.slider("So luong du doan hien thi", 1, 10, 5)
    show_confidence = st.checkbox("Hien thi confidence bar", value=True)
    st.markdown("---")
    st.markdown("### Thong tin model")
    try:
        predictor = load_predictor()
        st.write(f"**So lop:** {len(predictor.class_names)}")
        st.write(f"**Device:** {predictor.device}")
        with st.expander("Danh sach lop"):
            for name in sorted(predictor.class_names):
                st.write(f"- {name}")
    except Exception:
        st.warning("Chua co model. Vui long huan luyen truoc.")

# Tabs: Camera | Upload
tab_camera, tab_upload = st.tabs(["📷 Camera", "📁 Upload anh"])

with tab_camera:
    st.markdown("### Chup anh tu webcam")
    st.info("Nhan **Chup anh** de bat camera, roi chup anh trai cay.")
    camera_input = st.camera_input("Chup anh trai cay")

    if camera_input is not None:
        image = Image.open(camera_input)
        display_results(image, top_k, show_confidence)

with tab_upload:
    st.markdown("### Tai anh len")
    uploaded_file = st.file_uploader(
        "Chon anh trai cay",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="JPG, PNG, BMP, WEBP",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        display_results(image, top_k, show_confidence)
    else:
        st.info("Tai anh trai cay len de bat dau nhan dien!")
        st.markdown("### Huong dan")
        st.markdown("""
        1. Chup hoac tim anh trai cay (tao, chuoi, cam, dau...)
        2. Tai anh len bang nut ben tren
        3. Xem ket qua phan loai va do tin cay
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray;'>"
    "Fruit Recognition — Transfer Learning voi MobileNetV3 | PyTorch"
    "</div>",
    unsafe_allow_html=True,
)
