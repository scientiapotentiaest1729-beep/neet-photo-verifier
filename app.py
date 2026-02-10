import streamlit as st
from deepface import DeepFace
from PIL import Image
import numpy as np
import os

# 1. Setup & Privacy Silencing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
st.set_page_config(page_title="NTA Photo Verifier", page_icon="🆔", layout="wide")

# 2. Header & Privacy Shield
st.title("Student Photo Verifier")
st.caption("A self-help tool for NEET")

with st.container():
    st.markdown("""
    <div style="background-color:#f0f2f6; padding:15px; border-radius:10px; border:1px solid #dcdfe3">
        <small><b>Disclaimer & Privacy:</b> This is an <b>unofficial</b> AI demonstration tool. 
        It is not affiliated with the NTA, NIC, or UIDAI. This tool does not store, collect, or share your biometric data. 
        All processing happens in real-time and is purged instantly. Use this only as a personal reference.</small>
    </div>
    """, unsafe_allow_html=True)

st.write("---")
st.info("🔒 **Privacy Note:** We do not store your photos. Analysis happens in real-time on your session, and images are purged immediately when you close or refresh this tab.")

# 3. NTA Guidelines Section
with st.expander("⚠️ MUST-READ: NTA OFFICIAL PHOTO GUIDELINES"):
    st.markdown("""
    **To ensure 100% acceptance, verify these manually:**
    * **Face Coverage:** Your face should occupy **80%** of the frame.
    * **Background:** Must be **White** or off-white.
    * **Ears:** Both ears should be visible and not covered by hair.
    * **Spectacles:** Avoid wearing glasses during capture to prevent glare rejection.
    """)

# 4. Main Interface
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. ID")
    st.write("Upload the photo from your Aadhar card.")
    img1_file = st.file_uploader("Upload photo", type=['jpg', 'jpeg', 'png'])

with col2:
    st.subheader("2. Live Capture")
    # Toggle between Webcam and QR/Mobile Mimic
    capture_mode = st.radio("Choose Capture Method:", ["Direct Webcam", "Upload(could not integrate the QR feature haha)"])

    if capture_mode == "Direct Webcam":
        img2_file = st.camera_input("Take a Live Photo")
    else:
        img2_file = st.file_uploader("Upload photo", type=['jpg', 'jpeg', 'png'], key="manual_upload")

# 5. The Verification Engine
if img1_file and img2_file:
    # Load images
    img1 = Image.open(img1_file)
    img2 = Image.open(img2_file)
    
    # Display previews
    st.divider()
    c1, c2 = st.columns(2)
    with c1: st.image(img1, caption="Aadhaar Reference", width=300)
    with c2: st.image(img2, caption="Live Capture", width=300)

    if st.button("CHECK"):
        with st.spinner("Analyzing facial geometry against NTA database standards....."):
            try:
                # Process
                img1_arr = np.array(img1)
                img2_arr = np.array(img2)

                result = DeepFace.verify(
                    img1_path=img1_arr, 
                    img2_path=img2_arr, 
                    model_name='VGG-Face', 
                    distance_metric='cosine',
                    enforce_detection=True
                )
                
                dist = result['distance']
                confidence = max(0, 100 - (dist * 100))
                
                st.divider()
                
                # Decision Logic
                if dist < 0.35: 
                    st.success(f"✅ **MATCH VERIFIED** (Confidence: {confidence:.1f}%)")
                    st.balloons()
                    st.markdown("**Status:** Ready for Submission. Your biometric profile matches.")
                elif dist < 0.55:
                    st.warning(f"⚠️ **LOW CONFIDENCE** (Confidence: {confidence:.1f}%)")
                    st.markdown("""
                    **Likely a match, but follow these tips to be safe:**
                    * **Lighting:** Ensure your face is evenly lit (no shadows on one side).
                    * **Quality:** Your Aadhaar scan might be too blurry for the AI.
                    * **NTA Tip:** If you look significantly different now than in your Aadhaar photo, ensure your live photo is extremely clear.
                    """)
                else:
                    st.error(f"❌ **MATCH FAILED** (Confidence: {confidence:.1f}%)")
                    st.markdown("""
                    **The AI could not confidently verify your identity.**
                    * **Check your ID:** Is the Aadhaar photo clear and facing forward?
                    * **Check your Live Photo:** Are you looking directly at the camera?
                    * **Environment:** Try a plain white background to help the AI focus only on your face.
                    """)

            except ValueError:
                st.error("❌ **Face Not Detected!** Please ensure your face is fully visible and not covered by a mask, hand, or heavy glare.")
            except Exception as e:
                st.error(f"System Error: {e}")

# 6. Footer
st.divider()
st.markdown("""
<div style="text-align: center;">
    <p style="color: gray; font-size: 12px;">
        <b>Guidance for Students:</b> If the match fails here, don't panic. 
        Ensure your application photo has a white background and no spectacles.
        This tool is just an attempt to check your match and lower your stress, don't consider it as the final verdict. 
        <br>© 2026 Student Verification Community Initiative
    </p>
</div>
""", unsafe_allow_html=True)