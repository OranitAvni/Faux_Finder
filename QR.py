from turtle import st
import streamlit as st

import qrcode
from PIL import Image

# change by IP
ip_address = "10.200.117.176"
url = f"http://{ip_address}:8501"

# Generate the QR code
qr = qrcode.make(url)
qr.save("qr_code.png")

# Display the image in Streamlit
st.image("qr_code.png", caption="Scan this to open on mobile 📱")
