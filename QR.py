from turtle import st

import qrcode
from PIL import Image

# כתובת הגישה שלך
ip_address = "192.168.1.188"
url = f"http://{ip_address}:8501"

# צור את הקוד
qr = qrcode.make(url)
qr.save("qr_code.png")

# הצג את התמונה ב־Streamlit
st.image("qr_code.png", caption="Scan this to open on mobile 📱")
