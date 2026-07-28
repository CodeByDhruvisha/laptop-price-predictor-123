import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ------------------------------
# Page config
# ------------------------------
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="centered"
)

# ------------------------------
# Load model and dataset
# ------------------------------
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# ------------------------------
# Sidebar
# ------------------------------
# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.title("💻 Laptop Price Predictor")

st.sidebar.success("🤖 Machine Learning Project")

st.sidebar.markdown("""
### 👩‍💻 Developer
**Dhruvisha Vaghela**

🎓 B.Tech CSE (AI)  
Parul University

---

### 🛠 Tech Stack
- Python
- Streamlit
- Scikit-Learn
- Pandas
- NumPy

---

### 📌 Project
Predict laptop prices using Machine Learning.

---

### 🔗 Connect With Me

💼 **LinkedIn**  
https://www.linkedin.com/in/dhruvishavaghela/

🐙 **GitHub**  
https://github.com/CodeByDhruvisha
""")


# ----------------------
# Title
# ------------------------------
st.title("💻 Laptop Price Predictor")

st.caption("Predict the estimated laptop price using Machine Learning")

st.divider()

col1, col2 = st.columns(2)

with col1:

    Company = st.selectbox("🏢 Brand", sorted(df['Company'].unique()))

    laptop_type = st.selectbox("💼 Laptop Type", sorted(df['TypeName'].unique()))

    ram = st.selectbox("🧠 RAM (GB)", [2,4,6,8,12,16,24,32,64])

    weight = st.number_input("⚖️ Weight (kg)",0.5,5.0,1.5,0.1)

    touchscreen = st.selectbox("👆 Touchscreen",["No","Yes"])

    ips = st.selectbox("🖥 IPS Display",["No","Yes"])


with col2:

    screen_size = st.slider("📏 Screen Size",10.0,18.0,13.0)

    resolution = st.selectbox(
        "🖼 Resolution",
        [
            '1920x1080',
            '1366x768',
            '1600x900',
            '3840x2160',
            '3200x1800',
            '2880x1800',
            '2560x1600',
            '2560x1440',
            '2304x1440'
        ]
    )

    cpu = st.selectbox("⚙️ Processor", sorted(df['Cpu brand'].unique()))

    gpu = st.selectbox("🎮 GPU", sorted(df['Gpu brand'].unique()))

    ssd = st.selectbox("🚀 SSD (GB)", [0,8,128,256,512,1024])

    hdd = st.selectbox("💽 HDD (GB)", [0,128,256,512,1024,2048])

    os = st.selectbox("🪟 Operating System", sorted(df['os'].unique()))
st.divider()
st.subheader("📋 Laptop Configuration")

# ------------------------------
# Prediction
# ------------------------------
if st.button("🔍 Predict Laptop Price", use_container_width=True):
    touchscreen_val = 1 if touchscreen=='Yes' else 0
    ips_val = 1 if ips=='Yes' else 0

    # PPI calculation
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2 + Y_res**2)**0.5) / screen_size

    query = pd.DataFrame({
        'Company':[Company],
        'TypeName':[laptop_type],
        'Ram':[ram],
        'Weight':[weight],
        'Touchscreen':[touchscreen_val],
        'Ips':[ips_val],
        'ppi':[ppi],
        'Cpu brand':[cpu],
        'HDD':[hdd],
        'SSD':[ssd],
        'Gpu brand':[gpu],
        'os':[os]
    })

    price = np.exp(pipe.predict(query)[0])
    st.success(f"### 💰 Estimated Laptop Price: ₹ {int(price):,}")

    if price < 50000:
        category = "💸 Budget Laptop"
    elif price < 100000:
        category = "⚖️ Mid-Range Laptop"
    else:
        category = "💎 Premium Laptop"


    st.subheader("🎯 Prediction Result")

    col3, col4 = st.columns(2)

    with col3:
        st.metric("💰 Estimated Price", f"₹ {int(price):,}")

    with col4:
        st.metric("🏷 Category", category)

    st.info("📌 The predicted price is an estimate based on the selected laptop specifications.")

st.divider()

st.markdown(
"""
<div style="text-align:center; color:gray;">
© 2026 <b>Dhruvisha Vaghela</b><br>
B.Tech Computer Science Engineering (Artificial Intelligence)<br>
Parul University
</div>
""",
unsafe_allow_html=True)
