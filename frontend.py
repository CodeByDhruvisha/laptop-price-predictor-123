import streamlit as st
import requests
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
# FastAPI URL
# ------------------------------
API_URL = "http://127.0.0.1:8000/predict"

# ------------------------------
# Load ORIGINAL dataset
# ------------------------------
df = pd.read_csv("laptop_data.csv")

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
- FastAPI
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

# ------------------------------
# Title
# ------------------------------
st.title("💻 Laptop Price Predictor")

st.caption(
    "Predict the estimated laptop price using Machine Learning"
)

st.divider()

# ------------------------------
# Laptop Configuration
# ------------------------------
col1, col2 = st.columns(2)

with col1:

    Company = st.selectbox(
        "🏢 Brand",
        sorted(df["Company"].dropna().unique())
    )

    laptop_type = st.selectbox(
        "💼 Laptop Type",
        sorted(df["TypeName"].dropna().unique())
    )

    ram = st.selectbox(
        "🧠 RAM (GB)",
        [2, 4, 6, 8, 12, 16, 24, 32, 64]
    )

    weight = st.number_input(
        "⚖️ Weight (kg)",
        min_value=0.5,
        max_value=5.0,
        value=1.5,
        step=0.1
    )

    touchscreen = st.selectbox(
        "👆 Touchscreen",
        ["No", "Yes"]
    )

    ips = st.selectbox(
        "🖥 IPS Display",
        ["No", "Yes"]
    )


with col2:

    screen_size = st.slider(
        "📏 Screen Size",
        10.0,
        18.0,
        13.0
    )

    resolution = st.selectbox(
        "🖼 Resolution",
        [
            "1920x1080",
            "1366x768",
            "1600x900",
            "3840x2160",
            "3200x1800",
            "2880x1800",
            "2560x1600",
            "2560x1440",
            "2304x1440"
        ]
    )

    # ORIGINAL CPU
    cpu = st.selectbox(
        "⚙️ Processor",
        sorted(df["Cpu"].dropna().unique())
    )

    # ORIGINAL GPU
    gpu = st.selectbox(
        "🎮 GPU",
        sorted(df["Gpu"].dropna().unique())
    )

    ssd = st.selectbox(
        "🚀 SSD (GB)",
        [0, 8, 128, 256, 512, 1024]
    )

    hdd = st.selectbox(
        "💽 HDD (GB)",
        [0, 128, 256, 512, 1024, 2048]
    )

    # ORIGINAL OS
    os = st.selectbox(
        "🪟 Operating System",
        sorted(df["OpSys"].dropna().unique())
    )

st.divider()

st.subheader("📋 Laptop Configuration")

# ------------------------------
# Prediction
# ------------------------------
if st.button(
    "🔍 Predict Laptop Price",
    use_container_width=True
):

    # ------------------------------
    # Screen Resolution
    # ------------------------------

    screen_resolution = resolution

    if ips == "Yes":
        screen_resolution += " IPS"

    if touchscreen == "Yes":
        screen_resolution += " Touchscreen"

    # ------------------------------
    # Memory
    # ------------------------------

    if ssd > 0 and hdd > 0:

        memory = f"{hdd}GB HDD + {ssd}GB SSD"

    elif ssd > 0:

        memory = f"{ssd}GB SSD"

    elif hdd > 0:

        memory = f"{hdd}GB HDD"

    else:

        memory = "No storage"

    # ------------------------------
    # JSON Payload
    # ------------------------------

    payload = {
        "Company": Company,
        "TypeName": laptop_type,
        "Inches": screen_size,
        "ScreenResolution": screen_resolution,
        "Cpu": cpu,
        "Ram": f"{ram}GB",
        "Memory": memory,
        "Gpu": gpu,
        "OpSys": os,
        "Weight": f"{weight}kg"
    }

    # ------------------------------
    # Show API Request
    # ------------------------------

    with st.expander("🔎 View API Request"):
        st.json(payload)

    # ------------------------------
    # Send to FastAPI
    # ------------------------------

    try:

        response = requests.post(
            API_URL,
            json=payload,
            timeout=30
        )

        # ------------------------------
        # Successful response
        # ------------------------------

        if response.status_code == 200:

            result = response.json()

            price = result["predicted_price"]

            # ------------------------------
            # Category
            # ------------------------------

            if price < 50000:

                category = "💸 Budget Laptop"

            elif price < 100000:

                category = "⚖️ Mid-Range Laptop"

            else:

                category = "💎 Premium Laptop"

            # ------------------------------
            # Result
            # ------------------------------

            st.success(
                f"### 💰 Estimated Laptop Price: ₹ {int(price):,}"
            )

            st.subheader("🎯 Prediction Result")

            col3, col4 = st.columns(2)

            with col3:

                st.metric(
                    "💰 Estimated Price",
                    f"₹ {int(price):,}"
                )

            with col4:

                st.metric(
                    "🏷 Category",
                    category
                )

            st.info(
                "📌 The predicted price is an estimate based on "
                "the selected laptop specifications."
            )

        else:

            st.error(
                f"❌ API Error: {response.status_code}"
            )

            st.code(response.text)

    except requests.exceptions.ConnectionError:

        st.error(
            "❌ Cannot connect to FastAPI."
        )

        st.warning(
            "Start FastAPI first:"
        )

        st.code(
            "python -m uvicorn main:app --reload"
        )

    except requests.exceptions.Timeout:

        st.error(
            "⏳ FastAPI request timed out."
        )

    except Exception as e:

        st.error(
            f"❌ Error: {str(e)}"
        )

# ------------------------------
# Footer
# ------------------------------

st.divider()

st.markdown(
"""
<div style="text-align:center; color:gray;">

© 2026 <b>Dhruvisha Vaghela</b><br>

B.Tech Computer Science Engineering (Artificial Intelligence)<br>

Parul University

</div>
""",
unsafe_allow_html=True
)