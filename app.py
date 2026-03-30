import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ---------------------------------------------------
# Page config (this adds icon in browser tab)
# ---------------------------------------------------
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="centered"
)

# ---------------------------------------------------
# Load model
# ---------------------------------------------------
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

# ---------------------------------------------------
# Title with logo
# ---------------------------------------------------
st.markdown(
    "<h1 style='text-align: center;'>💻 Laptop Price Predictor</h1>",
    unsafe_allow_html=True
)

st.write("### Enter laptop specifications below")

st.write("---")

# ---------------------------------------------------
# Inputs
# ---------------------------------------------------

Company = st.selectbox('Brand', df['Company'].unique())
laptop_type = st.selectbox('Type', df['TypeName'].unique())

ram = st.selectbox('RAM (GB)', [2,4,6,8,12,16,24,32,64])
weight = st.number_input('Weight of Laptop (kg)', min_value=0.5, max_value=5.0, step=0.1)

touchscreen = st.selectbox('Touchscreen', ['No','Yes'])
ips = st.selectbox('IPS Display', ['No','Yes'])

screen_size = st.slider('Screen Size (inches)', 10.0, 18.0, 13.0)

resolution = st.selectbox(
    'Screen Resolution',
    ['1920x1080','1366x768','1600x900','3840x2160',
     '3200x1800','2880x1800','2560x1600','2560x1440','2304x1440']
)

cpu = st.selectbox('CPU Brand', df['Cpu brand'].unique())
hdd = st.selectbox('HDD (GB)', [0,128,256,512,1024,2048])
ssd = st.selectbox('SSD (GB)', [0,8,128,256,512,1024])
gpu = st.selectbox('GPU Brand', df['Gpu brand'].unique())
os = st.selectbox('Operating System', df['os'].unique())

st.write("---")

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------
if st.button('💰 Predict Price'):

    touchscreen_val = 1 if touchscreen == 'Yes' else 0
    ips_val = 1 if ips == 'Yes' else 0

    # Calculate PPI
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2 + Y_res**2)**0.5) / screen_size

    # DataFrame for prediction
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
    st.balloons()

