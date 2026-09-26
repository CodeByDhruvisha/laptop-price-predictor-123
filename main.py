from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import pickle


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="Laptop Price Predictor",
    description="Laptop Price Prediction using Machine Learning",
    version="1.0.0"
)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

try:
    with open("pipe.pkl", "rb") as file:
        model = pickle.load(file)

except FileNotFoundError:
    model = None


# =========================================================
# INPUT DATA
# =========================================================

class LaptopInput(BaseModel):

    Company: str = Field(
        description="Laptop company, example: Dell"
    )

    TypeName: str = Field(
        description="Laptop type, example: Notebook"
    )

    Inches: float = Field(
        description="Screen size, example: 15.6"
    )

    ScreenResolution: str = Field(
        description="Screen resolution, example: 1920x1080 Full HD IPS"
    )

    Cpu: str = Field(
        description="Processor, example: Intel Core i5 8250U"
    )

    Ram: str = Field(
        description="RAM, example: 8GB"
    )

    Memory: str = Field(
        description="Storage, example: 256GB SSD"
    )

    Gpu: str = Field(
        description="Graphics card, example: Intel UHD Graphics 620"
    )

    OpSys: str = Field(
        description="Operating system, example: Windows 10"
    )

    Weight: str = Field(
        description="Laptop weight, example: 2.2kg"
    )


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():

    return {
        "message": "Laptop Price Predictor API is running"
    }


# =========================================================
# PREPROCESSING
# =========================================================

def preprocess_laptop(laptop):

    # Create DataFrame
    df = pd.DataFrame([{
        "Company": laptop.Company,
        "TypeName": laptop.TypeName,
        "Inches": laptop.Inches,
        "ScreenResolution": laptop.ScreenResolution,
        "Cpu": laptop.Cpu,
        "Ram": laptop.Ram,
        "Memory": laptop.Memory,
        "Gpu": laptop.Gpu,
        "OpSys": laptop.OpSys,
        "Weight": laptop.Weight
    }])


    # =====================================================
    # RAM
    # =====================================================

    df["Ram"] = (
        df["Ram"]
        .astype(str)
        .str.replace("GB", "", regex=False)
        .str.strip()
    )

    df["Ram"] = pd.to_numeric(
        df["Ram"],
        errors="coerce"
    )

    if df["Ram"].isna().any():
        raise ValueError("Enter RAM like 4GB, 8GB, 16GB")


    # =====================================================
    # WEIGHT
    # =====================================================

    df["Weight"] = (
        df["Weight"]
        .astype(str)
        .str.replace("kg", "", regex=False)
        .str.strip()
    )

    df["Weight"] = pd.to_numeric(
        df["Weight"],
        errors="coerce"
    )

    if df["Weight"].isna().any():
        raise ValueError("Enter weight like 1.5kg or 2.2kg")


    # =====================================================
    # TOUCHSCREEN
    # =====================================================

    df["Touchscreen"] = df["ScreenResolution"].apply(
        lambda x: 1 if "Touchscreen" in x else 0
    )


    # =====================================================
    # IPS
    # =====================================================

    df["Ips"] = df["ScreenResolution"].apply(
        lambda x: 1 if "IPS" in x.upper() else 0
    )


    # =====================================================
    # SCREEN RESOLUTION
    # =====================================================

    resolution = df["ScreenResolution"].str.extract(
        r"(\d+)\s*[xX]\s*(\d+)"
    )

    if resolution.isna().any().any():
        raise ValueError(
            "Enter resolution like 1920x1080 Full HD"
        )

    df["X_res"] = resolution[0].astype(int)
    df["Y_res"] = resolution[1].astype(int)


    # =====================================================
    # PPI
    # =====================================================

    df["ppi"] = (
        (
            df["X_res"] ** 2 +
            df["Y_res"] ** 2
        ) ** 0.5
        / df["Inches"]
    )


    # Remove unused columns

    df.drop(
        columns=[
            "ScreenResolution",
            "Inches",
            "X_res",
            "Y_res"
        ],
        inplace=True
    )


    # =====================================================
    # CPU
    # =====================================================

    df["Cpu Name"] = df["Cpu"].apply(
        lambda x: " ".join(x.split()[0:3])
    )


    def fetch_processor(text):

        if text == "Intel Core i7":
            return "Intel Core i7"

        elif text == "Intel Core i5":
            return "Intel Core i5"

        elif text == "Intel Core i3":
            return "Intel Core i3"

        elif text.startswith("Intel"):
            return "Other Intel Processor"

        else:
            return "AMD Processor"


    df["Cpu brand"] = df["Cpu Name"].apply(
        fetch_processor
    )


    df.drop(
        columns=[
            "Cpu",
            "Cpu Name"
        ],
        inplace=True
    )


    # =====================================================
    # MEMORY
    # =====================================================

    df["Memory"] = (
        df["Memory"]
        .astype(str)
        .replace(r"\.0", "", regex=True)
        .str.replace("GB", "", regex=False)
        .str.replace("TB", "000", regex=False)
    )


    # Split storage

    new = df["Memory"].str.split(
        "+",
        n=1,
        expand=True
    )


    df["first"] = new[0].str.strip()


    if new.shape[1] > 1:
        df["second"] = new[1].fillna("0").str.strip()
    else:
        df["second"] = "0"


    # =====================================================
    # FIRST STORAGE
    # =====================================================

    df["Layer1HDD"] = df["first"].apply(
        lambda x: 1 if "HDD" in x else 0
    )

    df["Layer1SSD"] = df["first"].apply(
        lambda x: 1 if "SSD" in x else 0
    )


    df["first"] = df["first"].str.replace(
        r"\D",
        "",
        regex=True
    )


    # =====================================================
    # SECOND STORAGE
    # =====================================================

    df["Layer2HDD"] = df["second"].apply(
        lambda x: 1 if "HDD" in x else 0
    )

    df["Layer2SSD"] = df["second"].apply(
        lambda x: 1 if "SSD" in x else 0
    )


    df["second"] = df["second"].str.replace(
        r"\D",
        "",
        regex=True
    )


    # Convert to numbers

    df["first"] = pd.to_numeric(
        df["first"],
        errors="coerce"
    ).fillna(0).astype(int)

    df["second"] = pd.to_numeric(
        df["second"],
        errors="coerce"
    ).fillna(0).astype(int)


    # =====================================================
    # HDD
    # =====================================================

    df["HDD"] = (
        df["first"] * df["Layer1HDD"] +
        df["second"] * df["Layer2HDD"]
    )


    # =====================================================
    # SSD
    # =====================================================

    df["SSD"] = (
        df["first"] * df["Layer1SSD"] +
        df["second"] * df["Layer2SSD"]
    )


    # Remove temporary columns

    df.drop(
        columns=[
            "Memory",
            "first",
            "second",
            "Layer1HDD",
            "Layer1SSD",
            "Layer2HDD",
            "Layer2SSD"
        ],
        inplace=True
    )


    # =====================================================
    # GPU
    # =====================================================

    df["Gpu brand"] = df["Gpu"].apply(
        lambda x: x.split()[0]
    )

    df.drop(
        columns=["Gpu"],
        inplace=True
    )


    # =====================================================
    # OPERATING SYSTEM
    # =====================================================

    def cat_os(os):

        if os in [
            "Windows 10",
            "Windows 7",
            "Windows 10 S"
        ]:
            return "Windows"

        elif os in [
            "macOS",
            "Mac OS X"
        ]:
            return "Mac"

        else:
            return "Others/No OS/Linux"


    df["os"] = df["OpSys"].apply(
        cat_os
    )

    df.drop(
        columns=["OpSys"],
        inplace=True
    )


    # =====================================================
    # FINAL FEATURES
    # =====================================================

    final_features = [
        "Company",
        "TypeName",
        "Ram",
        "Weight",
        "Touchscreen",
        "Ips",
        "ppi",
        "Cpu brand",
        "HDD",
        "SSD",
        "Gpu brand",
        "os"
    ]

    return df[final_features]


# =========================================================
# PREDICT PRICE
# =========================================================

@app.post("/predict")
def predict_price(laptop: LaptopInput):

    if model is None:

        raise HTTPException(
            status_code=500,
            detail="pipe.pkl file not found"
        )

    try:

        # Preprocess input
        processed_data = preprocess_laptop(laptop)

        # Predict log(price)
        log_prediction = model.predict(
            processed_data
        )[0]

        # Convert log price to original price
        predicted_price = np.exp(
            log_prediction
        )

        return {
            "predicted_price": round(
                float(predicted_price),
                2
            )
        }

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )