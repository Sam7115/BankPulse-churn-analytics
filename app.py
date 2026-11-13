import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# -----------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------

st.set_page_config(
    page_title="BankPulse | Customer Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------
# LOAD MODEL AND PREPROCESSORS
# -----------------------------------------

@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model("model.h5")

    with open("onehot_encoder_geo.pkl", "rb") as file:
        onehot_encoder_geo = pickle.load(file)

    with open("label_encoder_gender.pkl", "rb") as file:
        label_encoder_gender = pickle.load(file)

    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

    return model, onehot_encoder_geo, label_encoder_gender, scaler


model, onehot_encoder_geo, label_encoder_gender, scaler = load_resources()


# -----------------------------------------
# CUSTOM CSS
# -----------------------------------------

st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0b1120 0%, #111c32 100%);
    color: #e5e7eb;
}

/* Remove default header */
header[data-testid="stHeader"] {
    background: transparent;
}

/* Main content spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

/* Main title */
.main-title {
    font-size: 38px;
    font-weight: 750;
    color: #f8fafc;
    letter-spacing: -1.5px;
    margin-bottom: 4px;
}

.subtitle {
    color: #94a3b8;
    font-size: 15px;
    margin-bottom: 30px;
}

/* Section headings */
.section-title {
    font-size: 19px;
    font-weight: 650;
    color: #f1f5f9;
    margin-bottom: 15px;
}

/* Input labels */
label {
    color: #cbd5e1 !important;
}

/* Input fields */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    background-color: #18243a;
    border: 1px solid #293950;
    border-radius: 9px;
}

div[data-baseweb="slider"] {
    padding-top: 5px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #2563eb, #0d9488);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 20px;
    font-weight: 650;
    width: 100%;
    transition: 0.2s ease;
}

.stButton > button:hover {
    border: none;
    transform: translateY(-2px);
    box-shadow: 0 5px 20px #2563eb40;
}

/* Form container */
div[data-testid="stForm"] {
    background-color: #131f33;
    border: 1px solid #26364e;
    border-radius: 15px;
    padding: 22px;
}

/* Result card */
.result-card {
    background: linear-gradient(145deg, #182942, #132137);
    border: 1px solid #30445e;
    border-radius: 16px;
    padding: 28px;
    margin-top: 15px;
}

.result-heading {
    color: #94a3b8;
    font-size: 14px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.result-value {
    font-size: 29px;
    font-weight: 750;
    margin-top: 12px;
    color: #f8fafc;
}

.result-description {
    color: #94a3b8;
    font-size: 14px;
    margin-top: 10px;
    line-height: 1.6;
}

.footer {
    text-align: center;
    color: #64748b;
    font-size: 12px;
    margin-top: 55px;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------------------
# HEADER
# -----------------------------------------

st.markdown("""
<div class="main-title">
    BankPulse <span style="color:#2dd4bf;">.</span>
</div>

<div class="subtitle">
    Customer Intelligence Platform &nbsp; • &nbsp;
    AI-Powered Churn Prediction
</div>
""", unsafe_allow_html=True)

st.divider()


# -----------------------------------------
# MAIN LAYOUT
# -----------------------------------------

left_col, right_col = st.columns([1.25, 0.75], gap="large")


with left_col:

    st.markdown(
        '<div class="section-title">Customer Information</div>',
        unsafe_allow_html=True
    )

    with st.form("customer_form"):

        # Personal Information
        st.caption("01 / PERSONAL DETAILS")

        col1, col2 = st.columns(2)

        with col1:
            geography = st.selectbox(
                "Geography",
                onehot_encoder_geo.categories_[0]
            )

            gender = st.selectbox(
                "Gender",
                label_encoder_gender.classes_
            )

        with col2:
            age = st.slider(
                "Age",
                min_value=18,
                max_value=92,
                value=30
            )

            tenure = st.slider(
                "Tenure (Years)",
                min_value=0,
                max_value=10,
                value=5
            )

        st.divider()

        # Financial Information
        st.caption("02 / FINANCIAL DETAILS")

        col3, col4 = st.columns(2)

        with col3:
            credit_score = st.number_input(
                "Credit Score",
                min_value=300,
                max_value=900,
                value=650
            )

            balance = st.number_input(
                "Account Balance",
                min_value=0.0,
                value=50000.0,
                step=1000.0
            )

        with col4:
            estimated_salary = st.number_input(
                "Estimated Salary",
                min_value=0.0,
                value=50000.0,
                step=1000.0
            )

            num_of_products = st.slider(
                "Number of Products",
                min_value=1,
                max_value=4,
                value=1
            )

        st.divider()

        # Banking Details
        st.caption("03 / BANKING ACTIVITY")

        col5, col6 = st.columns(2)

        with col5:
            has_cr_card = st.selectbox(
                "Has Credit Card?",
                [1, 0],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

        with col6:
            is_active_member = st.selectbox(
                "Active Member?",
                [1, 0],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

        st.write("")

        submitted = st.form_submit_button(
            "Analyze Customer  →",
            use_container_width=True
        )


# -----------------------------------------
# PREDICTION
# -----------------------------------------

with right_col:

    st.markdown(
        '<div class="section-title">Prediction Insights</div>',
        unsafe_allow_html=True
    )

    if submitted:

        # Prepare base input data
        input_data = pd.DataFrame({
            "CreditScore": [credit_score],
            "Gender": [label_encoder_gender.transform([gender])[0]],
            "Age": [age],
            "Tenure": [tenure],
            "Balance": [balance],
            "NumOfProducts": [num_of_products],
            "HasCrCard": [has_cr_card],
            "IsActiveMember": [is_active_member],
            "EstimatedSalary": [estimated_salary]
        })

        # One-hot encode Geography
        geo_input = pd.DataFrame({
            'Geography': [geography]
        })

        geo_encoded = onehot_encoder_geo.transform(geo_input).toarray()

        geo_encoded_df = pd.DataFrame(
            geo_encoded,
            columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
        )

        # Combine numerical and encoded features
        input_data = pd.concat(
            [input_data, geo_encoded_df],
            axis=1
        )

        # Maintain the exact feature order used during training
        if hasattr(scaler, "feature_names_in_"):
            input_data = input_data.reindex(
                columns=scaler.feature_names_in_
            )

        # Scale input
        input_data_scaled = scaler.transform(input_data)

        # Model prediction
        prediction = model.predict(input_data_scaled, verbose=0)

        prediction_proba = float(prediction[0][0])

        probability_percentage = prediction_proba * 100

        # Display result
        if prediction_proba > 0.5:

            st.markdown(f"""
            <div class="result-card">
                <div class="result-heading">Churn Prediction</div>
                <div class="result-value" style="color:#fb7185;">
                    Likely to Churn
                </div>
                <div class="result-description">
                    This customer has a higher probability of leaving
                    the bank. Consider proactive customer retention.
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown(f"""
            <div class="result-card">
                <div class="result-heading">Churn Prediction</div>
                <div class="result-value" style="color:#2dd4bf;">
                    Not Likely to Churn
                </div>
                <div class="result-description">
                    This customer has a lower probability of leaving
                    the bank. Continue monitoring engagement.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")

        st.markdown("**Churn Probability**")

        st.progress(prediction_proba)

        st.markdown(
            f"""
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#94a3b8; font-size:13px;">
                    Model confidence indicator
                </span>
                <span style="color:#f8fafc; font-weight:700;">
                    {probability_percentage:.2f}%
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")
        st.caption("Classification threshold: 50%")

    else:

        st.markdown("""
        <div class="result-card" style="min-height:270px;">
            <div class="result-heading">Awaiting Analysis</div>
            <div class="result-value" style="font-size:23px;">
                Ready when you are.
            </div>
            <div class="result-description">
                Enter the customer's information and click
                <b>Analyze Customer</b> to generate an AI-powered
                churn prediction.
                <br><br>
                Your prediction insights will appear here.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        st.caption("MODEL STATUS")

        st.markdown(
            '<span style="color:#2dd4bf;">●</span> '
            'Neural Network Loaded',
            unsafe_allow_html=True
        )


# -----------------------------------------
# FOOTER
# -----------------------------------------

st.markdown("""
<div class="footer">
    BANKPULSE &nbsp; • &nbsp; CUSTOMER INTELLIGENCE
    <br><br>
    Powered by TensorFlow & Streamlit
</div>
""", unsafe_allow_html=True)
