# BankPulse 📊
### AI-Powered Bank Customer Churn Prediction

BankPulse is a machine learning-powered web application that predicts whether a bank customer is likely to leave the bank (churn) based on their demographic, financial, and banking activity information.

Built using **TensorFlow and Streamlit**, the application uses an Artificial Neural Network (ANN) to analyze customer attributes and generate churn predictions along with probability scores.

---

## 🚀 Features

- **Customer Churn Prediction:** Predicts whether a customer is likely to churn.
- **Artificial Neural Network:** Uses a trained deep learning model built with TensorFlow/Keras.
- **Interactive Dashboard:** Clean and modern interface built using Streamlit.
- **Real-time Predictions:** Generates predictions based on user-provided customer information.
- **Data Preprocessing:** Implements Label Encoding, One-Hot Encoding, and Standard Scaling.
- **Churn Probability:** Displays the predicted probability of customer churn.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| TensorFlow / Keras | ANN model development and prediction |
| Streamlit | Interactive web application |
| Pandas | Data manipulation |
| NumPy | Numerical computations |
| Scikit-learn | Data preprocessing and feature encoding |
| Pickle | Saving and loading preprocessing objects |

---

## 🧠 Machine Learning Workflow

The application follows the following workflow:

1. **Data Preprocessing**
   - Label Encoding for categorical gender values.
   - One-Hot Encoding for geographical information.
   - Standard Scaling for numerical features.

2. **Model Development**
   - Developed an Artificial Neural Network (ANN).
   - Trained the model using processed customer data.
   - Saved the trained model for future inference.

3. **Model Deployment**
   - Loaded the trained model and preprocessing objects.
   - Accepted customer information through a Streamlit interface.
   - Applied the same preprocessing pipeline used during training.
   - Generated churn predictions and probability scores.

---

## 📋 Customer Attributes

The model uses the following customer attributes for prediction:

| Feature | Description |
|---|---|
| Credit Score | Customer's credit score |
| Geography | Customer's geographical location |
| Gender | Customer's gender |
| Age | Customer's age |
| Tenure | Number of years associated with the bank |
| Balance | Customer's account balance |
| Number of Products | Number of banking products owned |
| Has Credit Card | Whether the customer owns a credit card |
| Is Active Member | Whether the customer is an active bank member |
| Estimated Salary | Customer's estimated salary |

---

## 📂 Project Structure

```text
BankPulse/
│
├── app.py
├── model.h5
│
├── onehot_encoder_geo.pkl
├── label_encoder_gender.pkl
├── scaler.pkl
│
├── experiments.ipynb
├── requirements.txt
│
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/Sam7115/bankpulse.git
```

### 2. Navigate to the project directory

```bash
cd bankpulse
```

### 3. Create a virtual environment (Optional)

```bash
python -m venv venv
```

Activate it:

**Windows:**
```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📈 Prediction Output

BankPulse generates two possible outcomes:

- **Likely to Churn:** When the predicted churn probability exceeds 50%.
- **Not Likely to Churn:** When the predicted churn probability is 50% or lower.

The application also displays the predicted churn probability to provide additional insight into the model's output.

---

## 🔮 Future Improvements

- Add customer churn analytics and visualizations.
- Introduce customer segmentation based on churn risk.
- Improve prediction performance through hyperparameter tuning.
- Add historical prediction tracking.
- Deploy the application for public access.

---

## 👨‍💻 Author

**Samarth Nalkar**

B.Tech | Electrical Engineering

Interested in Machine Learning, Artificial Intelligence, and Software Development.

---

⭐ If you find this project interesting, consider giving the repository a star!
