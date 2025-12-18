import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Title Card
st.markdown("""
<div class="card">
    <h1>Linear Regression</h1>
    <p>Predict <b>Tip Amount</b> from <b>Total Bill</b> using Linear Regression.</p>
</div>
""", unsafe_allow_html=True)


# Load Data
@st.cache_data
def load_data():
    return sns.load_dataset("tips")

df = load_data()


# Dataset Preview
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df.head())
st.markdown('</div>', unsafe_allow_html=True)


# Prepare Data
x = df[["total_bill"]]
y = df["tip"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# Train Model
model = LinearRegression()
model.fit(x_train_scaled, y_train)

y_pred = model.predict(x_test_scaled)


# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - x_test.shape[1] - 1)


# Visualization
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Total Bill vs Tip Amount")

fig, ax = plt.subplots()
ax.scatter(df["total_bill"], df["tip"], alpha=0.6)

ax.plot(
    df["total_bill"],
    model.predict(scaler.transform(df[["total_bill"]])),
    color="red"
)

ax.set_xlabel("Total Bill ($)")
ax.set_ylabel("Tip Amount ($)")
st.pyplot(fig)

st.markdown('</div>', unsafe_allow_html=True)


# Performance Metrics
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Model Performance Metrics")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
c2.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")
c3.metric("R-squared (R²)", f"{r2:.3f}")
c4.metric("Adjusted R-squared", f"{adj_r2:.3f}")

st.markdown('</div>', unsafe_allow_html=True)


# Model Interpretation
st.markdown(f"""
<div class="card">
    <h3>Model Interpretation</h3>
    <p><b>Coefficient (b1):</b> {model.coef_[0]:.2f}</p>
    <p><b>Intercept (b0):</b> {model.intercept_:.2f}</p>
</div>
""", unsafe_allow_html=True)


# Prediction Section
st.markdown('<div class="card">', unsafe_allow_html=True)

bill = st.slider(
    "Select Total Bill Amount ($)",
    float(df["total_bill"].min()),
    float(df["total_bill"].max()),
    30.0
)

tip = model.predict(scaler.transform([[bill]]))[0]

st.markdown(
    f'<div class="predicition-box">Predicted Tip: $ {tip:.2f}</div>',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)
