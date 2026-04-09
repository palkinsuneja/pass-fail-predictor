import streamlit as st
import numpy as np

# -----------------------
# TRAINED MODEL (same logic)
# -----------------------
X = np.array([1,2,3,4,5,6,7,8]).reshape(-1,1)
y = np.array([0,0,0,0,1,1,1,1])

def sigmoid(z):
    return 1/(1+np.exp(-z))

def train_model(X, y):
    w = np.zeros(1)
    b = 0
    alpha = 0.1
    
    for _ in range(1000):
        dj_dw = 0
        dj_db = 0
        
        for i in range(len(y)):
            z = X[i]*w + b
            f = sigmoid(z)
            dj_dw += (f - y[i]) * X[i]
            dj_db += (f - y[i])
        
        dj_dw /= len(y)
        dj_db /= len(y)
        
        w -= alpha * dj_dw
        b -= alpha * dj_db
    
    return w, b

w, b = train_model(X, y)

# -----------------------
# UI PART 🔥
# -----------------------
st.title("🎓 Pass/Fail Predictor")

st.write("Enter study hours to predict result")

hours = st.slider("Study Hours", 0.0, 10.0, 1.0)

prob = sigmoid(hours*w + b)[0]
result = "Pass ✅" if prob >= 0.5 else "Fail ❌"

st.subheader("Prediction:")
st.write(result)

st.subheader("Probability:")
st.write(f"{prob:.2f}")