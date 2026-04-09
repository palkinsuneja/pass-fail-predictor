import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. DATASET (simple example)
# -----------------------------
# Study hours vs Pass(1)/Fail(0)
X = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1,1)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# -----------------------------
# 2. SIGMOID FUNCTION
# -----------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# -----------------------------
# 3. COST FUNCTION
# -----------------------------
def compute_cost(X, y, w, b):
    m = len(y)
    cost = 0
    
    for i in range(m):
        z = np.dot(X[i], w) + b
        f = sigmoid(z)
        cost += -y[i]*np.log(f) - (1-y[i])*np.log(1-f)
    
    return cost/m

# -----------------------------
# 4. GRADIENT DESCENT
# -----------------------------
def gradient_descent(X, y, w, b, alpha, iterations):
    m = len(y)
    
    for _ in range(iterations):
        dj_dw = np.zeros_like(w)
        dj_db = 0
        
        for i in range(m):
            z = np.dot(X[i], w) + b
            f = sigmoid(z)
            
            dj_db += (f - y[i])
            dj_dw += (f - y[i]) * X[i]
        
        dj_dw = dj_dw / m
        dj_db = dj_db / m
        
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
    
    return w, b

# -----------------------------
# 5. TRAIN MODEL
# -----------------------------
w = np.zeros(1)
b = 0

w, b = gradient_descent(X, y, w, b, alpha=0.1, iterations=1000)

print("Trained parameters:")
print("w:", w)
print("b:", b)

# -----------------------------
# 6. PREDICTION FUNCTION
# -----------------------------
def predict(X, w, b):
    preds = []
    for x in X:
        prob = sigmoid(np.dot(x, w) + b)
        preds.append(1 if prob >= 0.5 else 0)
    return np.array(preds)

# -----------------------------
# 7. ACCURACY
# -----------------------------
preds = predict(X, w, b)
accuracy = np.mean(preds == y) * 100
print("Accuracy:", accuracy, "%")

# -----------------------------
# 8. USER INPUT
# -----------------------------
hours = float(input("Enter study hours: "))
prob = sigmoid(hours * w + b)
result = 1 if prob >= 0.5 else 0

print(f"Prediction: {'Pass' if result==1 else 'Fail'}")
print(f"Probability: {prob[0]:.2f}")

# -----------------------------
# 9. VISUALIZATION
# -----------------------------
plt.scatter(X, y, color='red', label="Data")

x_vals = np.linspace(0, 10, 100)
y_vals = sigmoid(x_vals * w + b)

plt.plot(x_vals, y_vals, label="Sigmoid Curve")
plt.xlabel("Study Hours")
plt.ylabel("Probability of Passing")
plt.legend()
plt.title("Pass/Fail Predictor")
plt.show()