from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load dataset
data = pd.read_csv("diabetes.csv")
print("dataset size :",data.shape)

# Replace zero values
cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in cols:
    data[col] = data[col].replace(0, data[col].median())

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    values = [float(x) for x in request.form.values()]
    prediction = model.predict([values])

    if prediction[0] == 1:
        result = "Diabetic"
    else:
        result = "Not Diabetic"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)