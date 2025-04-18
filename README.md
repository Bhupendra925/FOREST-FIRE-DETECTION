# FOREST-FIRE-DETECTION
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load dataset (UCI Forest Fires dataset)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv"
df = pd.read_csv(url)

# 2. Feature selection
features = ['temp', 'RH', 'wind', 'rain']
X = df[features]

# Let's define a target: fire/no fire based on area burned
df['fire'] = df['area'].apply(lambda x: 1 if x > 0 else 0)  # 1 = fire, 0 = no fire
y = df['fire']

# 3. Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Train model (Logistic Regression)
model = LogisticRegression()
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 7. Predict on new data
def predict_fire(temp, RH, wind, rain):
    input_data = scaler.transform([[temp, RH, wind, rain]])
    prediction = model.predict(input_data)[0]
    return "🔥 Fire Risk Detected!" if prediction == 1 else "✅ No Fire Risk"

# Example usage
print(predict_fire (22.1,	60	,3.4	,0.2))  # Try your own values here
