import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the data (assuming it's loaded from a CSV)
try:
    df = pd.read_csv('sampledata.csv')
    print("Data loaded successfully.")
except Exception as e:
    print(f"Error loading data: {e}")

# Encode the 'Label' column to convert it into numerical form
try:
    label_encoder = LabelEncoder()
    df['Label'] = label_encoder.fit_transform(df['Label'])
    print("Labels encoded successfully.")
except Exception as e:
    print(f"Error encoding labels: {e}")

# Split the data into features and labels
try:
    X = df[['ppm']]
    y = df['Label']
    print("Data split into features and labels.")
except Exception as e:
    print(f"Error splitting data: {e}")

# Split into training and testing sets
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data split into training and testing sets.")
except Exception as e:
    print(f"Error splitting data into train and test sets: {e}")

# Train the Logistic Regression model
try:
    model = LogisticRegression()
    model.fit(X_train, y_train)
    print("Model trained successfully.")
except Exception as e:
    print(f"Error training model: {e}")

# Test the model
try:
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy}")
except Exception as e:
    print(f"Error testing model: {e}")

# Save the model and label encoder using pickle
try:
    with open('smell_classifier.pkl', 'wb') as file:
        pickle.dump(model, file)
    with open('label_encoder.pkl', 'wb') as file:
        pickle.dump(label_encoder, file)
    print("Model and label encoder saved successfully.")
except Exception as e:
    print(f"Error saving model and label encoder: {e}")
