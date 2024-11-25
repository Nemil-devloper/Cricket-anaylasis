import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Data Collection
data = {
    'Match_ID': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'Venue': ['India', 'Australia', 'England', 'New Zealand', 'South Africa', 'Sri Lanka', 'UAE', 'Bangladesh', 'West Indies'],
    'India_Score': [300, 250, 280, 320, 290, 270, 310, 240, 260],
    'Opposition_Score': [290, 260, 300, 310, 280, 230, 300, 250, 270],
    'Toss_Winner': ['India', 'Australia', 'India', 'New Zealand', 'South Africa', 'Sri Lanka', 'UAE', 'Bangladesh', 'West Indies'],
    'Outcome': ['India', 'Australia', 'India', 'New Zealand', 'India', 'Sri Lanka', 'India', 'Bangladesh', 'West Indies']
}

# Create DataFrame
df = pd.DataFrame(data)
df['Outcome'] = df['Outcome'].map({'India': 1, 'Australia': 0, 'New Zealand': 0, 'Sri Lanka': 0, 'Bangladesh': 0, 'West Indies': 0})

# Data Preprocessing
df = pd.get_dummies(df, columns=['Venue', 'Toss_Winner'], drop_first=True)

# Features and target variable
X = df.drop(columns=['Match_ID', 'Outcome'])
y = df['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and Train the Logistic Regression Model
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)

# Make Predictions
y_pred = logistic_model.predict(X_test)

# Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
class_report = classification_report(y_test, y_pred)
print(class_report)

# Step to predict outcomes for new matches
new_matches = {
    'Venue': ['India', 'Australia', 'England'],
    'India_Score': [310, 290, 280],
    'Opposition_Score': [295, 280, 300],
    'Toss_Winner': ['India', 'Australia', 'India'],
}

new_matches_df = pd.DataFrame(new_matches)
new_matches_df = pd.get_dummies(new_matches_df, columns=['Venue', 'Toss_Winner'], drop_first=True)

# Ensure the same feature columns as the training data
new_matches_df = new_matches_df.reindex(columns=X.columns, fill_value=0)

# Make predictions
predictions = logistic_model.predict(new_matches_df)

# Add predictions to the DataFrame
new_matches_df['Predicted_Outcome'] = predictions
new_matches_df['Predicted_Outcome'] = new_matches_df['Predicted_Outcome'].map({1: 'India Wins', 0: 'Opponent Wins'})

# Print the results
print(new_matches_df[['India_Score', 'Opposition_Score', 'Predicted_Outcome']])
