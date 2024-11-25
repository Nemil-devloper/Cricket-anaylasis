import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Data Collection
# Sample dataset creation
data = {
    'Match_ID': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'Venue': ['India', 'Australia', 'England', 'New Zealand', 'South Africa', 'Sri Lanka', 'UAE', 'Bangladesh', 'West Indies'],
    'India_Score': [300, 250, 280, 320, 290, 270, 310, 240, 260],
    'Opposition_Score': [290, 260, 300, 310, 280, 230, 300, 250, 270],
    'Toss_Winner': ['India', 'Australia', 'India', 'New Zealand', 'South Africa', 'Sri Lanka', 'UAE', 'Bangladesh', 'West Indies'],
    'Outcome': ['India', 'Australia', 'India', 'New Zealand', 'India', 'Sri Lanka', 'India', 'Bangladesh', 'West Indies']  # Match outcome
}

# Create DataFrame
df = pd.DataFrame(data)
df['Outcome'] = df['Outcome'].map({'India': 1, 'Australia': 0, 'New Zealand': 0, 'Sri Lanka': 0, 'Bangladesh': 0, 'West Indies': 0})  # Map outcomes to 1 and 0

# Data Preprocessing
# Convert categorical variables to numerical format using one-hot encoding
df = pd.get_dummies(df, columns=['Venue', 'Toss_Winner'], drop_first=True)

# Features and target variable
X = df.drop(columns=['Match_ID', 'Outcome'])
y = df['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and Train the Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make Predictions
y_pred = rf_model.predict(X_test)

# Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
class_report = classification_report(y_test, y_pred)
print(class_report)

# Feature Importance
importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 5))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()



# Step to predict outcomes for new matches
new_matches = {
    'Venue': ['India', 'Australia', 'England'],  # List of venues
    'India_Score': [310, 290, 280],              # India scores
    'Opposition_Score': [295, 280, 300],         # Opposition scores
    'Toss_Winner': ['India', 'Australia', 'India'],  # Toss winners
}

new_matches_df = pd.DataFrame(new_matches)

# Preprocess the new match data
new_matches_df = pd.get_dummies(new_matches_df, columns=['Venue', 'Toss_Winner'], drop_first=True)

# Ensure the same feature columns as the training data
# Reindex using the feature columns from the training data (X)
new_matches_df = new_matches_df.reindex(columns=X.columns, fill_value=0)

# Make predictions
predictions = rf_model.predict(new_matches_df)

# Add predictions to the DataFrame
new_matches_df['Predicted_Outcome'] = predictions

# Map the outcome to readable format
new_matches_df['Predicted_Outcome'] = new_matches_df['Predicted_Outcome'].map({1: 'India Wins', 0: 'Opponent Wins'})

# Print the results without the 'Venue' column
print(new_matches_df[['India_Score', 'Opposition_Score', 'Predicted_Outcome']])

# Include the actual venue based on one-hot encoding
for index, row in new_matches_df.iterrows():
    venue = 'India' if row.get('Venue_India', 0) == 1 else (
            'Australia' if row.get('Venue_Australia', 0) == 1 else (
            'England' if row.get('Venue_England', 0) == 1 else 'Unknown'))
    print(f"Match Venue: {venue}, India Score: {row['India_Score']}, Opposition Score: {row['Opposition_Score']}, Predicted Outcome: {row['Predicted_Outcome']}")
    
