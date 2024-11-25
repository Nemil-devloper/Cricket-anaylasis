import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sample DataFrame creation for the top 4 current Indian players
data = {
    'Player': ['Virat Kohli', 'Rohit Sharma', 'KL Rahul', 'Hardik Pandya'],
    'ODI Runs': [13000, 9000, 6000, 3000],  # Hypothetical ODI runs
    'T20 Runs': [3000, 3500, 1800, 1200],   # Hypothetical T20 runs
    'Test Runs': [7000, 4000, 2500, 1500],  # Hypothetical Test runs
    'Average ODI': [57.34, 48.90, 45.50, 39.00],  # Hypothetical ODI averages
    'Average T20': [50.00, 30.75, 34.50, 35.00],  # Hypothetical T20 averages
}

df = pd.DataFrame(data)

# Set index for heatmap
heatmap_data = df.set_index('Player')

# Generate the heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt='.1f')
plt.title('Performance Heatmap of Top 4 Indian Players')
plt.xlabel('Performance Metrics')
plt.ylabel('Players')
plt.show()
