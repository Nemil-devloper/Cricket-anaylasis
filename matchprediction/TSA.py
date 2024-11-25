import pandas as pd
import matplotlib.pyplot as plt

# Create the dataset for Yashasvi Jaiswal
data = {
    'Year': [2020, 2021, 2022, 2023],  # Assuming he played from 2020
    'Runs': [140, 200, 300, 550]  # Sample runs data; replace with actual stats
}

# Create a DataFrame
df = pd.DataFrame(data)

# Set the 'Year' column as the index
df.set_index('Year', inplace=True)

# Plotting the time series
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['Runs'], marker='o', color='blue')
plt.title("Yashasvi Jaiswal's ODI Runs (2020-2023)")
plt.xlabel("Year")
plt.ylabel("Runs")
plt.grid()
plt.xticks(df.index)  # Show all year ticks
plt.show()

# Time series analysis: Moving Average
df['Moving Average'] = df['Runs'].rolling(window=2).mean()

# Plotting with moving average
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['Runs'], marker='o', label='Runs', color='blue')
plt.plot(df.index, df['Moving Average'], label='Moving Average', linestyle='--', color='orange')
plt.title("Yashasvi Jaiswal's ODI Runs with Moving Average (2020-2023)")
plt.xlabel("Year")
plt.ylabel("Runs")
plt.legend()
plt.grid()
plt.xticks(df.index)  # Show all year ticks
plt.show()
