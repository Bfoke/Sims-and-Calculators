import pandas as pd
import matplotlib.pyplot as plt

filename = "conduction3.csv"

# Read CSV
df = pd.read_csv(filename)

# Convert to seconds
df["time_sec"] = (df["time_ms"] - df["time_ms"].iloc[0]) / 1000.0

df["delta_T"] = df["temp1_c"] - df["temp2_c"]

# Plot temperatures
plt.figure(figsize=(10, 5))

plt.plot(df["time_sec"], df["temp1_c"], lw=2, label="Temp 1")
plt.plot(df["time_sec"], df["temp2_c"], lw=2, label="Temp 2")
plt.plot(df["time_sec"], df["delta_T"], lw=2, label="delta_T")

plt.xlabel("Time (s)")
plt.ylabel("Temperature (°C)")
plt.title("Temperature vs Time")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()