import pandas as pd
import matplotlib.pyplot as plt

filename = "Cell_C_Test_9.csv"

# Read CSV
df = pd.read_csv(filename)

# Convert to seconds
df["time_sec"] = (df["time_ms"] - df["time_ms"].iloc[0]) / 1000.0

# Find minimum temperature
min_idx = df["temp_c"].idxmin()
min_temp = df.loc[min_idx, "temp_c"]
min_time = df.loc[min_idx, "time_sec"]

#find max temp
max_idx = df["temp_c"].idxmax()
max_temp = df.loc[max_idx, "temp_c"]
max_time = df.loc[max_idx, "time_sec"]

# Plot temperature
plt.figure(figsize=(10, 5))
plt.plot(df["time_sec"], df["temp_c"], lw=2, label="Temperature")

# Highlight minimum point
plt.scatter(min_time, min_temp, color="red", s=80, zorder=5)

plt.scatter(max_time, max_temp, color="green", s=80, zorder=5)

# Label minimum point
plt.annotate(
    f"Min Temp\n{min_temp:.2f} °C\nat {min_time:.1f} s",
    xy=(min_time, min_temp),
    xytext=(30, 20),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->", color="black"),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.9)
)

plt.annotate(
    f"Max Temp\n{max_temp:.2f} °C\nat {max_time:.1f} s",
    xy=(max_time, max_temp),
    xytext=(60, -30),
    textcoords="offset points",
    arrowprops=dict(arrowstyle="->", color="black"),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.9)
)

plt.xlabel("Time (s)")
plt.ylabel("Temperature (°C)")
plt.title("Battery Temperature vs Time")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()