import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("dark_background")

profiling_file = "profiling_results.txt"

with open(profiling_file, "r") as file:
    lines = file.readlines()

data = []
for line in lines:
    parts = line.strip().split()
    if len(parts) >= 6 and parts[-1].count(":") == 1:
        try:
            ncalls = int(parts[0])  
            cumtime = float(parts[4])
            function = parts[-1]
            data.append((function, ncalls, cumtime))
        except ValueError:
            continue

df = pd.DataFrame(data, columns=["Function", "Calls", "Cumulative Time"])

df["Time per Call"] = df["Cumulative Time"] / df["Calls"]
df_top = df.sort_values(by="Cumulative Time", ascending=False).head(15)
df_top_calls = df.sort_values(by="Calls", ascending=False).head(15)
df_time_per_call = df.sort_values(by="Time per Call", ascending=False).head(15)

fig, axes = plt.subplots(3, 1, figsize=(14, 18))

for ax in axes:
    ax.set_facecolor("black")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")

# 1. Cumulative Execution Time per Function
axes[0].barh(df_top["Function"], df_top["Cumulative Time"], color='dodgerblue')
axes[0].set_xlabel("Cumulative Execution Time (Seconds)", color="white")
axes[0].set_ylabel("Function", color="white")
axes[0].set_title("Top 15 Functions by Cumulative Execution Time", color="white")
axes[0].invert_yaxis()
axes[0].grid(axis='x', linestyle='--', alpha=0.7, color="gray")

# 2. Number of Function Calls
axes[1].barh(df_top_calls["Function"], df_top_calls["Calls"], color='limegreen')
axes[1].set_xlabel("Number of Calls", color="white")
axes[1].set_ylabel("Function", color="white")
axes[1].set_title("Top 15 Functions by Number of Calls", color="white")
axes[1].invert_yaxis()
axes[1].grid(axis='x', linestyle='--', alpha=0.7, color="gray")

# 3. Execution Time per Call
axes[2].barh(df_time_per_call["Function"], df_time_per_call["Time per Call"], color='orchid')
axes[2].set_xlabel("Execution Time per Call (Seconds)", color="white")
axes[2].set_ylabel("Function", color="white")
axes[2].set_title("Top 15 Functions by Execution Time per Call", color="white")
axes[2].invert_yaxis()
axes[2].grid(axis='x', linestyle='--', alpha=0.7, color="gray")

plt.tight_layout()
plt.savefig("profiling_visualization.png")