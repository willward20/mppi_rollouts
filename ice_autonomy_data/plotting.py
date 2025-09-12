import csv
import numpy as np
import matplotlib.pyplot as plt

# Path to your saved CSV
csv_name = "n_basis=5_hidden_size=32/2025-09-09_14-39-44"
csv_path = f"src/mppi_rollouts/ice_autonomy_data/fe_rls_errors/{csv_name}.csv"
# /home/administrator/ws/src/mppi_rollouts/ice_autonomy_data/fe_rls_errors/n_basis=5_hidden_size=32/2025-09-09_14-28-43.csv

time_array = []
ice_model_err = []
pave_model_err = []
rls_err = []
node_err = []
fe_err = []

# Read CSV
with open(csv_path, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Convert strings to floats, handle NaN safely
        time_array.append(float(row["time_array"]) if row["time_array"] else float("nan"))
        rls_err.append(float(row["rls_err"]) if row["rls_err"] else float("nan"))
        # node_err.append(float(row["node_err"]) if row["node_err"] else float("nan"))
        # fe_err.append(float(row["fe_err"]) if row["fe_err"] else float("nan"))

# Subtract the beginning of time. 

# Plot
plt.figure(figsize=(8, 5))
plt.plot(time_array, rls_err, label="FE-RLS", color="#2ca02c")
# plt.plot(time_array, fe_err, label="FE", color="#1F77B4")
# plt.plot(time_array, node_err, label="NODE", color="#D62728")

plt.xlabel("Time")
plt.ylabel("Error")
plt.yscale("log")
plt.title("Model Errors Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.savefig(f"src/mppi_rollouts/_data/{csv_name}.png", dpi=300)
plt.show()


# --- Figure 2: Accumulated error curves ---
plt.figure(figsize=(8, 5))
plt.plot(time_array, np.nancumsum(rls_err), label="FE-RLS", color="#2ca02c")
# plt.plot(time_array, np.nancumsum(fe_err), label="FE", color="#1F77B4")
# plt.plot(time_array, np.nancumsum(node_err), label="NODE", color="#D62728")

plt.xlabel("Time")
plt.ylabel("Accumulated Error")
plt.title("Accumulated Model Errors Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
# plt.savefig(f"src/mppi_rollouts/{csv_name}_accumulated.png", dpi=300)