from src.funcs.cell_detection_funcs import *


# Call the function with the folder path
folder_path = "data/images/sample 1"
efs_end_list, efs_start_list, ef_gradients_list, cell_to_target_list, cell_to_origin_list, radi_list, voltages_list = compute_voltage_ramping(folder_path)

CM_factor = 0.8
buffer_permittivity = 78
DEP_forces = []
for i in range(len(efs_end_list)):
    DEP_force = calculate_DEP_force(radi_list[i], buffer_permittivity, CM_factor, ef_gradients_list[i])
    DEP_forces.append(DEP_force)

# Create a figure with 2 subplots (1 row, 2 columns)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))

# Scatter plot 1
ax1.scatter(range(len(efs_end_list)), ef_gradients_list, label="EF gradient in the cell")
ax1.set_title('Squared electric field gradient (V^2/m^3)')
ax1.set_xlabel("Image number")
ax1.set_ylabel('Electric field gradient (V^2/m^3)')

# Scatter plot 2
ax2.scatter(range(len(DEP_forces)), DEP_forces, label="DEP force in the cell (pN)")
ax2.set_title('DEP force in the cell (pN)')
ax2.set_xlabel("Image number")
ax2.set_ylabel('DEP force in the cell (pN)')

# Scatter plot 3
ax3.scatter(ef_gradients_list, voltages_list, label="Voltage applied (Vpp)")
ax3.set_title('Voltage applied (Vpp)')
ax3.set_xlabel("Image number")
ax3.set_ylabel('Voltage applied (Vpp)')

# Scatter plot 4
ax4.scatter(efs_start_list, voltages_list, label="Distance from target to cell")
ax4.set_title('Distance from target to cell (um)')
ax4.set_xlabel("Image number")
ax4.set_ylabel('Distance from target to cell (um)')

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the image
plt.savefig('scatter_plots.png')

# Display the plots
plt.show()