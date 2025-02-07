import math
import numpy as np


def calculate_velocity(flow_rate_ul_min, width_mm, height_mm, particle_height_um, particle_offset_x_um):
    """
    Refined calculation of the velocity of a laminar flow at a given height and lateral position in a rectangular channel.
    Assumes fully developed flow and uses a 2D parabolic profile approximation with adjusted flattening near the walls.

    Parameters:
    - flow_rate_ul_min: Volumetric flow rate in µL/min
    - width_mm: Channel width in mm
    - height_mm: Channel height in mm
    - particle_height_um: Distance of the particle from the bottom wall in µm
    - particle_offset_x_um: Distance of the particle from the center of the channel width in µm

    Returns:
    - v_particle: Velocity at the particle's location in m/s
    """
    # Convert units to SI (meters and cubic meters per second)
    flow_rate_m3_s = (flow_rate_ul_min * 1e-9) / 60  # µL/min to m³/s
    width_m = width_mm * 1e-3  # mm to m
    height_m = height_mm * 1e-3  # mm to m
    particle_height_m = particle_height_um * 1e-6  # µm to m
    particle_offset_x_m = particle_offset_x_um * 1e-6  # µm to m

    # Calculate cross-sectional area
    area_m2 = width_m * height_m

    # Calculate average velocity
    v_avg = flow_rate_m3_s / area_m2

    # Compute maximum velocity (refined approximation for fully developed flow)
    v_max = 1.9 * v_avg

    # Adjust flattening near the walls with a scaling factor (alpha)
    alpha = 0.45  # Adjust this parameter to refine the decay rate near walls

    # Compute velocity at the particle's location using a refined 2D parabolic profile approximation
    v_particle = v_max * (1 - alpha * (particle_offset_x_m / (width_m / 2)) ** 2) * (
                1 - ((particle_height_m - height_m / 2) / (height_m / 2)) ** 2)


    return v_particle


def average_velocity(flow_rate_ul_min, width_mm, height_mm):
    """
    Computes the average velocity in a rectangular microchannel.

    Parameters:
    - flow_rate_ul_min (float): Volumetric flow rate in µL/min.
    - width_mm (float): Channel width in mm.
    - height_mm (float): Channel height in mm.

    Returns:
    - avg_velocity (float): Average velocity in m/s.
    """
    # Convert units
    W = width_mm * 1e-3  # Convert mm to meters
    H = height_mm * 1e-3  # Convert mm to meters
    Q = flow_rate_ul_min * 1e-9 / 60  # Convert µL/min to m³/s

    # Compute average velocity
    avg_velocity = Q / (W * H)

    return avg_velocity


def calculate_drag_force(velocity_ms, particle_radius_um, fluid_viscosity_Pas, distance_to_wall_um, apply_correction=True, minimum_gap_um=0.001):
    """
    Calculate the drag force exerted on a particle in a laminar flow, with optional near-wall effects.

    Parameters:
    - velocity_ms: Flow velocity at the particle's location (m/s)
    - particle_radius_um: Radius of the particle in µm
    - fluid_viscosity_Pas: Dynamic viscosity of the fluid (Pa·s)
    - distance_to_wall_um: Distance of the particle's center from the wall in µm
    - apply_correction: Boolean flag to enable or disable near-wall correction

    Returns:
    - drag_force_pN: Drag force exerted on the particle in picoNewtons (pN)
    """
    #print(f"velocity: {velocity_ms}")

    # Convert particle radius and wall distance to meters
    particle_radius_m = particle_radius_um * 1e-6
    distance_to_wall_m = distance_to_wall_um * 1e-6
    minimum_gap_m = minimum_gap_um * 0.000001  # Minimum gap to avoid division by zero

    # Calculate classical Stokes drag force (in Newtons)
    drag_force = 6 * np.pi * fluid_viscosity_Pas * particle_radius_m * velocity_ms

    corrected_drag_force = drag_force
    if apply_correction:
        # Calculate the gap between the particle and the wall and set the minimum gap if it is bellow or equal to it
        gap_m = distance_to_wall_m - particle_radius_m  # Minimum gap to avoid division by zero
        if gap_m <= minimum_gap_m:
            print(f"Gap is zero or negative, assigning the minimum gap equal to {minimum_gap_m}.")
            gap_m = minimum_gap_m

        # Calculation according to Faxén's correction
        d = 2 * particle_radius_m # Diameter of the particle
        T = gap_m + particle_radius_m

        faxen_k = 1 / (1 - 9/32 * (d/T) + 1/64 * (d/T)**3 - 45/4096 * (d/T)**4 - 1/512 * (d/T)**5)

        corrected_drag_force = drag_force * faxen_k
        #print(f"Drag_force: {drag_force * 1e12}, Corrected_drag_force: {corrected_drag_force * 1e12}, pN")

        drag_force = corrected_drag_force

    # Convert force from Newtons (N) to picoNewtons (pN) (1 N = 1e12 pN)
    drag_force_pN = drag_force * 1e12

    return drag_force_pN


if __name__ == "__main__":
    # General Parameters
    particle_radius_um = 100 # µm

    # Parameters to Test the function with some example values
    flow_rate_ul_min = 100  # µL/min
    width_mm = 8  # mm
    height_mm = 3  # mm
    offset_um = 0  # µm



    # Parameters to test drag force calculation
    fluid_viscosity = 1e-3  # Water at 20°C (Pa·s)

    v_particle = calculate_velocity(flow_rate_ul_min, width_mm, height_mm, particle_radius_um, offset_um)
    avg_velocity = average_velocity(flow_rate_ul_min, width_mm, height_mm)

    pos_list = [[0, 1500], [0, 1000], [0, 500], [0, 250], [0, 100],
                [500,1500], [500, 1000], [500, 500], [500, 250], [500, 100],
                [1000, 1500], [1000, 1000], [1000, 500], [1000, 250], [1000, 100],
                [2000, 1500], [2000, 1000], [2000, 500], [2000, 250], [2000, 100],
                [3000, 1500], [3000, 1000], [3000, 500], [3000, 250], [3000, 100]]

    v_particle_list = []
    for i in pos_list:
        v_particle_local = calculate_velocity(flow_rate_ul_min, width_mm, height_mm, i[1], i[0])
        v_particle_list.append(v_particle_local)

    avg_velocity_2 = np.average(v_particle_list)


    drag_force = calculate_drag_force(v_particle, particle_radius_um, fluid_viscosity, particle_radius_um, apply_correction=True)
    drag_force_wo_corr = calculate_drag_force(v_particle, particle_radius_um, fluid_viscosity, particle_radius_um, apply_correction=False)

    print(f"Average velocity in the channel: {avg_velocity:.6e} m/s")
    print(f"Average velocity in the channel_2: {avg_velocity_2:.6e} m/s")
    print(f"Velocity at the particle's location: {v_particle:.6e} m/s")
    print(f"Drag force exerted on the particle: {drag_force:.2e} pN")
    print(f"Drag force without correction: {drag_force_wo_corr:.2e} pN")



