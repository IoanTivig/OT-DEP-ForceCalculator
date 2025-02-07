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


def calculate_drag_force_old(velocity, particle_radius_um, fluid_viscosity, distance_to_wall_um, apply_correction=True):
    """
    Calculate the drag force exerted on a particle in a laminar flow, with optional near-wall effects.

    Parameters:
    - velocity: Flow velocity at the particle's location (m/s)
    - particle_radius_um: Radius of the particle in µm
    - fluid_viscosity: Dynamic viscosity of the fluid (Pa·s)
    - distance_to_wall_um: Distance of the particle's center from the wall in µm
    - apply_correction: Boolean flag to enable or disable near-wall correction

    Returns:
    - drag_force_pN: Drag force exerted on the particle in picoNewtons (pN)
    """
    # Convert particle radius and wall distance to meters
    particle_radius_m = particle_radius_um * 1e-6
    distance_to_wall_m = distance_to_wall_um * 1e-6

    # Calculate Stokes drag force
    drag_force = 6 * np.pi * fluid_viscosity * particle_radius_m * velocity  # Force in Newtons (N)

    if apply_correction:
        # Apply near-wall correction based on distance
        if distance_to_wall_m > particle_radius_m:
            # Standard Faxén's correction
            correction_factor = 1 - (9 / 16) * (particle_radius_m / distance_to_wall_m)
        elif distance_to_wall_m == particle_radius_m:
            # Special case: Particle is touching the wall, use asymptotic correction
            correction_factor = 1 - (9 / 16) * (particle_radius_m / distance_to_wall_m) + (1 / 8) * (
                        particle_radius_m / distance_to_wall_m) ** 3
        else:
            # Unphysical case where the particle overlaps the wall
            raise ValueError("Particle radius is larger than distance to the wall, indicating overlap.")

        drag_force /= correction_factor

    # Convert force from Newtons (N) to picoNewtons (pN)
    drag_force_pN = drag_force * 1e12

    return drag_force_pN


def calculate_drag_force(velocity, particle_radius_um, fluid_viscosity, distance_to_wall_um, apply_correction=True):
    """
    Calculate the drag force exerted on a particle in a laminar flow, with optional near-wall effects.

    Parameters:
    - velocity: Flow velocity at the particle's location (m/s)
    - particle_radius_um: Radius of the particle in µm
    - fluid_viscosity: Dynamic viscosity of the fluid (Pa·s)
    - distance_to_wall_um: Distance of the particle's center from the wall in µm
    - apply_correction: Boolean flag to enable or disable near-wall correction

    Returns:
    - drag_force_pN: Drag force exerted on the particle in picoNewtons (pN)
    """
    # Convert particle radius and wall distance to meters
    particle_radius_m = particle_radius_um * 1e-6
    distance_to_wall_m = distance_to_wall_um * 1e-6

    # Calculate classical Stokes drag force (in Newtons)
    drag_force = 6 * np.pi * fluid_viscosity * particle_radius_m * velocity

    if apply_correction:
        # Calculate the gap between the particle surface and the wall.
        # distance_to_wall_m is the distance from the particle's center to the wall,
        # so the gap (epsilon) is given by:
        epsilon = distance_to_wall_m - particle_radius_m

        # If epsilon is zero or negative (particle in contact with the wall), assign a small positive gap.
        if epsilon <= 0:
            print("epsilon is zero or negative, assigning a small positive gap.")
            epsilon = 1e-9  # Adjust as needed based on physical context

        # Compute the correction factor phi(epsilon)
        phi = 2 * math.log(particle_radius_m / epsilon) - 0.9588  # Classic 8/15 correction, 0.9588 is the constant
        phi = 3.08

        # Apply the correction by dividing the classical drag force by phi(epsilon)
        drag_force = drag_force * phi

    # Convert force from Newtons (N) to picoNewtons (pN) (1 N = 1e12 pN)
    drag_force_pN = drag_force * 1e12

    return drag_force_pN


if __name__ == "__main__":
    # General Parameters
    particle_radius_um = 5  # µm

    # Parameters to Test the function with some example values
    flow_rate_ul_min = 1500  # µL/min
    width_mm = 8  # mm
    height_mm = 3  # mm
    offset_um = 0  # µm

    # Parameters to test drag force calculation
    fluid_viscosity = 1e-3  # Water at 20°C (Pa·s)

    v_particle = calculate_velocity(flow_rate_ul_min, width_mm, height_mm, particle_radius_um, offset_um)
    drag_force = calculate_drag_force(v_particle, particle_radius_um, fluid_viscosity, particle_radius_um, apply_correction=True)
    drag_force_wo_corr = calculate_drag_force(v_particle, particle_radius_um, fluid_viscosity, particle_radius_um, apply_correction=False)

    print(f"Velocity at the particle's location: {v_particle:.2e} m/s")
    print(f"Drag force exerted on the particle: {drag_force:.2e} pN")
    print(f"Drag force without correction: {drag_force_wo_corr:.2e} pN")



