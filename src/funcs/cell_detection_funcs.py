import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# First, we need to detect the cell of intrest which will remain the same thorugh all the rest of the images
# We will need to detect the cells and select the one that is of interest
# After selection a ROI will be selected and the cell will be tracked thorugh the rest of the images
# At each tracking step we will need to get the cell coordinates and the cell size,
# distance from the EF origin (tip of electrode), from the previous image and from the initial image,
# where we selected the cell in its stable position

def calculate_rsquare(y, y_fit):
    y_mean = np.mean(y)
    y = np.array(y)
    y_fit = np.array(y_fit)
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r_square = 1 - (ss_res / ss_tot)
    return r_square

def plot_stiffness(path, cell_to_origin_list, DEP_forces_list, DEP_forces_fit_list, k, r_square):
    # Create a figure with 2 subplots (1 row, 2 columns)
    fig = plt.figure(figsize=(8, 6))

    # Scatter plot 1
    plt.scatter(cell_to_origin_list, DEP_forces_list, label="ΔX (µm) vs Force (pN)")
    plt.plot(cell_to_origin_list, DEP_forces_fit_list, label="Fit line", color='red')
    plt.title('Force (pN) vs ΔX (µm)', fontsize=22)
    plt.xlabel("ΔX (µm)", fontsize=14)
    plt.ylabel('Force (pN)', fontsize=14)

    plt.figtext(0.15, 0.8, f"Stiffness (k): {k:.2f} pN/μm, \nRsquared: {r_square:.2f}",
                fontsize=16,
                color='black')

    # Save the image
    file_path = os.path.join(path, "results.png")
    plt.savefig(file_path)

def plot_data(path, time_list, ef_gradients_list, DEP_forces_list, DEP_forces_fit_list, cell_to_origin_list, voltages_list):
    # Create a figure with 2 subplots (1 row, 2 columns)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))

    # Scatter plot 1
    ax1.scatter(time_list, ef_gradients_list, label="EF gradient in the cell")
    ax1.set_title('Squared electric field gradient (V^2/m^3)')
    ax1.set_xlabel("Time(sec)")
    ax1.set_ylabel('Electric field gradient (V^2/m^3)')

    # Scatter plot 2
    ax2.scatter(time_list, DEP_forces_list, label="DEP force in the cell (pN)")
    ax2.set_title('DEP force in the cell (pN)')
    ax2.set_xlabel("Time(sec)")
    ax2.set_ylabel('DEP force in the cell (pN)')

    # Scatter plot 3
    ax3.scatter(cell_to_origin_list, DEP_forces_list, label="Offset (µm) vs DEP force (pN)")
    ax3.plot(cell_to_origin_list, DEP_forces_fit_list, label="Fit line", color='red')
    ax3.set_title('Offset (µm) vs DEP force (pN)')
    ax3.set_xlabel("Offset (µm)")
    ax3.set_ylabel('DEP force (pN)')

    # Scatter plot 4
    ax4.scatter(cell_to_origin_list, voltages_list, label="Distance from target to cell")
    ax4.set_title('Offset (µm) vs Escaping voltage (V)')
    ax4.set_xlabel("Offset (µm)")
    ax4.set_ylabel('Escaping voltage (V)')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the image
    file_path = os.path.join(path, "results.png")
    plt.savefig(file_path)

def draw_parameters_to_image(image, radius, cell_to_target_distance, cell_to_origin_distance, ef_gradient, ef_start,
                             ef_end):
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 255, 255)
    thickness = 2

    cv2.putText(image, f"Cell radius: {radius:.2f} um", (10, 30), font, 0.75, color, thickness)
    cv2.putText(image, f"Distance from cell to target: {cell_to_target_distance:.2f} um", (10, 60), font, 0.75, color,
                thickness)
    cv2.putText(image, f"Distance from origin to cell: {cell_to_origin_distance:.2f} um", (10, 90), font, 0.75, color,
                thickness)
    cv2.putText(image, f"EF gradient: {ef_gradient:.2f} V^3/m^2", (10, 120), font, 0.75, color, thickness)
    cv2.putText(image, f"EF at the start of the cell: {ef_start:.2f} V/m", (10, 150), font, 0.75, color, thickness)
    cv2.putText(image, f"EF at the end of the cell: {ef_end:.2f} V/m", (10, 180), font, 0.75, color, thickness)

    return image


def draw_highlight_rectangle(image, top_left, bottom_right, color, transparency=0.5):
    # Create an overlay image
    overlay = image.copy()

    # Draw a filled rectangle on the overlay
    cv2.rectangle(overlay, top_left, bottom_right, color, thickness=cv2.FILLED, lineType=cv2.LINE_AA)

    # Blend the overlay with the original image
    cv2.addWeighted(overlay, transparency, image, 1 - transparency, 0, image)

    # Add rectangle border
    image = cv2.rectangle(image, top_left, bottom_right, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    return image


def draw_highlight_circle(image, center, radius, color, transparency=0.5):
    # Create an overlay image
    overlay = image.copy()

    # Draw a filled circle on the overlay
    cv2.circle(overlay, center, radius, color, thickness=cv2.FILLED)

    # Blend the overlay with the original image
    cv2.addWeighted(overlay, transparency, image, 1 - transparency, 0, image)

    # Add circle border
    image = cv2.circle(image, center, radius, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    image = cv2.circle(image, center, 3, (255, 255, 255), -1, lineType=cv2.LINE_AA)

    return image


def check_cell_at_location(cell_population_details, location_coords):
    # check if input location is on a cell area, and if yes, return the cell number, and center location
    x_coord = None
    y_coord = None
    radius = None
    cell_id = None

    for i in cell_population_details:
        x, y = i[1]
        r = i[2]
        if (location_coords[0] - x) ** 2 + (location_coords[1] - y) ** 2 <= r ** 2:
            x_coord = x
            y_coord = y
            radius = r
            cell_id = i[0]

    return cell_id, x_coord, y_coord, radius

# This is the initial detection of the cell of interest
def detect_cell_of_interest(image, min_radius_pixels, max_radius_pixels, param1=60, param2=30):
    # Some initial graphical parameters
    color = 255, 255, 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2

    # Check if the image is already in grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image = image

    # Apply a Gaussian blur to the image to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Use HoughCircles to detect circles in the image
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_radius_pixels,
                               param1=param1, param2=param2, minRadius=min_radius_pixels, maxRadius=max_radius_pixels)

    # Initialize an empty list to store cell coordinates
    cell_coords = []
    cell_radi = []

    cell_no = 1
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # append the cell coordinates and radius to the list
            cell_coords.append((x, y))
            cell_radi.append(r)
            # Draw a circle around the cell
            draw_highlight_circle(image, (x, y), r, color, transparency=0.25)
            # Add text to the image with the cell number at a position slightly above the cell perimeter
            cv2.putText(image, f"Cell {cell_no}", (x, y - r - 10), fontFace=font, fontScale=0.75, color=color,
                        thickness=thickness)
            cell_no += 1

    cell_population_details = []
    for i in range(len(cell_coords)):
        cell_details = [i + 1, cell_coords[i], cell_radi[i]]
        cell_population_details.append(cell_details)

    for i in cell_population_details:
        print(f"Cell: {i[0]} Coordinates: {i[1]}, Radius: {i[2]}")

    return cell_population_details, image


def detect_cells_in_roi(image, min_radius_pixels, max_radius_pixels, roi):
    # Load the image in grayscale
    success = True

    # Define the region of interest (ROI)
    x, y, w, h = roi
    roi_image = image[y:y + h, x:x + w]

    # Apply a Gaussian blur to the ROI to reduce noise
    blurred_roi = cv2.GaussianBlur(roi_image, (9, 9), 2)

    # Use HoughCircles to detect circles in the ROI
    circles = cv2.HoughCircles(blurred_roi, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                               param1=60, param2=30, minRadius=min_radius_pixels, maxRadius=max_radius_pixels)

    # Check if only one cell is detected
    if circles is None or len(circles[0]) != 1:
        print("Error: Adjust the ROI or change the image. Detected cells count:",
              0 if circles is None else len(circles[0]))
        success = False
        return None, None, image, success

    # Get the coordinates and radius of the detected cell and convert them to the original image coordinates
    circle = np.round(circles[0, 0]).astype("int")
    x_center, y_center, radius = circle
    cell_adjusted_coords = (x + x_center, y + y_center)  # Adjust coordinates to original image

    return cell_adjusted_coords, radius, image, success


def calculate_distances(cell_coords, origin_coords, target_coords, radius):
    # Calculate the distance from the origin to the cell position and from the cell to the target position
    cell_to_origin = np.sqrt((cell_coords[0] - origin_coords[0]) ** 2 + (cell_coords[1] - origin_coords[1]) ** 2)
    cell_to_target = np.sqrt((target_coords[0] - cell_coords[0]) ** 2 + (target_coords[1] - cell_coords[1]) ** 2)

    target_to_cell_end = cell_to_target + radius
    target_to_cell_start = cell_to_target - radius

    return cell_to_origin, cell_to_target, target_to_cell_end, target_to_cell_start


def EF_conversion_function(distance_from_target):
    # Calculate the electric field (EF) with the polynomial : electric_field = 471.76e^-6E-04*distance_from_target
    electric_field = -0.05 * distance_from_target ** 3 + 6.6909 * distance_from_target ** 2 - 310.27 * distance_from_target + 9302.7
    return electric_field


def calculate_EF_gradient(cell_start_to_target, cell_end_to_target, voltage=0.5):
    # Calculate the electric field (EF) at the start and end of the cell
    EF_start = EF_conversion_function(cell_start_to_target) * voltage
    EF_end = EF_conversion_function(cell_end_to_target) * voltage

    # Calulate the gradient of the squared electric field
    squared_EF_start = EF_start ** 2
    squared_EF_end = EF_end ** 2

    distance_um = cell_end_to_target - cell_start_to_target
    distance_m = distance_um / 1e6

    EF_gradient = (squared_EF_start - squared_EF_end) / distance_m
    EF_gradient = abs(EF_gradient)

    return EF_gradient, EF_start, EF_end


def calculate_EF_gradient_avg(cell_start_to_target, cell_end_to_target, split_points=100):
    # Calculate the electric field (EF) at the start and end of the cell
    values = np.linspace(cell_start_to_target, cell_end_to_target, split_points)
    local_EF_gradients = []
    for i in range(len(values) - 1):
        EF_start_local = EF_conversion_function(values[i])
        EF_end_local = EF_conversion_function(values[i + 1])

        # Calulate the gradient of the squared electric field
        squared_EF_start = EF_start_local ** 2
        squared_EF_end = EF_end_local ** 2

        distance_um = values[i + 1] - values[i]
        distance_m = distance_um / 1e6

        EF_gradient_local = (squared_EF_start - squared_EF_end) / distance_m
        EF_gradient_local = abs(EF_gradient_local)

        local_EF_gradients.append(EF_gradient_local)

    avg_EF_gradient = np.mean(local_EF_gradients)
    print(local_EF_gradients)

    EF_start = EF_conversion_function(cell_start_to_target)
    EF_end = EF_conversion_function(cell_end_to_target)
    return avg_EF_gradient, EF_start, EF_end


def calculate_DEP_force(cell_radius, buffer_permittivity, CM_factor, EF_gradient):
    # Convert buffer permittivity to absolute permittivity
    buffer_permittivity_abs = buffer_permittivity * 8.854e-12

    # Convert radius to meters
    cell_radius = cell_radius * 1e-6

    # Calculate the DEP force with the formula: F_DEP = 2 * pi * radius^3 * buffer_permittivity * CM_factor * EF_gradient
    DEP_force = 2 * np.pi * cell_radius ** 3 * buffer_permittivity_abs * CM_factor * EF_gradient

    # Convert force to picoNewtons
    DEP_force = DEP_force * 1e12

    return DEP_force


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Mouse click at position: ({x}, {y})")


def compute_voltage_ramping(folder_path,
                            min_radius_microns=5,
                            max_radius_microns=20,
                            target_coords_microns=(166, 143),
                            roi_size_microns=(50, 50),
                            microns_per_pixel=0.1923,
                            voltage_ramp=0.25,
                            frames_per_voltage=4,
                            start_voltage=0.5
                            ):
    # Initialize the lists for cell parameters per image
    efs_end_list = []
    efs_start_list = []
    ef_gradients_list = []
    cell_to_target_list = []
    cell_to_origin_list = []
    radi_list = []
    voltages_list = []

    # Create processed images folder
    processed_folder = os.path.join(folder_path, "_processed")
    if not os.path.exists(processed_folder):
        os.makedirs(processed_folder)

    # Convert parameters from microns to pixels
    min_radius_pixels = int(min_radius_microns / microns_per_pixel)
    max_radius_pixels = int(max_radius_microns / microns_per_pixel)
    target_coords_pixels = (int(target_coords_microns[0] / microns_per_pixel),
                            int(target_coords_microns[1] / microns_per_pixel))
    roi_size_pixels = (int(roi_size_microns[0] / microns_per_pixel),
                       int(roi_size_microns[1] / microns_per_pixel))

    # Load the first image, check if it is a valid image file and get the path, if no image is found raise an error
    image_path = None
    for file in os.listdir(folder_path):
        if file.endswith(".tif") or file.endswith(".png") or file.endswith(".jpg"):
            image_path = os.path.join(folder_path, file)
            break
    if image_path is None:
        raise FileNotFoundError(f"No image found in folder: {folder_path}")

    else:
        # Load the first image as grayscale and detect the cell of interest
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        cell_population_details, labeled_image = detect_cell_of_interest(image, min_radius_pixels, max_radius_pixels)

        # Display the image with the detected cell
        cv2.namedWindow('Resizable Window', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Resizable Window', mouse_callback)
        cv2.imshow('Resizable Window', labeled_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Ask the user to input the cell number to track
        cell_no = input("Enter the cell number to track: ")
        cell_no = int(cell_no)

        # Define the region of interest (ROI) for the cell tracking
        x = cell_population_details[cell_no - 1][1][0] - roi_size_pixels[0] // 2
        y = cell_population_details[cell_no - 1][1][1] - roi_size_pixels[1] // 2
        w = roi_size_pixels[0]
        h = roi_size_pixels[1]

        # Define the origin coordinates in pixels
        origin_coords_pixels = (cell_population_details[cell_no - 1][1][0], cell_population_details[cell_no - 1][1][1])

        iter_no = 0
        # Loop through the images in the folder
        for file in os.listdir(folder_path):
            if not file.endswith(".tif") and not file.endswith(".png") and not file.endswith(".jpg"):
                continue
            # Load the iterated image
            image_path_iter = os.path.join(folder_path, file)
            image_iter = cv2.imread(image_path_iter, cv2.IMREAD_GRAYSCALE)

            # Get the coordinates and radius of the detected cell in the ROI
            roi = (x, y, w, h)
            cell_coords, radius, image_iter, success = detect_cells_in_roi(image_iter, min_radius_pixels,
                                                                           max_radius_pixels, roi)
            if not success:
                continue

            # Calculate the distance from the origin to the cell position and radius
            origin_to_cell, cell_to_target, target_to_cell_end, target_to_cell_start = calculate_distances(cell_coords,
                                                                                                           origin_coords_pixels,
                                                                                                           target_coords_pixels,
                                                                                                           radius)

            # if cell distance is longer than 1.5 times the diameter of the cell, break the loop
            if origin_to_cell > 2 * radius:
                break

            # Calculate the voltage for the current image
            voltage = start_voltage + (iter_no // frames_per_voltage) * voltage_ramp
            iter_no += 1

            # Calculate the electric field (EF) gradient
            ef_gradient, ef_start, ef_end = calculate_EF_gradient(target_to_cell_start * microns_per_pixel,
                                                                  target_to_cell_end * microns_per_pixel, voltage)

            # Display the ROI with the detected cell
            roi_image = draw_highlight_rectangle(image_iter, (x, y), (x + w, y + h), (255, 255, 255), transparency=0.1)
            roi_image = draw_highlight_circle(roi_image, cell_coords, radius, (255, 255, 255), transparency=0.25)

            # Draw the initial position of cell
            roi_image = draw_highlight_circle(roi_image, origin_coords_pixels, 3, (255, 255, 255), transparency=0.25)

            # Draw the target position
            roi_image = draw_highlight_circle(roi_image, target_coords_pixels, 3, (255, 255, 255), transparency=0.25)

            # Draw the parameters on the image
            image_iter = draw_parameters_to_image(roi_image, radius * microns_per_pixel,
                                                  cell_to_target * microns_per_pixel,
                                                  origin_to_cell * microns_per_pixel, ef_gradient, ef_start, ef_end)

            # Save image as processed with a text added before suffix -processed
            processed_image_path = os.path.join(processed_folder, file.replace(".", "-processed."))
            cv2.imwrite(processed_image_path, image_iter)

            # Append the calculated parameters to the lists
            efs_end_list.append(ef_end)
            efs_start_list.append(ef_start)
            ef_gradients_list.append(ef_gradient)
            cell_to_target_list.append(cell_to_target * microns_per_pixel)
            cell_to_origin_list.append(origin_to_cell * microns_per_pixel)
            radi_list.append(radius * microns_per_pixel)
            voltages_list.append(voltage)

            print(f"Image: {file}")

    return efs_end_list, efs_start_list, ef_gradients_list, cell_to_target_list, cell_to_origin_list, radi_list, voltages_list


def compute_voltage_ramping_from_ui(folder_path,
                                    min_radius_microns=5,
                                    max_radius_microns=20,
                                    param1=60,
                                    param2=30,
                                    target_coords_pixels=(0, 0),
                                    origin_coords_pixels=(0, 0),
                                    roi_size_microns=(50, 50),
                                    microns_per_pixel=0.1923,
                                    voltage_incr=0.25,
                                    frames_per_voltage=4,
                                    frames_per_second=2,
                                    start_voltage=0.5,
                                    cell_radius_threshold=2,
                                    cm_factor=1,
                                    buffer_permittivity=78
                                    ):

    # Initialize the lists for cell parameters per image
    efs_end_list = []
    efs_start_list = []
    ef_gradients_list = []
    cell_to_target_list = []
    cell_to_origin_list = []
    radi_list = []
    voltages_list = []
    DEP_forces_list = []
    DEP_forces_fit_list = []
    time_list = []

    # Create processed images folder
    processed_folder = os.path.join(folder_path, "_processed")
    if not os.path.exists(processed_folder):
        os.makedirs(processed_folder)

    data_folder = os.path.join(folder_path, "_data")
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    # Convert parameters from microns to pixels
    min_radius_pixels = int(min_radius_microns / microns_per_pixel)
    max_radius_pixels = int(max_radius_microns / microns_per_pixel)
    roi_size_pixels = (int(roi_size_microns[0] / microns_per_pixel),
                       int(roi_size_microns[1] / microns_per_pixel))

    # Define the region of interest (ROI) for the cell tracking
    x = origin_coords_pixels[0] - roi_size_pixels[0] // 2
    y = origin_coords_pixels[1] - roi_size_pixels[1] // 2
    w = roi_size_pixels[0]
    h = roi_size_pixels[1]

    iter_no = 0
    # Loop through the images in the folder
    for file in os.listdir(folder_path):
        if not file.endswith(".tif") and not file.endswith(".png") and not file.endswith(".jpg"):
            continue
        # Load the iterated image
        image_path_iter = os.path.join(folder_path, file)
        image_iter = cv2.imread(image_path_iter, cv2.IMREAD_GRAYSCALE)

        # Get the coordinates and radius of the detected cell in the ROI
        roi = (x, y, w, h)
        cell_coords, radius, image_iter, success = detect_cells_in_roi(image_iter,
                                                                       min_radius_pixels,
                                                                       max_radius_pixels,
                                                                       roi)
        if not success:
            continue

        # Calculate the distance from the origin to the cell position and radius
        origin_to_cell, cell_to_target, target_to_cell_end, target_to_cell_start = calculate_distances(cell_coords,
                                                                                                       origin_coords_pixels,
                                                                                                       target_coords_pixels,
                                                                                                       radius)

        # if cell distance is longer than set times the diameter of the cell, break the loop
        if origin_to_cell > cell_radius_threshold * radius:
            break

        # Calculate the voltage for the current image
        voltage = start_voltage + (iter_no // frames_per_voltage) * voltage_incr
        iter_no += 1

        # Calculate the electric field (EF) gradient
        ef_gradient, ef_start, ef_end = calculate_EF_gradient(target_to_cell_start * microns_per_pixel,
                                                              target_to_cell_end * microns_per_pixel, voltage)

        # Display the ROI with the detected cell
        roi_image = draw_highlight_rectangle(image_iter, (x, y), (x + w, y + h), (255, 255, 255), transparency=0.1)
        roi_image = draw_highlight_circle(roi_image, cell_coords, radius, (255, 255, 255), transparency=0.25)

        # Draw the initial position of cell
        roi_image = draw_highlight_circle(roi_image, origin_coords_pixels, 3, (255, 255, 255), transparency=0.25)

        # Draw the target position
        roi_image = draw_highlight_circle(roi_image, target_coords_pixels, 3, (255, 255, 255), transparency=0.25)

        # Draw the parameters on the image
        image_iter = draw_parameters_to_image(roi_image, radius * microns_per_pixel,
                                              cell_to_target * microns_per_pixel,
                                              origin_to_cell * microns_per_pixel, ef_gradient, ef_start, ef_end)

        # Save image as processed with a text added before suffix -processed
        processed_image_path = os.path.join(processed_folder, file.replace(".", "-processed."))
        cv2.imwrite(processed_image_path, image_iter)

        # Calculate time for each image
        time = iter_no / frames_per_second

        # Append the calculated parameters to the lists
        time_list.append(time)
        efs_end_list.append(ef_end)
        efs_start_list.append(ef_start)
        ef_gradients_list.append(ef_gradient)
        cell_to_target_list.append(cell_to_target * microns_per_pixel)
        cell_to_origin_list.append(origin_to_cell * microns_per_pixel)
        radi_list.append(radius * microns_per_pixel)
        voltages_list.append(voltage)
        print(f"Image: {file}")

    # Calculate DEP forces
    for i in range(len(efs_end_list)):
        DEP_force = calculate_DEP_force(radi_list[i], buffer_permittivity, cm_factor, ef_gradients_list[i])
        DEP_forces_list.append(DEP_force)

    # Fit offset vs DEP force to a linear function
    def hookes_law(x, k):
        return k * x

    popt, pcov = curve_fit(hookes_law, cell_to_origin_list, DEP_forces_list)
    k = popt[0]
    for i in cell_to_origin_list:
        DEP_forces_fit_list.append(i * float(k))

    r_square = calculate_rsquare(DEP_forces_list, DEP_forces_fit_list)
    print(f"Stiffness (k): {k} pN/μm, Rsquared: {r_square}")

    # Plot the data
    #plot_data(data_folder, time_list, ef_gradients_list, DEP_forces_list, DEP_forces_fit_list, cell_to_origin_list, voltages_list)
    plot_stiffness(data_folder, cell_to_origin_list, DEP_forces_list, DEP_forces_fit_list, k, r_square)

    # Calculate the average and stdev of radius of particle
    avg_particle_radius = np.mean(radi_list)
    stdev_particle_radius = np.std(radi_list)
    # Calculate the escaping voltage
    escaping_voltage = voltages_list[-1]
    # Calculate the escaping force
    escaping_force = DEP_forces_list[-1]

    # Save the results to a csv file
    results_file = os.path.join(data_folder, "results.csv")
    if os.path.exists(results_file):
        os.remove(results_file)

    with open(results_file, "w") as file:

        file.write(f"Average particle radius (µm), {avg_particle_radius}, {stdev_particle_radius}\n")
        file.write(f"Escaping voltage (V), {escaping_voltage}\n")
        file.write(f"Escaping force (pN), {escaping_force}\n")
        file.write(f"Stiffness (pN/µm), {k}\n")
        file.write(f"Rsquared, {r_square}\n")
        file.write("\n\n")

        file.write("Time (s), Voltage (V), EF gradient (V^2/m^3), DEP force (pN), Offset (µm)\n")
        for i in range(len(time_list)):
            file.write(f"{time_list[i]}, {voltages_list[i]}, {ef_gradients_list[i]}, {DEP_forces_list[i]}, {cell_to_origin_list[i]}\n")

    return DEP_forces_list, efs_end_list, efs_start_list, ef_gradients_list, voltages_list, cell_to_target_list, cell_to_origin_list, time_list