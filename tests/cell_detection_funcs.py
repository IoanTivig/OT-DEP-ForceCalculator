import cv2
import numpy as np

# First, we need to detect the cell of intrest which will remain the same thorugh all the rest of the images
# We will need to detect the cells and select the one that is of interest
# After selection a ROI will be selected and the cell will be tracked thorugh the rest of the images
# At each tracking step we will need to get the cell coordinates and the cell size, distance from the EF origin (tip of electrode),
# from the previous image and from the initial image, where we selected the cell in its stable position



# This is the initial detection of the cell of interest
def detect_cell_of_intrst(image, min_radius_pixels, max_radius_pixels):
    # Check if the image is already in grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image = image

    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Apply a Gaussian blur to the image to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Use HoughCircles to detect circles in the image
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                               param1=50, param2=30, minRadius=min_radius_pixels, maxRadius=max_radius_pixels)

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
            cv2.circle(image, (x, y), r, (0, 255, 0), 2)
            # Draw a small circle (centroid) in the center of the detected circle
            cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
            # Add text to the image with the cell number
            cv2.putText(image, f"Cell {cell_no}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return cell_coords, cell_radi, image


def detect_cells(image, min_radius_pixels, max_radius_pixels):
    # Check if the image is already in grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image = image

    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Apply a Gaussian blur to the image to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Use HoughCircles to detect circles in the image
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                               param1=50, param2=30, minRadius=min_radius_pixels, maxRadius=max_radius_pixels)

    # Initialize an empty list to store cell coordinates
    cell_coords = []
    cell_radi = []

    # If circles are detected
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cell_coords.append((x, y))
            # Draw the circle in the output image
            cv2.circle(image, (x, y), r, (0, 255, 0), 2)
            # Draw a small circle (centroid) in the center of the detected circle
            cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
            cell_radi.append(r)

    return cell_coords, cell_radi, image, len(cell_coords)

def calculate_start_end_points(cell_coords, orientation='vertical'):
    if orientation == 'vertical-up' or orientation == 'vertical-down':
        start_point = (cell_coords[0][0], cell_coords[0][1] + cell_radi[0])
        end_point = (cell_coords[-1][0], cell_coords[-1][1] - cell_radi[-1])

    elif orientation == 'horizontal-left' or orientation == 'horizontal-right':
        start_point = (cell_coords[0][0] - cell_radi[0], cell_coords[0][1])
        end_point = (cell_coords[-1][0] + cell_radi[-1], cell_coords[-1][1])

    return start_point, end_point

def mark_cell_movement_axis(image, origin_coords, axis_length_microns=100, orientation='vertical'):
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    cv2.circle(image, (origin_coords[0], origin_coords[1]), 5, (75, 75, 75), -1)
    if orientation == 'vertical-up':
        cv2.line(image, (origin_coords[0], origin_coords[1]), (origin_coords[0], origin_coords[1] + axis_length_microns), (75, 75, 75), 2)

    elif orientation == 'horizontal-left':
        cv2.line(image, (origin_coords[0], origin_coords[1]), (origin_coords[0] + axis_length_microns, origin_coords[1]), (75, 75, 75), 2)

    elif orientation == 'vertical-down':
        cv2.line(image, (origin_coords[0], origin_coords[1]), (origin_coords[0], origin_coords[1] - axis_length_microns), (75, 75, 75), 2)

    elif orientation == 'horizontal-right':
        cv2.line(image, (origin_coords[0], origin_coords[1]), (origin_coords[0] - axis_length_microns, origin_coords[1]), (75, 75, 75), 2)

    # Add text to the image with the origin coordinates
    return image


# mark_cell_boundary
def mark_cell_boundary(image, cell_coords, cell_radi, orientation='vertical'):
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    if orientation == 'vertical-up' or orientation == 'vertical-down':
        for cell in cell_coords:
            # add vertical end and start line
            #cv2.line(image, (cell[0]+cell_radi[cell_coords.index(cell)]*2, cell[1] - cell_radi[cell_coords.index(cell)]), (cell[0] + cell_radi[cell_coords.index(cell)]*2, cell[1] + cell_radi[cell_coords.index(cell)]), (0, 255, 0), 2)
            # add horizontal end and start lines
            cv2.line(image, (cell[0] - cell_radi[cell_coords.index(cell)], cell[1] - cell_radi[cell_coords.index(cell)]), (cell[0] + cell_radi[cell_coords.index(cell)], cell[1] - cell_radi[cell_coords.index(cell)]), (0, 255, 0), 2)
            cv2.line(image, (cell[0] - cell_radi[cell_coords.index(cell)], cell[1] + cell_radi[cell_coords.index(cell)]), (cell[0] + cell_radi[cell_coords.index(cell)], cell[1] + cell_radi[cell_coords.index(cell)]), (0, 255, 0), 2)


    elif orientation == 'horizontal-left' or orientation == 'horizontal-right':
        for cell in cell_coords:
            cv2.line(image, (cell[0] - cell_radi[cell_coords.index(cell)], cell[1]), (cell[0] + cell_radi[cell_coords.index(cell)], cell[1]), (0, 255, 0), 2)

    return image

def add_cells_details(image, cell_coords, cell_radi, microns_per_pixel):


    return image

# Convert the radius from microns to pixels
# 100 microns = 520 pixels on 63x magnification Zeiss OT microscope
# 1 micron = 5.2 pixels
# 1 pixel = 0.1923 microns

microns_per_pixel = 0.1923

# Parameters
min_radius_microns = 5
max_radius_microns = 20

axis_length_microns = 40
origin_coords_microns = (166, 143)

orientation = 'vertical-up'

# Convert the radius from microns to pixels
min_radius_pixels = int(min_radius_microns / microns_per_pixel)
max_radius_pixels = int(max_radius_microns / microns_per_pixel)

axis_length_pixels = int(axis_length_microns / microns_per_pixel)
origin_coords_pixels = (int(origin_coords_microns[0] / microns_per_pixel),
                        int(origin_coords_microns[1] / microns_per_pixel))

# Example usage
image_path = 'DEP-OT test sample 1 - origin.tif'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
cell_coords, cell_radi, labeled_image, num_cells = detect_cells(image=image,
                                                     min_radius_pixels=min_radius_pixels,
                                                     max_radius_pixels=max_radius_pixels)



image_with_axis = mark_cell_movement_axis(image=labeled_image,
                                          origin_coords=origin_coords_pixels,
                                          axis_length_microns=axis_length_pixels,
                                          orientation=orientation)

image_with_details = mark_cell_boundary(image=image_with_axis,
                                        cell_coords=cell_coords,
                                        cell_radi=cell_radi,
                                        orientation=orientation)

# Calculate the distance between the origin and the detected cells
for cell in cell_coords:
    distance = np.sqrt((cell[0] - origin_coords_pixels[0])**2 + (cell[1] - origin_coords_pixels[1])**2)
    print(f"Distance from origin to cell: {distance*microns_per_pixel} um")

# Print position of cell and size in microns
for cell in cell_coords:
    print(f"Cell at: {cell[0]*microns_per_pixel} um, {cell[1]*microns_per_pixel} um")
    print(f"Cell size: {cell_radi[cell_coords.index(cell)]*microns_per_pixel} um")


# Display the labeled image
cv2.imshow('Labeled Image', image_with_details)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the output image if needed
#cv2.imwrite('electrode_tip_detected.png', image_with_axis)