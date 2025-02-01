from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QTableWidgetItem
from PyQt5.uic import loadUi

from src.funcs.cell_detection_funcs import *
from src.funcs.other import *
from ui.resources.graphical_resources import *

'''
OpenDEP Force Calculator
    Copyright (C) 2024  Ioan Cristian Tivig

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

    You can contact the developer/owner of OpenDEP at "ioan.tivig@gmail.com".
'''


class MainUI(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        loadUi("ui/main.ui", self)
        self.setWindowTitle("OpenDEP Force Calculator")
        self.setWindowIcon(QIcon("icon.png"))

        # Initial values and ui elements
        self.target_selection = True
        self.pyqt5_label_selection_text.setText("Select target")
        self.target_coordinates = None
        self.origin_coordinates = None

        self.labeled_image_original = None
        self.labeled_image = None
        self.cell_population_details = None

        # Connect the buttons to the functions
        self.pyqt5_button_loadfolderpath.clicked.connect(self.open_path)
        self.pyqt5_button_main_results_load.clicked.connect(self.open_main_results_path)
        self.pyqt5_button_replace_results_load.clicked.connect(self.open_replace_results_path)
        self.pyqt5_button_loadloadimage.clicked.connect(self.load_cv2_image)
        self.pyqt5_button_select_target.clicked.connect(self.select_target)
        self.pyqt5_button_select_cell.clicked.connect(self.select_cell)
        self.pyqt5_button_refresh_parameters.clicked.connect(self.load_cv2_image)
        self.pyqt5_button_compute_stiffness.clicked.connect(self.process_folder_stiffness)
        self.pyqt5_button_compute_spectra.clicked.connect(self.process_folder_frequency)
        self.pyqt5_button_combine_spectras.clicked.connect(self.combine_dep_results)
        self.pyqt5_button_compute_flowbased_stiffness.clicked.connect(self.process_folder_stiffness_flow_based)

    def open_path(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        self.pyqt5_entry_folderpath.setText(folder_path)

    def open_main_results_path(self):
        file_path = QFileDialog.getOpenFileName(self, "Select File")
        self.pyqt5_entry_main_results_path.setText(file_path[0])

    def open_replace_results_path(self):
        file_path = QFileDialog.getOpenFileName(self, "Select File")
        self.pyqt5_entry_replace_results_path.setText(file_path[0])

    def select_target(self):
        self.target_selection = True
        self.pyqt5_label_selection_text.setText("Select target")

    def select_cell(self):
        self.target_selection = False
        self.pyqt5_label_selection_text.setText("Select cell")

    def refresh_target_and_roi(self):
        self.labeled_image = self.labeled_image_original.copy()
        if self.origin_coordinates is not None:
            # Add ROI to the image
            height = int(int(self.pyqt5_entry_roisize_h.text()) / float(self.pyqt5_entry_micronspixelration.text()))
            width = int(int(self.pyqt5_entry_roisize_w.text()) / float(self.pyqt5_entry_micronspixelration.text()))
            print(f"ROI size: {width}x{height}")
            left_top_corner = (
            int(self.origin_coordinates[0] - width / 2), int(self.origin_coordinates[1] - height / 2))
            right_bottom_corner = (
            int(self.origin_coordinates[0] + width / 2), int(self.origin_coordinates[1] + height / 2))

            self.labeled_image = draw_highlight_rectangle(self.labeled_image,
                                                          left_top_corner,
                                                          right_bottom_corner,
                                                          (255, 255, 255),
                                                          transparency=0.1)

        if self.target_coordinates is not None:
            # Add target to the image
            self.labeled_image = cv2.circle(self.labeled_image, self.target_coordinates, 5, (255, 255, 255), -1,
                                            lineType=cv2.LINE_AA)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.target_selection:
                print(f"Target click at position: ({x}, {y})")
                self.pyqt5_entry_targetlocation_x.setText(str(x))
                self.pyqt5_entry_targetlocation_y.setText(str(y))
                self.target_coordinates = (x, y)

            else:
                print(f"Cell click at position: ({x}, {y})")
                # Verify if the click is inside the cell
                click_location = (x, y)
                cell_id, cell_x, cell_y, cell_radius = check_cell_at_location(self.cell_population_details,
                                                                              click_location)
                if cell_x is None:
                    print("No cell found at this location")
                else:
                    print(f"Cell {cell_id} found at position: ({cell_x}, {cell_y}) with radius: {cell_radius} pixels")
                    self.pyqt5_label_cell_id.setText(str(cell_id))
                    self.pyqt5_label_cell_x.setText(str(cell_x))
                    self.pyqt5_label_cell_y.setText(str(cell_y))
                    self.origin_coordinates = (cell_x, cell_y)

            self.refresh_target_and_roi()
            cv2.imshow('Cell Detection Window', self.labeled_image)

    def load_cv2_image(self):
        folder_path = self.pyqt5_entry_folderpath.text()
        for file in os.listdir(folder_path):
            if file.startswith("_baseline"):
                if file.endswith(".tif") or file.endswith(".png") or file.endswith(".jpg"):
                    image_path = os.path.join(folder_path, file)
                    break

        if image_path is None:
            raise FileNotFoundError(f"No image found in folder: {folder_path}")

        else:
            # Get the parameters from the UI
            microns_per_pixel = float(self.pyqt5_entry_micronspixelration.text())
            min_radius_microns = float(self.pyqt5_entry_particlesize_min.text())
            max_radius_microns = float(self.pyqt5_entry_particlesize_max.text())

            # Convert the parameters to pixels
            min_radius_pixels = int(min_radius_microns / microns_per_pixel)
            max_radius_pixels = int(max_radius_microns / microns_per_pixel)
            distance_particles_pixels = int(float(self.pyqt5_entry_mindistparticles.text()) / microns_per_pixel)

            # Load the first image as grayscale and detect the cell of interest
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            param1 = int(self.pyqt5_entry_param1.text())
            param2 = int(self.pyqt5_entry_param2.text())

            self.cell_population_details, self.labeled_image_original = detect_cell_of_interest(image,
                                                                                                distance_particles_pixels,
                                                                                                min_radius_pixels,
                                                                                                max_radius_pixels,
                                                                                                param1=param1,
                                                                                                param2=param2)

            # Mark target on cv2 image
            self.labeled_image = self.labeled_image_original.copy()
            self.labeled_image = cv2.circle(self.labeled_image, self.target_coordinates, 5, (255, 255, 255), -1,
                                            lineType=cv2.LINE_AA)
            self.refresh_target_and_roi()

            # Display the image with the detected cell
            cv2.namedWindow('Cell Detection Window', cv2.WINDOW_NORMAL)
            cv2.setMouseCallback('Cell Detection Window', self.mouse_callback)
            cv2.imshow('Cell Detection Window', self.labeled_image)

    def process_folder_stiffness(self):
        compute_voltage_ramping_from_ui(
            folder_path=self.pyqt5_entry_folderpath.text(),
            distance_particles_microns=float(self.pyqt5_entry_mindistparticles.text()),
            min_radius_microns=float(self.pyqt5_entry_particlesize_min.text()),
            max_radius_microns=float(self.pyqt5_entry_particlesize_max.text()),
            param1=int(self.pyqt5_entry_param1.text()),
            param2=int(self.pyqt5_entry_param2.text()),
            target_coords_pixels=self.target_coordinates,
            origin_coords_pixels=self.origin_coordinates,
            roi_size_microns=(int(self.pyqt5_entry_roisize_w.text()), int(self.pyqt5_entry_roisize_h.text())),
            microns_per_pixel=float(self.pyqt5_entry_micronspixelration.text()),
            ef_model=self.pyqt5_combo_ef_formula.currentIndex(),
            distance_from_surface_source=self.pyqt5_combo_distfromsurfacesource.currentIndex(),
            distance_from_surface_microns=float(self.pyqt5_entry_distfromsurface.text()),
            voltage_incr=float(self.pyqt5_entry_vpp_increment.text()),
            frames_per_voltage=int(self.pyqt5_entry_frames_vpp_increment.text()),
            frames_per_second=int(self.pyqt5_entry_frames_per_second.text()),
            start_voltage=float(self.pyqt5_entry_vpp_start.text()),
            min_threshold=float(self.pyqt5_entry_min_thereshold.text()),
            max_threshold=float(self.pyqt5_entry_max_threshold.text()),
            cm_factor=float(self.pyqt5_entry_cm_factor.text()),
            buffer_permittivity=float(self.pyqt5_entry_buffer_perm.text()),
        )

    def process_folder_stiffness_flow_based(self):
        compute_flow_ramping_from_ui(
            folder_path=self.pyqt5_entry_folderpath.text(),
            distance_particles_microns=float(self.pyqt5_entry_mindistparticles.text()),
            min_radius_microns=float(self.pyqt5_entry_particlesize_min.text()),
            max_radius_microns=float(self.pyqt5_entry_particlesize_max.text()),
            param1=int(self.pyqt5_entry_param1.text()),
            param2=int(self.pyqt5_entry_param2.text()),
            origin_coords_pixels=self.origin_coordinates,
            roi_size_microns=(int(self.pyqt5_entry_roisize_w.text()), int(self.pyqt5_entry_roisize_h.text())),
            microns_per_pixel=float(self.pyqt5_entry_micronspixelration.text()),
            distance_from_surface_source=self.pyqt5_combo_flow_distfromsurfacesource.currentIndex(),
            distance_from_surface_microns=float(self.pyqt5_entry_flow_distfromsurface.text()),
            channel_width_mm=float(self.pyqt5_entry_flow_channel_width.text()),
            channel_height_mm=float(self.pyqt5_entry_flow_channel_height.text()),
            particle_offset_microns=float(self.pyqt5_entry_flow_particle_centeroffset.text()),
            fluid_dynamic_viscosity=float(self.pyqt5_entry_flow_liquid_viscosity.text()),
            near_wall_correction=self.pyqt5_combo_flow_stiffness_correction.currentIndex(),
            flow_rate_incr=float(self.pyqt5_entry_flow_ramping_rate_incr.text()),
            flow_rate_start=float(self.pyqt5_entry_flow_ramping_rate_start.text()),
            min_threshold=float(self.pyqt5_entry_flow_ramping_min_thershold.text()),
            max_threshold=float(self.pyqt5_entry_flow_ramping_max_thershold.text()),
        )

    def process_folder_frequency(self):
        compute_frequency_ramping_from_ui(
            folder_path=self.pyqt5_entry_folderpath.text(),
            distance_particles_microns=float(self.pyqt5_entry_mindistparticles.text()),
            min_radius_microns=float(self.pyqt5_entry_particlesize_min.text()),
            max_radius_microns=float(self.pyqt5_entry_particlesize_max.text()),
            param1=int(self.pyqt5_entry_param1.text()),
            param2=int(self.pyqt5_entry_param2.text()),
            target_coords_pixels=self.target_coordinates,
            origin_coords_pixels=self.origin_coordinates,
            roi_size_microns=(int(self.pyqt5_entry_roisize_w.text()), int(self.pyqt5_entry_roisize_h.text())),
            microns_per_pixel=0.1923,
            voltage=float(self.pyqt5_entry_dep_vpp.text()),
        )

    def combine_dep_results(self):
        combine_dep_spectras(path_one=self.pyqt5_entry_main_results_path.text(),
                             path_two=self.pyqt5_entry_replace_results_path.text()
                             )

    def closeEvent(self, event):
        cv2.destroyAllWindows()
        event.accept()


