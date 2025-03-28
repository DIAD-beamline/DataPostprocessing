import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.widgets import Button
from matplotlib.ticker import FuncFormatter, MultipleLocator
import matplotlib.cm as cm
#%matplotlib widget

from ipyfilechooser import FileChooser
import h5py
import os, sys, glob
import pandas as pd
import ipympl
from matplotlib.patches import Rectangle

from scipy.optimize import curve_fit
from scipy.optimize import minimize
from matplotlib.widgets import RangeSlider
from matplotlib.widgets import RangeSlider, TextBox
from ipywidgets import FloatText, Layout, HBox, VBox

class DiffractionDataAnalysis:
    def __init__(self, diff_path=None, img_path=None, out_path=None, projection_index=0, type = 'Histogram'):
        self.type = type # alternative options 'Histogram' or'Peak'
        
        self.VisitPath = "/dls/k11/data"
        self.Dif_Path = diff_path
        self.Img_Path = img_path
        self.Out_Path = out_path
        self.indx = projection_index
        
        # Diffraction profile data
        self.Ivals = None
        self.qvs2t = None
        self.qvals = None
        self.dim = None
        self.kbx = None
        self.kby = None
        self.theta = None
        
        # Imaging data
        self.proj = None
        
        # Peak analysis
        self.pk_Area = []
        self.pk_Mean = []
        self.pk_FWHM = []
        self.pk_popt = []
        
        # Image properties
        self.pixel_size = 0.54
        self.binning = 1
        self.x_range_img = 2560
        self.y_range_img = 2160
        self.aspect_ratio = 2560.0 / 2160.0
        self.selection_range = 10
        
        # Figure setup
        self.fig_width = 10
        self.img_height = None
        self.fig_height = None
        self.fig = None
        self.gs = None
        self.ax1 = None
        self.ax2 = None
        self.ax3 = None
        self.save_img_button_all = None
        self.save_img_button_sel = None
        
        # Data arrays
        self.img_array = None
        self.temp = []
        self.scatter_plots_mean = []
        self.scatter_plots_area = []
        self.scatter_plots_fwhm = []
        self.xrd = []
        
        # Placeholder
        self.test = None

    def load_diffraction(self, chooser):
        if chooser.selected:
            self.Dif_Path = chooser.selected
    
    def load_imaging(self, chooser):
        if chooser.selected:
            self.Img_Path = chooser.selected
    
    def load_output(self, chooser):
        if chooser.selected:
            self.Out_Path = chooser.selected

    def import_diffraction_data(self):
        with h5py.File(self.Dif_Path,'r') as f:
            self.Ivals=f['processed/result/data'][()]
            self.dim=len(self.Ivals.shape)-1
            
            if 'processed/result/q' in f:
                self.qvs2t="Scattering Momentum"
                self.qvals=f['processed/result/q'][()]
            elif 'processed/result/2-theta' in f:
                self.qvs2t="2-Theta Angle"
                self.qvals=f['processed/result/2-theta'][()]

            if 'processed/result/kb_cs_x' in f:
                self.kbx = f['processed/result/kb_cs_x'][()]
            elif 'entry/diffraction/kb_cs_x' in f:
                self.kbx = f['entry/diffraction/kb_cs_x'][()]
            
            if 'processed/result/kb_cs_y' in f:
                self.kby = f['processed/result/kb_cs_y'][()]
            elif 'entry/diffraction/kb_cs_y' in f:
                self.kby = f['entry/diffraction/kb_cs_y'][()]
            
            if 'processed/result/gts_theta' in f:
                self.theta = round(f['processed/result/gts_theta'][()].max(),2)
            elif 'entry/diffraction_sum/gts_theta' in f:
                self.theta = round(f['entry/diffraction_sum/gts_theta'][()].max(),2)
    
    def import_imaging_data(self):
        with h5py.File(self.Img_Path,'r') as f:
            if 'entry/input_data/tomo/rotation_angle' in f:
                #Assumed Tomography Reconstruction Data
                self.indx = np.where(np.abs(f['entry/input_data/tomo/rotation_angle'][()] - self.theta) <= 0.05)[0][0]
                self.proj=f['entry/intermediate/1-DarkFlatFieldCorrection-tomo/data'][self.indx,:,:]
            elif 'entry/imaging_sum/gts_theta' in f:
                #Assumed Tomography Raw Data
                self.indx = np.where(np.abs(f['entry/imaging_sum/gts_theta'][()] - self.theta) <= 0.05)[0][0]
                self.proj=f['entry/imaging/data'][self.indx,:,:]
            else:
                #Assumed Radiography Row Data
                self.proj=f['entry/imaging/data'][self.indx,:,:]

    def initialize_configuration(self):
        self.x_range_img = len(self.proj[0])
        self.y_range_img = len(self.proj)
        self.aspect_ratio = self.y_range_img / self.x_range_img
        self.binning = 2560 / self.x_range_img
        self.binning = 2560.0 / (len(self.proj[0]) * self.binning)
        self.selection_range = self.y_range_img / 50  # Range of spot selection
        self.img_array = np.array(self.proj)  # Convert image to numpy array for matplotlib

    def scale_x(self, value, tick_number):
        x = ((value * self.binning) - 1280) * self.pixel_size
        return f'{x:.0f}'
    
    def scale_y(self, value, tick_number):
        y = ((value * self.binning) - 1080) * self.pixel_size
        return f'{y:.0f}'

    def InputOutput(self):
        if self.Dif_Path is None:
            dfile_chooser = FileChooser(self.VisitPath)
            dfile_chooser.title = 'Diffraction Data Reduction:'
            dfile_chooser.register_callback(self.load_diffraction)
            display(dfile_chooser)
    
        if self.Img_Path is None:
            ifile_chooser = FileChooser(self.VisitPath)
            ifile_chooser.title = 'Tomography Reconstruction:'
            ifile_chooser.register_callback(self.load_imaging)
            display(ifile_chooser)
    
        if self.Out_Path is None:
            ofile_chooser = FileChooser(self.VisitPath)
            ofile_chooser.title = 'Output File Path Root (without extension):'
            ofile_chooser.register_callback(self.load_output)
            display(ofile_chooser)

    def pseudo_voigt(self, x, amplitude, center, sigma, fraction):
        """ Pseudo-Voigt profile function """
        sigma_g = sigma / np.sqrt(2 * np.log(2))
        sigma_l = sigma / 2
        gauss = (1 - fraction) * np.exp(-((x - center) ** 2) / (2 * sigma_g ** 2))
        lorentz = fraction / (1 + ((x - center) / sigma_l) ** 2)
        return amplitude * (gauss + lorentz)

    def chebyshev(self, x, *coeffs):
        """ Chebyshev polynomial of the first kind """
        return np.polynomial.chebyshev.chebval(x, coeffs)

    def combined_function(self, x, amplitude, center, sigma, fraction, *cheb_coeffs):
        """ Combined Pseudo-Voigt and Chebyshev function """
        return self.pseudo_voigt(x, amplitude, center, sigma, fraction) + self.chebyshev(x, *cheb_coeffs)

    def fit_combined(self, x_fit, y_fit, n):
        # Initial guess for the parameters
        initial_guess = [max(y_fit), x_fit[np.argmax(y_fit)], np.std(x_fit), 0.5] + [0] * (n + 1)
    
        # Fit the combined function
        popt, _ = curve_fit(self.combined_function, x_fit, y_fit, p0=initial_guess)
    
        amplitude, center, sigma, fraction = popt[:4]
    
        # Calculate integral area and FWHM of the Pseudo-Voigt component
        area = amplitude * (fraction * np.pi * sigma / 2 + (1 - fraction) * sigma * np.sqrt(2 * np.pi))
        fwhm = sigma * (fraction + np.sqrt(2 * np.log(2)) * (1 - fraction))
        
        return center, area, fwhm, popt

    def compute_statistics(self, x_fit, y_fit):
        # Calculate the statistical mean of the dataset
        mean = np.average(x_fit, weights=y_fit)
        
        # Calculate the standard deviation of the dataset
        variance = np.average((x_fit - mean)**2, weights=y_fit)
        std_dev = np.sqrt(variance)
        
        # Calculate the integral area using the trapezoidal rule
        area = np.trapz(y_fit, x_fit)
        
        return mean, area, std_dev, None

    def get_profile_peak_parameters(self, x_min, x_max, n=0):
        mask = (self.qvals >= x_min) & (self.qvals <= x_max)
        x_fit = self.qvals[mask]
    
        pk_Mean = []
        pk_Area = []
        pk_FWHM = []
        pk_popt = []
    
        if self.dim == 1:
            for i in range(len(self.kbx)):
                y_fit = self.Ivals[i][mask]
                if self.type == 'Peak':
                    mean, area, fwhm, popt = self.fit_combined(x_fit, y_fit, n)
                elif self.type == 'Histogram':
                    mean, area, fwhm, popt = self.compute_statistics(x_fit, y_fit)
                else:
                    mean, area, fwhm = 0.0
                pk_Mean.append(mean)
                pk_Area.append(area)
                pk_FWHM.append(fwhm)
                pk_popt.append(popt)
        else:
            for i in range(len(self.kbx)):
                row_Mean = []
                row_Area = []
                row_FWHM = []
                row_popt = []
                for j in range(len(self.kby)):
                    y_fit = self.Ivals[j, i][mask]
                    if self.type == 'Peak':
                        mean, area, fwhm, popt = self.fit_combined(x_fit, y_fit, n)
                    elif self.type == 'Histogram':
                        mean, area, fwhm, popt = self.compute_statistics(x_fit, y_fit)
                    else:
                        mean, area, fwhm = 0.0
                    row_Mean.append(mean)
                    row_Area.append(area)
                    row_FWHM.append(fwhm)
                    row_popt.append(popt)
                pk_Mean.append(row_Mean)
                pk_Area.append(row_Area)
                pk_FWHM.append(row_FWHM)
                pk_popt.append(row_popt)
    
        return pk_Mean, pk_Area, pk_FWHM, pk_popt

    def plot_scatter_values(self, axs, kbx, kby, cry, norm, cmap, scatter_plots):
        combinedsc = [sc for sc in scatter_plots]
        
        if self.dim == 1:
            for i in range(len(kbx)):
                self.temp.append([kbx[i], kby[i]])
                color = cmap(norm(cry[i]))
                sc = axs.scatter(kbx[i], kby[i], c=[color], s=10, alpha=0.6)
                scatter_plots.append(sc)
        else:
            for i in range(len(kbx)):
                for j in range(len(kby)):
                    self.temp.append([kbx[i], kby[j]])
                    color = cmap(norm(cry[i][j]))
                    sc = axs.scatter(kbx[i], kby[j], c=[color], s=10, alpha=0.6)
                    scatter_plots.append(sc)

    def update_color_scale(self, val, ax, kbx, kby, cry, cmap, scatter_plots):
        min_val, max_val = val
        norm = plt.Normalize(min_val, max_val)
        
        # Clear existing scatter plots
        for sc in scatter_plots:
            sc.remove()
        scatter_plots.clear()
        
        # Create new scatter plots with updated colors
        if self.dim == 1:
            for i in range(len(kbx)):
                if cry[i] < min_val:
                    color = (0,0,1,1)
                elif cry[i] > max_val:
                    color = (1,0,0,1)
                else:
                    color = cmap(norm(cry[i]))
                sc = ax.scatter(kbx[i], kby[i], c=[color], s=10, alpha=0.6)
                scatter_plots.append(sc)
        else:
            for i in range(len(kbx)):
                for j in range(len(kby)):
                    if cry[i][j] < min_val:
                        color = (0,0,1,1)
                    elif cry[i][j] > max_val:
                        color = (1,0,0,1)
                    else:
                        color = cmap(norm(cry[i][j]))
                    sc = ax.scatter(kbx[i], kby[j], c=[color], s=10, alpha=0.6)
                    scatter_plots.append(sc)
        
        # Redraw the axis
        ax.figure.canvas.draw_idle()

    def ImageCorrelatedCrystallography(self, x_min, x_max, n = 0):
        self.import_diffraction_data()
        self.import_imaging_data()
        self.initialize_configuration()
        self.pk_Mean, self.pk_Area, self.pk_fwhm, self.pk_popt = self.get_profile_peak_parameters(x_min, x_max, n)
        
        self.img_height = self.fig_width * self.aspect_ratio / 2.2
        self.fig_height = self.img_height * (1.00 + 0.2 + 0.1)
        self.fig = plt.figure(figsize=(self.fig_width, self.fig_height))
        self.gs = GridSpec(3, 3, width_ratios=[1, 1, 1], height_ratios=[1.00, 0.2, 0.1], figure=self.fig)
        
        cmap = plt.cm.get_cmap('coolwarm')  # Blue to red color scale
        
        # Plot images and scatter values
        self.ax1 = self.fig.add_subplot(self.gs[0, 0])
        self.ax1.imshow(self.img_array, cmap='Greys', aspect='equal')
        self.ax1.set_xlim(0, self.x_range_img)
        self.ax1.set_ylim(self.y_range_img, 0)
        self.ax1.set_xlabel("(um)")
        self.ax1.set_ylabel("(um)")
        if self.type == 'Peak':
            self.ax1.set_title("Peak Mean")
        elif self.type == 'Histogram':
            self.ax1.set_title("Intensity Mean")
        else:
            self.ax1.set_title("Not Defined")
        self.ax1.xaxis.set_major_locator(MultipleLocator(256 / self.binning))
        self.ax1.yaxis.set_major_locator(MultipleLocator(216 / self.binning))
        self.ax1.xaxis.set_major_formatter(FuncFormatter(self.scale_x))
        self.ax1.yaxis.set_major_formatter(FuncFormatter(self.scale_y))
        norm1 = plt.Normalize(np.min(self.pk_Mean), np.max(self.pk_Mean))
        self.plot_scatter_values(self.ax1, self.kbx, self.kby, self.pk_Mean, norm1, cmap, self.scatter_plots_mean)
        
        self.ax2 = self.fig.add_subplot(self.gs[0, 1])
        self.ax2.imshow(self.img_array, cmap='Greys', aspect='equal')
        self.ax2.set_xlim(0, self.x_range_img)
        self.ax2.set_ylim(self.y_range_img, 0)
        self.ax2.set_xlabel("(um)")
        self.ax2.set_ylabel("(um)")
        if self.type == 'Peak':
            self.ax2.set_title("Peak Area")
        elif self.type == 'Histogram':
            self.ax2.set_title("Intensity Integral Area")
        else:
            self.ax2.set_title("Not Defined")
        self.ax2.xaxis.set_major_locator(MultipleLocator(256 / self.binning))
        self.ax2.yaxis.set_major_locator(MultipleLocator(216 / self.binning))
        self.ax2.xaxis.set_major_formatter(FuncFormatter(self.scale_x))
        self.ax2.yaxis.set_major_formatter(FuncFormatter(self.scale_y))
        norm2 = plt.Normalize(np.min(self.pk_Area), np.max(self.pk_Area))
        self.plot_scatter_values(self.ax2, self.kbx, self.kby, self.pk_Area, norm2, cmap, self.scatter_plots_area)
        
        self.ax3 = self.fig.add_subplot(self.gs[0, 2])
        self.ax3.imshow(self.img_array, cmap='Greys', aspect='equal')
        self.ax3.set_xlim(0, self.x_range_img)
        self.ax3.set_ylim(self.y_range_img, 0)
        self.ax3.set_xlabel("(um)")
        self.ax3.set_ylabel("(um)")
        if self.type == 'Peak':
            self.ax3.set_title("Peak FWHM")
        elif self.type == 'Histogram':
            self.ax3.set_title("Intensity Standard Deviation")
        else:
            self.ax3.set_title("Not Defined")
        self.ax3.xaxis.set_major_locator(MultipleLocator(256 / self.binning))
        self.ax3.yaxis.set_major_locator(MultipleLocator(216 / self.binning))
        self.ax3.xaxis.set_major_formatter(FuncFormatter(self.scale_x))
        self.ax3.yaxis.set_major_formatter(FuncFormatter(self.scale_y))
        norm3 = plt.Normalize(np.min(self.pk_fwhm), np.max(self.pk_fwhm))
        self.plot_scatter_values(self.ax3, self.kbx, self.kby, self.pk_fwhm, norm3, cmap, self.scatter_plots_fwhm)
    
        # Plot histograms
        self.hist_ax1 = self.fig.add_subplot(self.gs[1, 0])
        self.hist_ax1.hist([item for sublist in self.pk_Mean for item in sublist], bins=40, alpha=1.0, color='gray')
        self.hist_ax1.yaxis.set_visible(False)
        
        self.hist_ax2 = self.fig.add_subplot(self.gs[1, 1])
        self.hist_ax2.hist([item for sublist in self.pk_Area for item in sublist], bins=40, alpha=1.0, color='gray')
        self.hist_ax2.yaxis.set_visible(False)
        
        self.hist_ax3 = self.fig.add_subplot(self.gs[1, 2])
        self.hist_ax3.hist([item for sublist in self.pk_fwhm for item in sublist], bins=40, alpha=1.0, color='gray')
        self.hist_ax3.yaxis.set_visible(False)
        
        # Set up RangeSliders and TextBoxes for color scale adjustment
        self.sliders = []  # Keep a reference to the sliders
        self.float_texts = []  # Keep a reference to the FloatText widgets
        for i, (pk_param, scatter_plots, ax) in enumerate(zip(
            [self.pk_Mean, self.pk_Area, self.pk_fwhm], 
            [self.scatter_plots_mean, self.scatter_plots_area, self.scatter_plots_fwhm],
            [self.ax1, self.ax2, self.ax3]
        )):
            ax_slider = self.fig.add_subplot(self.gs[2, i])
            slider = RangeSlider(ax_slider, '', np.min(pk_param), np.max(pk_param), valinit=(np.min(pk_param), np.max(pk_param)), orientation='horizontal')
            slider.valtext.set_visible(True)
            slider.valtext.set_position((0.5, -0.5))
            slider.on_changed(lambda val, ax=ax, pk_param=pk_param, scatter_plots=scatter_plots: 
                              self.update_color_scale(val, ax, self.kbx, self.kby, pk_param, cmap, scatter_plots))
            self.sliders.append(slider)  # Store the slider in the list
            
            # Add FloatText widgets for min and max values
            float_text_min = FloatText(value=round(np.min(pk_param), 4), layout=Layout(width=f'{self.fig_width / 9}in'))
            float_text_min.observe(lambda change, slider=slider: self.update_slider_min(change['new'], slider), names='value')
            self.float_texts.append(float_text_min)
            
            float_text_max = FloatText(value=round(np.max(pk_param), 4), layout=Layout(width=f'{self.fig_width / 9}in'))
            float_text_max.observe(lambda change, slider=slider: self.update_slider_max(change['new'], slider), names='value')
            self.float_texts.append(float_text_max)
            
            # Update FloatText values when RangeSlider is adjusted
            def update_float_texts(val, float_text_min=float_text_min, float_text_max=float_text_max):
                float_text_min.value, float_text_max.value = round(val[0], 4), round(val[1], 4)
            
            slider.on_changed(update_float_texts)
            
            
            # Display FloatText widgets below the figure
            #display(float_text_min, float_text_max)
            
        # Enable interactive mode and adjust layout
        plt.ion()
        plt.tight_layout(pad=0)
        plt.subplots_adjust(top=0.95, bottom=0.15)  # Adjust the top and bottom padding as needed
        plt.show()

        # Display FloatText widgets below the figure in a single row with spaces between specified pairs
        float_text_widgets = []
        for float_text_min, float_text_max in zip(self.float_texts[::2], self.float_texts[1::2]):
            float_text_widgets.append(HBox([float_text_min, float_text_max]))
        
        # Add blank spaces between the second and third, and the fourth and fifth fields
        float_text_widgets.insert(0, VBox(layout=Layout(width='90px')))        
        float_text_widgets.insert(2, VBox(layout=Layout(width='110px')))
        float_text_widgets.insert(4, VBox(layout=Layout(width='110px')))
        
        display(HBox(float_text_widgets))
    
    def update_slider_min(self, val, slider):
        slider.valmin = float(val)
        slider.set_val((slider.valmin, slider.valmax))
    
    def update_slider_max(self, val, slider):
        slider.valmax = float(val)
        slider.set_val((slider.valmin, slider.valmax))

    def plot_row_with_fit(self, kbx, kby, x_min, x_max):
        """ Plot the row data and overlay the fit profile within the specified range """
        if self.dim == 1:
            x_data = self.qvals
            y_data = self.Ivals[kbx]
            fit_params = self.pk_popt[kbx]
        else:
            x_data = self.qvals
            y_data = self.Ivals[kby][kbx]
            fit_params = self.pk_popt[kbx][kby]
        
        # Mask to limit the range
        mask = (x_data >= x_min) & (x_data <= x_max)
        x_data = x_data[mask]
        y_data = y_data[mask]
        
        # Generate the fit profile
        fit_profile = self.combined_function(x_data, *fit_params)
        
        # Generate the Chebyshev component
        cheb_coeffs = fit_params[4:]  # Assuming the Chebyshev coefficients start from the 5th parameter
        cheb_profile = self.chebyshev(x_data, *cheb_coeffs)
        
        # Plot the data
        plt.figure(figsize=(10, 6))
        plt.plot(x_data, y_data, label='Row Data', marker='o', linestyle='', color='blue')
        plt.plot(x_data, fit_profile, label='Fit Profile', linestyle='--', color='red')
        plt.plot(x_data, cheb_profile, label='Chebyshev Component', linestyle='-.', color='green')
        plt.xlabel('Q-values')
        plt.ylabel('Intensity')
        plt.title(f'Row Data and Fit Profile (kbx={kbx}, kby={kby})')
        plt.legend()
        plt.grid(True)
        plt.show()
