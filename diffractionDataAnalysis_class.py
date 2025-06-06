import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.widgets import Slider, RangeSlider, Button, TextBox
from matplotlib.ticker import FuncFormatter, MultipleLocator
import matplotlib.cm as cm
from ipywidgets import Button, Layout, VBox
from IPython.display import display
#%matplotlib widget

from ipyfilechooser import FileChooser
import h5py
import os, sys, glob
import pandas as pd
import ipympl
from matplotlib.patches import Rectangle

from scipy.optimize import curve_fit
from scipy.optimize import minimize
from ipywidgets import FloatText, Layout, HBox, VBox

from scipy.ndimage import center_of_mass
import tifffile as tiff

#Texture Analysis:
#Datasets in: '/processed/intermediate/4-Moving Beam Cake Remapping/data'
#Dataset structure: 4D ... kby, kbx, sector, intensity
#Sector angles in: '/processed/intermediate/4-Moving Beam Cake Remapping/azimuthal angle (degrees)'

class DiffractionFlux:
    def __init__(self, path=None):
        self.Img_Path = path
        self.KnownFlux = []
        self.order = 3
        self.coeffs = None
        self.scale = 1.0 #10000

    def GetFluxMap(self):
        base_path = self.Img_Path + '/projections'
        frame_list = [f for f in os.listdir(base_path) if f.lower().endswith(".tiff") or f.lower().endswith(".tif")]

        for name in frame_list:
            path = base_path + '/' + name
            
            img = tiff.imread(path)
            
            if img.ndim == 3:
                # Use a simple luminosity method to convert to gray-scale.
                # Here weights are chosen for the R,G,B channels (if the image is in that format).
                if img.shape[2] >= 3:
                    img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
                else:
                    # If there are extra channels but not color, select the first channel.
                    img = img[..., 0]
            
            img = img.astype(np.float64)
            
            Y, X = np.indices(img.shape)
            
            total_intensity = np.sum(img)
            average_intensity = total_intensity / (img.shape[0]*img.shape[1])
            threshold_intensity = average_intensity + (np.max(img)-average_intensity) * 0.10
            
            binary_mask = img > threshold_intensity
            center = center_of_mass(binary_mask.astype(float))
            flux = np.sum(img[binary_mask])
    
            self.KnownFlux.append([center[1],center[0],flux])
        
        self.fit_2d_polynomial()
        
    def fit_2d_polynomial(self):
        """
        Fits a 2D polynomial of specified order to the data and normalizes it to a maximum of 10000.
    
        Args:
            data (list of lists): List of [x, y, z] sublists, where x, y are coordinates and z is intensity.
            order (int): Order of the polynomial (default: 3rd order).
    
        Returns:
            numpy array: Normalized coefficients of the fitted polynomial.
        """
        data = np.array(self.KnownFlux)
        x = data[:, 0]  # x-coordinates
        y = data[:, 1]  # y-coordinates
        z = data[:, 2]  # intensity values
        
        # Create a design matrix for polynomial fitting
        def poly_terms(x, y, order):
            terms = []
            for i in range(self.order + 1):
                for j in range(self.order + 1 - i):
                    terms.append((x ** i) * (y ** j))
            return np.column_stack(terms)
    
        # Fit a 2D polynomial using least squares regression
        X_poly = poly_terms(x, y, self.order)
        self.coeffs, _, _, _ = np.linalg.lstsq(X_poly, z, rcond=None)
        
        # Compute the fitted values
        fitted_values = X_poly @ self.coeffs
        
        # Normalize coefficients so the max fitted value is 1
        max_value = np.max(fitted_values)
        if max_value != 0:  # Avoid division by zero
            self.coeffs /= (max_value/self.scale)

    def get_flux(self, x_query, y_query):
        """
        Predicts intensity at an arbitrary (x, y) position using the fitted polynomial.
    
        Args:
            coeffs (numpy array): Coefficients of the fitted polynomial.
            x_query (float): x-coordinate for prediction.
            y_query (float): y-coordinate for prediction.
            order (int): Order of the polynomial (default: 3rd order).
    
        Returns:
            float: Approximated intensity value.
        """
        # Create the polynomial terms for the query point
        def poly_terms(x, y, order):
            terms = []
            for i in range(self.order + 1):
                for j in range(self.order + 1 - i):
                    terms.append((x ** i) * (y ** j))
            return np.column_stack(terms)
    
        X_query_poly = poly_terms(np.array([x_query]), np.array([y_query]), self.order)
        predicted_value = np.dot(X_query_poly, self.coeffs)
    
        return predicted_value[0]

class DiffractionDataAnalysis:
    def __init__(self, diff_path=None, kbmap_path=None, img_path=None, out_path=None, projection_index=0):
        self.VisitPath = "/dls/k11/data"
        self.Dif_Path = diff_path
        self.Flx_Path = kbmap_path
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
        self.Psi = None
        self.theta = None
        
        # Imaging data
        self.proj = None

        # Peak analysis
        self.x_min = None
        self.x_max = None
        self.GoF_threshold = None;

        self.pk_Area_Normalized = []
        self.pk_Area = []
        self.pk_Mean = []
        self.pk_FWHM = []
        self.pk_popt = []
        self.pk_GoF = []
        
        # Image properties
        self.pixel_size = 0.54
        self.binning = 1
        self.x_range_img = 2560
        self.y_range_img = 2160
        self.aspect_ratio = 2560.0 / 2160.0
        self.selection_range = 20
        
        # Figure setup
        self.fig_width = 10
        self.img_height = None
        self.fig_height = None
        self.fig = None
        self.gs = None
        self.ax1 = None
        self.ax2 = None
        
        # Data arrays
        self.img_array = None
        self.scatter_plots_mean = []
        self.scatter_plots_area = []
        self.scatter_plots_fwhm = []
        self.scatter_plots_GoF = []
        self.xrd = []

        self.kb_ix = None
        self.kb_iy = None
        
    def load_diffraction(self, chooser):
        if chooser.selected:
            self.Dif_Path = chooser.selected

    def load_kbmap(self, chooser):
        if chooser.selected:
            self.Flx_Path = chooser.selected
    
    def load_imaging(self, chooser):
        if chooser.selected:
            self.Img_Path = chooser.selected
    
    def load_output(self, chooser):
        if chooser.selected:
            self.Out_Path = chooser.selected

    def import_diffractiondata_Azimuthal(self):
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
            
            if '/processed/intermediate/4-Moving Beam Cake Remapping/azimuthal angle (degrees)' in f:
                self.Psi=f['/processed/intermediate/4-Moving Beam Cake Remapping/azimuthal angle (degrees)'][()]
        
        self.x_min = min(self.qvals)
        self.x_max = max(self.qvals)

    def import_diffractiondata_Cake(self):
        with h5py.File(self.Dif_Path,'r') as f:
            self.Ivals=f['/processed/intermediate/4-Moving Beam Cake Remapping/data'][()]
            self.dim=len(self.Ivals.shape)-2
            
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
            
            self.Psi=f['/processed/intermediate/4-Moving Beam Cake Remapping/azimuthal angle (degrees)'][()]
        
        self.x_min = min(self.qvals)
        self.x_max = max(self.qvals)
    
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
        self.selection_range = self.y_range_img / 50  # Range of spot selection
        self.img_array = np.array(self.proj)  # Convert image to numpy array for matplotlib

    def scale_x(self, value, tick_number):
        x = ((value * self.binning) - 1280) * self.pixel_size
        return f'{x:.0f}'
    
    def scale_y(self, value, tick_number):
        y = (1080 - (value * self.binning)) * self.pixel_size
        return f'{y:.0f}'

    def InputOutput(self):
        if self.Dif_Path is None:
            dfile_chooser = FileChooser(self.VisitPath)
            dfile_chooser.title = 'Diffraction Data Reduction:'
            dfile_chooser.register_callback(self.load_diffraction)
            display(dfile_chooser)

        if self.Flx_Path is None:
            dfile_chooser = FileChooser(self.VisitPath)
            dfile_chooser.title = 'KB-map Intensity Reference:'
            dfile_chooser.register_callback(self.load_kbmap)
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

    def compute_statistics(self, x_fit, y_fit):
        # Calculate the statistical mean of the dataset
        mean = np.average(x_fit, weights=y_fit)
        
        # Calculate the standard deviation of the dataset
        #variance = np.average((x_fit - mean)**2, weights=y_fit)
        #std_dev = np.sqrt(variance)

        # Full width Half Maximum
        half_max = np.max(y_fit) / 2.0
        indices = np.where(y_fit >= half_max)[0]
        if len(indices)>0: fwhm = x_fit[indices[-1]] - x_fit[indices[0]]
        else: fwhm = 0.0
        
        # Calculate the integral area using the trapezoidal rule
        area = np.trapz(y_fit, x_fit)
        
        return mean, area, fwhm, 1.0, None

    def pseudo_voigt_old(self, x, amplitude, center, sigma, fraction):
        """ Pseudo-Voigt profile function """
        sigma_g = sigma / np.sqrt(2 * np.log(2))
        sigma_l = sigma / 2
        gauss = (1 - fraction) * np.exp(-((x - center) ** 2) / (2 * sigma_g ** 2))
        lorentz = fraction / (1 + ((x - center) / sigma_l) ** 2)
        return amplitude * (gauss + lorentz)

    def pseudo_voigt(self, x, amplitude, center, sigma_g, sigma_l, fraction):
        """ Pseudo-Voigt profile function """
        gauss = (1 - fraction) * np.exp(-((x - center) ** 2) / (2 * sigma_g ** 2))
        lorentz = fraction / (1 + ((x - center) / sigma_l) ** 2)
        return amplitude * (gauss + lorentz)

    def chebyshev(self, x, *coeffs):
        """ Chebyshev polynomial of the first kind """
        return np.polynomial.chebyshev.chebval(x, coeffs)

    def combined_function(self, x, amplitude, center, sigma_g, sigma_l, fraction, *cheb_coeffs):
        """ Combined Pseudo-Voigt and Chebyshev function """
        return self.pseudo_voigt(x, amplitude, center, sigma_g, sigma_l, fraction) + self.chebyshev(x, *cheb_coeffs)
    
    def fit_combined(self, x_fit, y_fit, n):
        # Initial guess for the parameters
        sigma = np.std(x_fit)
        sigma_g = sigma / np.sqrt(2 * np.log(2))
        sigma_l = sigma / 2

        amplitude = x_fit[np.argmax(y_fit)]
        initial_guess = [max(y_fit), amplitude, sigma_g, sigma_l, 0.5] + [0] * (n + 1)
        #initial_guess = [max(y_fit), x_fit[np.argmax(y_fit)], np.std(x_fit), 0.5] + [0] * (n + 1)
##        
        # Fit the combined function
        try: popt, _ = curve_fit(self.combined_function, x_fit, y_fit, p0=initial_guess)
        except RuntimeError:
            popt = initial_guess
##        
#        from scipy.optimize import least_squares
#        
#        # Define a residual function for least_squares
#        def residuals(params, x, y, func):
#            return y - func(x, *params)
#        
#        # Lower bounds: amplitude, center, sigma_g, sigma_l, fraction, cheb_coeffs...
#        lower_bounds = [0, -np.inf, 1e-6, 1e-6, 0.0] + [-np.inf]*(n + 1)
#        upper_bounds = [np.inf, np.inf, np.inf, np.inf, 1.0] + [np.inf]*(n + 1)
#        
#        # Perform optimization using least_squares
#        
#        result = least_squares(residuals, initial_guess, args=(x_fit, y_fit, self.combined_function),bounds=(lower_bounds, upper_bounds))
#        #result = least_squares(residuals, initial_guess, args=(x_fit, y_fit, self.combined_function))
#        
#        # Extract optimized parameters
#        popt = result.x
##
        
        # Calculate integral area and FWHM of the Pseudo-Voigt component
        y_peak = self.pseudo_voigt(x_fit, *popt[:5])
        
        mean, area, fwhm, _, _ = self.compute_statistics(x_fit, y_peak)

        y_pred = self.combined_function(x_fit, *popt)
        residuals = y_fit - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
        GoF = 1 - (ss_res / ss_tot)
        
        return mean, area, fwhm, GoF, popt
    
    def get_ProfilePeakParameters_Azimuthal(self, x_min, x_max, n=0):
        mask = (self.qvals >= x_min) & (self.qvals <= x_max)
        x_fit = self.qvals[mask]
    
        pk_Mean = []
        pk_Area = []
        pk_FWHM = []
        pk_popt = []
        pk_GoF = []
    
        if self.dim == 1:
            for i in range(len(self.kbx)):
                y_fit = self.Ivals[i][mask]
                mean, area, fwhm, GoF, popt = self.fit_combined(x_fit, y_fit, n)
                pk_Mean.append(mean)
                pk_Area.append(area)
                pk_FWHM.append(fwhm)
                pk_popt.append(popt)
                pk_GoF.append(GoF)
        else:
            for i in range(len(self.kbx)):
                row_Mean = []
                row_Area = []
                row_FWHM = []
                row_popt = []
                row_GoF = []
                for j in range(len(self.kby)):
                    y_fit = self.Ivals[j, i][mask]
                    mean, area, fwhm, GoF, popt = self.fit_combined(x_fit, y_fit, n)
                    row_Mean.append(mean)
                    row_Area.append(area)
                    row_FWHM.append(fwhm)
                    row_popt.append(popt)
                    row_GoF.append(GoF)
                pk_Mean.append(row_Mean)
                pk_Area.append(row_Area)
                pk_FWHM.append(row_FWHM)
                pk_popt.append(row_popt)
                pk_GoF.append(row_GoF)
    
        return pk_Mean, pk_Area, pk_FWHM, pk_popt, pk_GoF

    def get_ProfilePeakParameters_Cake(self, x_min, x_max, n=0):
        mask = (self.qvals >= x_min) & (self.qvals <= x_max)
        x_fit = self.qvals[mask]
        
        pk_Mean = []
        pk_Area = []
        pk_FWHM = []
        pk_popt = []
        pk_GoF = []
        
        if self.dim == 1:
            for i in range(len(self.kbx)):
                y_fit = self.Ivals[i][mask]
                mean, area, fwhm, GoF, popt = self.fit_combined(x_fit, y_fit, n)
                set_Mean = []
                set_Area = []
                set_FWHM = []
                set_popt = []
                set_GoF = []
                for k in range(len(self.Psi)):
                    y_fit = self.Ivals[i, k][mask]
                    mean, area, fwhm, GoF, popt = self.fit_combined(x_fit, y_fit, n)
                    set_Mean.append(mean)
                    set_Area.append(area)
                    set_FWHM.append(fwhm)
                    set_popt.append(popt)
                    set_GoF.append(GoF)
                pk_Mean.append(self.Psi[set_Mean.index(max(set_Mean))])
                pk_Area.append(self.Psi[set_Area.index(max(set_Area))])
                pk_FWHM.append(self.Psi[set_FWHM.index(max(set_FWHM))])
                pk_GoF.append(1.0 - np.abs(set_GoF-1.0).max)
        else:
            for i in range(len(self.kbx)):
                row_Mean = []
                row_Area = []
                row_FWHM = []
                row_popt = []
                row_GoF = []
                for j in range(len(self.kby)):
                    set_Mean = []
                    set_Area = []
                    set_FWHM = []
                    set_popt = []
                    set_GoF = []
                    for k in range(len(self.Psi)):
                        y_fit = self.Ivals[j, i, k][mask]
                        mean, area, fwhm, GoF, popt = self.fit_combined(x_fit, y_fit, n)
                        set_Mean.append(mean)
                        set_Area.append(area)
                        set_FWHM.append(fwhm)
                        set_popt.append(popt)
                        set_GoF.append(GoF)
                    row_Mean.append(self.Psi[set_Mean.index(max(set_Mean))])
                    row_Area.append(self.Psi[set_Area.index(max(set_Area))])
                    row_FWHM.append(self.Psi[set_FWHM.index(max(set_FWHM))])
                    row_GoF.append(1.0 - max(np.abs(np.array(set_GoF) - 1.0)))
                pk_Mean.append(row_Mean)
                pk_Area.append(row_Area)
                pk_FWHM.append(row_FWHM)
                pk_GoF.append(row_GoF)
        
        return pk_Mean, pk_Area, pk_FWHM, pk_popt, pk_GoF

    def plot_scatter_values(self, axs, cmap, scatter_plots, kbx, kby, cry, GoF, GoF_threshold):
        combinedsc = [sc for sc in scatter_plots]

        if self.dim < 2:
            flat_cry = cry
            flat_GoF = [np.abs(item - 1.0) for item in GoF]
        else:
            flat_cry = [item for sublist in cry for item in sublist]
            flat_GoF = [np.abs(item - 1.0) for sublist in GoF for item in sublist]
        
        if GoF_threshold is None:
            data = [flat_cry for flat_cry in flat_cry if flat_cry > 0.0]
        else:
            data = [flat_cry for flat_cry, flat_GoF in zip(flat_cry, flat_GoF) if flat_GoF < GoF_threshold]
        
        norm = plt.Normalize(np.min(data), np.max(data))
        
        if self.dim == 1:
            for i in range(len(kbx)):
                if GoF_threshold is None:
                    mask = (cry[i] > 0.0)
                else:
                    mask = (np.abs(GoF[i]-1.0) < GoF_threshold)
                
                color = cmap(norm(cry[i])) if mask else 'black'
                sc = axs.scatter(kbx[i], kby[i], c=[color], s=10, alpha=0.6)
                scatter_plots.append(sc)
        else:
            for i in range(len(kbx)):
                for j in range(len(kby)):
                    if GoF_threshold is None:
                        mask = (cry[i][j] > 0.0)
                    else:
                        mask = (np.abs(GoF[i][j]-1.0) < GoF_threshold)
                    
                    color = cmap(norm(cry[i][j])) if mask else 'black'
                    self.facec = cmap(norm(cry[i][j])) if mask else (0,0,0,0)
                    sc = axs.scatter(kbx[i], kby[j], facecolors=[self.facec], edgecolors=[color], s=10, alpha=0.6)
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
                elif cry[i] > nmax_val:
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

    def update_diff_plot(self, axs, kb_ix, kb_iy, legend):
        axs.clear()
        #axs.set_title('Diffraction dataset: ' + str(diffileno), fontsize=10)
        axs.set_xlabel(self.qvs2t)
        axs.set_ylabel("Intensity (counts)")
        axs.set_xlim(self.x_min, self.x_max)
        axs.set_ylim(0, 2.0)
    
        if self.dim == 1:
            text = "(" + str(kb_iy+1) + ")"
        else:
            text = "(" + str(kb_iy+1) + " " + str(kb_ix+1) + ")"
        
        mask = (self.qvals >= self.x_min) & (self.qvals <= self.x_max)
        
        if self.dim == 1:
            y_min = min(self.Ivals[kb_iy][mask])
            y_max = max(self.Ivals[kb_iy][mask])
            fit_params = self.pk_popt[kb_ix]
            if legend:
                axs.plot(self.qvals, self.Ivals[kb_iy], color='b', label=text)
            else:
                axs.plot(self.qvals, self.Ivals[kb_iy], color='b')
        else:
            y_min = min(self.Ivals[kb_iy,kb_ix][mask])
            y_max = max(self.Ivals[kb_iy,kb_ix][mask])
            fit_params = self.pk_popt[kb_ix][kb_iy]
            if legend:
                axs.plot(self.qvals, self.Ivals[kb_iy,kb_ix], color='b', label=text)
            else:
                axs.plot(self.qvals, self.Ivals[kb_iy,kb_ix], color='b')
    
        # Generate the fit profile
        fit_profile = self.combined_function(self.qvals[mask], *fit_params)
        
        # Generate the Chebyshev component
        cheb_coeffs = fit_params[5:]  # Assuming the Chebyshev coefficients start from the 5th parameter
        cheb_profile = self.chebyshev(self.qvals[mask], *cheb_coeffs)
        
        axs.plot(self.qvals[mask], fit_profile, label='Fit Profile', linestyle='--', color='red')
        axs.plot(self.qvals[mask], cheb_profile, label='Chebyshev Component', linestyle='-.', color='green')
        
        axs.set_ylim(y_min * 0.95, y_max * 1.05)
        
        # Add the extra line in the legend
        handles, labels = axs.get_legend_handles_labels()
        if self.dim == 1:
            handles.append(plt.Line2D([], [], linestyle='-', color='none', label=f'Mean:  {self.pk_Mean[kb_iy]:.4f}'))
            handles.append(plt.Line2D([], [], linestyle='-', color='none', label=f'Area: {self.pk_Area[kb_iy]:.4f}'))
            handles.append(plt.Line2D([], [], linestyle='-', color='none', label=f'Normalized Area: {self.pk_Area_Normalized[kb_iy]:.4f}'))
            handles.append(plt.Line2D([], [], linestyle='-', color='none', label=f'FWHM:  {self.pk_fwhm[kb_iy]:.4f}'))
            handles.append(plt.Line2D([], [], linestyle='-', color='none', label=f'GoF:   {self.pk_GoF[kb_iy]:.4f}'))
        else:
            handles.append(plt.Line2D([], [], linestyle='-', color='none', label=f'Mean:  {self.pk_Mean[kb_ix][kb_iy]:.4f}'))
            handles.append(plt.Line2D([], [], linestyle='-', color='none', label=f'Area: {self.pk_Area[kb_ix][kb_iy]:.4f}'))
            handles.append(plt.Line2D([], [], linestyle='-', color='none', label=f'Normalized Area: {self.pk_Area_Normalized[kb_ix][kb_iy]:.4f}'))
            handles.append(plt.Line2D([], [], linestyle='-', color='none', label=f'FWHM:  {self.pk_fwhm[kb_ix][kb_iy]:.4f}'))
            handles.append(plt.Line2D([], [], linestyle='-', color='none', label=f'GoF:   {self.pk_GoF[kb_ix][kb_iy]:.4f}'))
        axs.legend(handles=handles, loc='upper left', fontsize=8)
    
    def onclick(self, event):
        if event.inaxes == self.ax1:  # Only respond to clicks in the first subplot
            if event.xdata is not None and event.ydata is not None:
                x = int(event.xdata)
                y = int(event.ydata)
                
                if 0 <= x < self.x_range_img and 0 <= y < self.y_range_img:
                    kbx_i = np.where((self.kbx >= x-self.selection_range/self.binning) & (self.kbx <= x+self.selection_range/self.binning))[0] # Get kbx index at the clicked position
                    kby_i = np.where((self.kby >= y-self.selection_range/self.binning) & (self.kby <= y+self.selection_range/self.binning))[0] # Get kby index at the clicked position
                    if len(kbx_i) > 1: kbx_i = kby_i
                    if len(kby_i) > 1: kby_i = kbx_i
                    
                    if len(kbx_i) & len(kby_i) == 1:
                        self.kb_ix = kbx_i.item()
                        self.kb_iy = kby_i.item()
                        self.update_diff_plot(self.ax2, self.kb_ix, self.kb_iy, True)
                    
                    self.fig.canvas.draw_idle()
    
    def ImageCorrelatedCrystallography_Azimuthal_Explore(self, x_min, x_max, n = 0, GoF_threshold = 0.02):
        self.import_diffractiondata_Azimuthal()
        self.import_imaging_data()
        self.initialize_configuration()

        self.x_min = x_min
        self.x_max = x_max
        self.GoF_threshold = GoF_threshold
        
        self.img_height = self.fig_width * self.aspect_ratio / 2
        self.fig_height = self.img_height
        self.fig = plt.figure(figsize=(self.fig_width, self.fig_height))
        self.gs = GridSpec(1, 2, height_ratios=[self.img_height], width_ratios=[1, 1], figure=self.fig)
        
        self.pk_Mean, self.pk_Area, self.pk_fwhm, self.pk_popt, self.pk_GoF = self.get_ProfilePeakParameters_Azimuthal(x_min, x_max, n)

        # Normalize Peak Area
        Obj = DiffractionFlux(self.Flx_Path)
        Obj.GetFluxMap()

        if self.dim == 1:
            self.pk_Area_Normalized = self.pk_Area[:]
            for ix in range(len(self.kbx)):
                self.pk_Area_Normalized[ix] /= Obj.get_flux(self.kbx[ix], self.kby[ix])
        else:
            self.pk_Area_Normalized = [sublist[:] for sublist in self.pk_Area]
            for ix in range(len(self.kbx)):
                for iy in range(len(self.kby)):
                    self.pk_Area_Normalized[ix][iy] /= Obj.get_flux(self.kbx[ix], self.kby[iy])
        
        cmap = plt.cm.get_cmap('coolwarm')  # Blue to red color scale
        
        ### Left Column ## ##
        
        # Cell (0,0): Image
        self.ax1 = self.fig.add_subplot(self.gs[0, 0])
        self.img_plot = self.ax1.imshow(self.img_array, cmap='Greys', aspect='equal')
        
        self.ax1.set_xlim(0, self.x_range_img)
        self.ax1.set_ylim(self.y_range_img,0)
        self.ax1.set_xlabel("(um)")
        self.ax1.set_ylabel("(um)")
        
        self.ax1.xaxis.set_major_locator(MultipleLocator(256/self.binning))
        self.ax1.yaxis.set_major_locator(MultipleLocator(216/self.binning))
        
        self.ax1.xaxis.set_major_formatter(FuncFormatter(self.scale_x))
        self.ax1.yaxis.set_major_formatter(FuncFormatter(self.scale_y))
        
        norm3 = plt.Normalize(np.min(self.pk_GoF), np.max(self.pk_GoF))
        self.plot_scatter_values(self.ax1, cmap, self.scatter_plots_GoF, self.kbx, self.kby, self.pk_GoF, self.pk_GoF, self.GoF_threshold)
        
        #### Right Column ## ##
        
        # Cell (0,1): Second figure
        self.ax2 = self.fig.add_subplot(self.gs[0, 1])
        self.ax2.set_xlim(x_min, x_max)
        self.ax2.set_ylim(0, 1.0)
        self.ax2.set_xlabel(self.qvs2t)
        self.ax2.set_ylabel("Intensity (counts)")
        
        #### Connect the events
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)

        #### Enable interactive mode
        plt.ion()
        
        # Save button callback
        def save_plots(event):
            self.fig.savefig(f'{self.Out_Path}_GoF.tiff', format='tiff', dpi=600)
        
        # Add button
        save_button = Button(description="Save Figure", layout=Layout(width='120px'))
        save_button.on_click(save_plots)
        display(VBox([save_button]))
        
        #### Adjust layout to prevent overlapping
        plt.tight_layout(pad=0)
        plt.subplots_adjust(top=0.95)  # Adjust the top padding as needed
        plt.show()

    def ImageCorrelatedCrystallography_Azimuthal(self, x_min, x_max, n = 0, GoF_threshold=0.02):
        self.import_diffractiondata_Azimuthal()
        self.import_imaging_data()
        self.initialize_configuration()

        self.x_min = x_min
        self.x_max = x_max
        self.GoF_threshold = GoF_threshold
        
        self.pk_Mean, self.pk_Area, self.pk_fwhm, self.pk_popt, self.pk_GoF = self.get_ProfilePeakParameters_Azimuthal(x_min, x_max, n)
        
        # Normalize Peak Area
        Obj = DiffractionFlux(self.Flx_Path)
        Obj.GetFluxMap()

        if self.dim == 1:
            self.pk_Area_Normalized = self.pk_Area[:]
            for ix in range(len(self.kbx)):
                self.pk_Area_Normalized[ix] /= Obj.get_flux(self.kbx[ix], self.kby[ix])
        else:
            self.pk_Area_Normalized = [sublist[:] for sublist in self.pk_Area]
            for ix in range(len(self.kbx)):
                for iy in range(len(self.kby)):
                    self.pk_Area_Normalized[ix][iy] /= Obj.get_flux(self.kbx[ix], self.kby[iy])
        
        self.img_height = self.fig_width * self.aspect_ratio / 2
        fig_height = self.img_height
        fig = plt.figure(figsize=(self.fig_width, fig_height))
        gs = GridSpec(2, 3, height_ratios=[0.7,0.3], width_ratios=[1, 1, 1], figure=fig)
        
        cmap = plt.cm.get_cmap('coolwarm')  # Blue to red color scale
        
        # Plot images and scatter values
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(self.img_array, cmap='Greys', aspect='equal')
        ax1.set_xlim(0, self.x_range_img)
        ax1.set_ylim(self.y_range_img, 0)
        ax1.set_xlabel("(um)")
        ax1.set_ylabel("(um)")
        ax1.set_title("Peak Mean")
        ax1.xaxis.set_major_locator(MultipleLocator(256 / self.binning * 2))
        ax1.yaxis.set_major_locator(MultipleLocator(216 / self.binning * 2))
        ax1.xaxis.set_major_formatter(FuncFormatter(self.scale_x))
        ax1.yaxis.set_major_formatter(FuncFormatter(self.scale_y))
        self.plot_scatter_values(ax1, cmap, self.scatter_plots_mean, self.kbx, self.kby, self.pk_Mean, self.pk_GoF, GoF_threshold)
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(self.img_array, cmap='Greys', aspect='equal')
        ax2.set_xlim(0, self.x_range_img)
        ax2.set_ylim(self.y_range_img, 0)
        ax2.set_xlabel("(um)")
        ax2.set_ylabel("(um)")
        ax2.set_title("Normalized Peak Area")
        ax2.xaxis.set_major_locator(MultipleLocator(256 / self.binning * 2))
        ax2.yaxis.set_major_locator(MultipleLocator(216 / self.binning * 2))
        ax2.xaxis.set_major_formatter(FuncFormatter(self.scale_x))
        ax2.yaxis.set_major_formatter(FuncFormatter(self.scale_y))
        self.plot_scatter_values(ax2, cmap, self.scatter_plots_area, self.kbx, self.kby, self.pk_Area_Normalized, self.pk_GoF, GoF_threshold)
        
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(self.img_array, cmap='Greys', aspect='equal')
        ax3.set_xlim(0, self.x_range_img)
        ax3.set_ylim(self.y_range_img, 0)
        ax3.set_xlabel("(um)")
        ax3.set_ylabel("(um)")
        ax3.set_title("Peak FWHM")
        ax3.xaxis.set_major_locator(MultipleLocator(256 / self.binning * 2))
        ax3.yaxis.set_major_locator(MultipleLocator(216 / self.binning * 2))
        ax3.xaxis.set_major_formatter(FuncFormatter(self.scale_x))
        ax3.yaxis.set_major_formatter(FuncFormatter(self.scale_y))
        self.plot_scatter_values(ax3, cmap, self.scatter_plots_fwhm, self.kbx, self.kby, self.pk_fwhm, self.pk_GoF, GoF_threshold)
        
        # Plot histograms
        if self.dim < 2:
            flat_GoF = [np.abs(item - 1.0) for item in self.pk_GoF]
        else:
            flat_GoF = [np.abs(item - 1.0) for sublist in self.pk_GoF for item in sublist]

        
        if self.dim < 2:
            flat_cry = self.pk_Mean
        else:
            flat_cry = [item for sublist in self.pk_Mean for item in sublist]
        if self.GoF_threshold is None:
            data = [flat_cry for flat_cry in flat_cry if flat_cry > 0.0]
        else:
            data = [flat_cry for flat_cry, flat_GoF in zip(flat_cry, flat_GoF) if flat_GoF < self.GoF_threshold]
        norm = plt.Normalize(np.min(data), np.max(data))
        hist_ax1 = fig.add_subplot(gs[1, 0])
        n, bins, patches = hist_ax1.hist(data, bins=40, alpha=1.0)
        for patch, value in zip(patches, bins):
            color = cmap(norm(value))
            patch.set_facecolor(color)
        #hist_ax1.hist(data, bins=40, alpha=1.0, color='gray')
        hist_ax1.yaxis.set_visible(False)

        if self.dim < 2:
            flat_cry = self.pk_Area_Normalized
        else:
            flat_cry = [item for sublist in self.pk_Area_Normalized for item in sublist]
        if self.GoF_threshold is None:
            data = [flat_cry for flat_cry in flat_cry if flat_cry > 0.0]
        else:
            data = [flat_cry for flat_cry, flat_GoF in zip(flat_cry, flat_GoF) if flat_GoF < self.GoF_threshold]
        norm = plt.Normalize(np.min(data), np.max(data))
        hist_ax2 = fig.add_subplot(gs[1, 1])
        n, bins, patches = hist_ax2.hist(data, bins=40, alpha=1.0)
        for patch, value in zip(patches, bins):
            color = cmap(norm(value))
            patch.set_facecolor(color)
        #hist_ax2.hist(data, bins=40, alpha=1.0, color='gray')
        hist_ax2.yaxis.set_visible(False)

        if self.dim < 2:
            flat_cry = self.pk_fwhm
        else:
            flat_cry = [item for sublist in self.pk_fwhm for item in sublist]
        if self.GoF_threshold is None:
            data = [flat_cry for flat_cry in flat_cry if flat_cry > 0.0]
        else:
            data = [flat_cry for flat_cry, flat_GoF in zip(flat_cry, flat_GoF) if flat_GoF < self.GoF_threshold]
        norm = plt.Normalize(np.min(data), np.max(data))
        hist_ax3 = fig.add_subplot(gs[1, 2])
        n, bins, patches = hist_ax3.hist(data, bins=40, alpha=1.0)
        for patch, value in zip(patches, bins):
            color = cmap(norm(value))
            patch.set_facecolor(color)
        #hist_ax3.hist(data, bins=40, alpha=1.0, color='gray')
        hist_ax3.yaxis.set_visible(False)
                
        # Save button callback
        def save_plots(event):
            fig.savefig(f'{self.Out_Path}_Crystallography.tiff', format='tiff', dpi=600)
        
        # Add button
        save_button = Button(description="Save Figure", layout=Layout(width='120px'))
        save_button.on_click(save_plots)
        display(VBox([save_button]))
        
        # Enable interactive mode and adjust layout
        plt.ion()
        plt.tight_layout(pad=0)
        plt.show()

    def ImageCorrelatedCrystallography_Cake(self, x_min, x_max, n = 0, GoF_threshold=0.02):
        self.import_diffractiondata_Cake()
        self.import_imaging_data()
        self.initialize_configuration()

        self.x_min = x_min
        self.x_max = x_max
        self.GoF_threshold = GoF_threshold
        
        self.pk_Mean, self.pk_Area, self.pk_fwhm, self.pk_popt, self.pk_GoF = self.get_ProfilePeakParameters_Cake(x_min, x_max, n)
        
        self.img_height = self.fig_width * self.aspect_ratio / 2
        fig_height = self.img_height
        fig = plt.figure(figsize=(self.fig_width, fig_height))
        gs = GridSpec(2, 3, height_ratios=[0.7,0.3], width_ratios=[1, 1, 1], figure=fig)
        
        cmap = plt.cm.get_cmap('coolwarm')  # Blue to red color scale
        
        # Plot images and scatter values
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(self.img_array, cmap='Greys', aspect='equal')
        ax1.set_xlim(0, self.x_range_img)
        ax1.set_ylim(self.y_range_img, 0)
        ax1.set_xlabel("(um)")
        ax1.set_ylabel("(um)")
        ax1.set_title("Peak Mean")
        ax1.xaxis.set_major_locator(MultipleLocator(256 / self.binning * 2))
        ax1.yaxis.set_major_locator(MultipleLocator(216 / self.binning * 2))
        ax1.xaxis.set_major_formatter(FuncFormatter(self.scale_x))
        ax1.yaxis.set_major_formatter(FuncFormatter(self.scale_y))
        self.plot_scatter_values(ax1, cmap, self.scatter_plots_mean, self.kbx, self.kby, self.pk_Mean, self.pk_GoF, GoF_threshold)
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(self.img_array, cmap='Greys', aspect='equal')
        ax2.set_xlim(0, self.x_range_img)
        ax2.set_ylim(self.y_range_img, 0)
        ax2.set_xlabel("(um)")
        ax2.set_ylabel("(um)")
        ax2.set_title("Normalized Peak Area")
        ax2.xaxis.set_major_locator(MultipleLocator(256 / self.binning * 2))
        ax2.yaxis.set_major_locator(MultipleLocator(216 / self.binning * 2))
        ax2.xaxis.set_major_formatter(FuncFormatter(self.scale_x))
        ax2.yaxis.set_major_formatter(FuncFormatter(self.scale_y))
        self.plot_scatter_values(ax2, cmap, self.scatter_plots_area, self.kbx, self.kby, self.pk_Area, self.pk_GoF, GoF_threshold)
        
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(self.img_array, cmap='Greys', aspect='equal')
        ax3.set_xlim(0, self.x_range_img)
        ax3.set_ylim(self.y_range_img, 0)
        ax3.set_xlabel("(um)")
        ax3.set_ylabel("(um)")
        ax3.set_title("Peak FWHM")
        ax3.xaxis.set_major_locator(MultipleLocator(256 / self.binning * 2))
        ax3.yaxis.set_major_locator(MultipleLocator(216 / self.binning * 2))
        ax3.xaxis.set_major_formatter(FuncFormatter(self.scale_x))
        ax3.yaxis.set_major_formatter(FuncFormatter(self.scale_y))
        self.plot_scatter_values(ax3, cmap, self.scatter_plots_fwhm, self.kbx, self.kby, self.pk_fwhm, self.pk_GoF, GoF_threshold)
        
        # Plot histograms
        if self.dim < 2:
            flat_GoF = [np.abs(item - 1.0) for item in self.pk_GoF]
        else:
            flat_GoF = [np.abs(item - 1.0) for sublist in self.pk_GoF for item in sublist]

        
        if self.dim < 2:
            flat_cry = self.pk_Mean
        else:
            flat_cry = [item for sublist in self.pk_Mean for item in sublist]
        if self.GoF_threshold is None:
            data = [flat_cry for flat_cry in flat_cry if flat_cry > 0.0]
        else:
            data = [flat_cry for flat_cry, flat_GoF in zip(flat_cry, flat_GoF) if flat_GoF < self.GoF_threshold]
        norm = plt.Normalize(np.min(data)*0.99, np.max(data)*1.01)
        hist_ax1 = fig.add_subplot(gs[1, 0])
        n, bins, patches = hist_ax1.hist(data, bins=40, alpha=1.0)
        for patch, value in zip(patches, bins):
            color = cmap(norm(value))
            patch.set_facecolor(color)
        #hist_ax1.hist(data, bins=40, alpha=1.0, color='gray')
        hist_ax1.yaxis.set_visible(False)

        if self.dim < 2:
            flat_cry = self.pk_Area_Normalized
        else:
            flat_cry = [item for sublist in self.pk_Area_Normalized for item in sublist]
        if self.GoF_threshold is None:
            data = [flat_cry for flat_cry in flat_cry if flat_cry > 0.0]
        else:
            data = [flat_cry for flat_cry, flat_GoF in zip(flat_cry, flat_GoF) if flat_GoF < self.GoF_threshold]
        
#        norm = plt.Normalize(np.min(data)*0.99, np.max(data)*1.01)
#        hist_ax2 = fig.add_subplot(gs[1, 1])
#        n, bins, patches = hist_ax2.hist(data, bins=40, alpha=1.0)
#        for patch, value in zip(patches, bins):
#            color = cmap(norm(value))
#            patch.set_facecolor(color)
#        #hist_ax2.hist(data, bins=40, alpha=1.0, color='gray')
#        hist_ax2.yaxis.set_visible(False)
#
#        if self.dim < 2:
#            flat_cry = self.pk_fwhm
#        else:
#            flat_cry = [item for sublist in self.pk_fwhm for item in sublist]
#        if self.GoF_threshold is None:
#            data = [flat_cry for flat_cry in flat_cry if flat_cry > 0.0]
#        else:
#            data = [flat_cry for flat_cry, flat_GoF in zip(flat_cry, flat_GoF) if flat_GoF < self.GoF_threshold]
#        norm = plt.Normalize(np.min(data)*0.99, np.max(data)*1.01)
#        hist_ax3 = fig.add_subplot(gs[1, 2])
#        n, bins, patches = hist_ax3.hist(data, bins=40, alpha=1.0)
#        for patch, value in zip(patches, bins):
#            color = cmap(norm(value))
#            patch.set_facecolor(color)
#        #hist_ax3.hist(data, bins=40, alpha=1.0, color='gray')
#        hist_ax3.yaxis.set_visible(False)
                
        # Save button callback
        def save_plots(event):
            fig.savefig(f'{self.Out_Path}_Texture.tiff', format='tiff', dpi=600)
        
        # Add button
        save_button = Button(description="Save Figure", layout=Layout(width='120px'))
        save_button.on_click(save_plots)
        display(VBox([save_button]))
        
        # Enable interactive mode and adjust layout
        plt.ion()
        plt.tight_layout(pad=0)
        plt.show()