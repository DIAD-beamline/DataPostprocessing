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

class DiffractionDataExplorer:
    def __init__(self,  diff_path = None, img_path = None, out_path = None, projection_index = 1):
        self.VisitPath = "/dls/k11/data"
        self.Dif_Path = diff_path
        self.Img_Path = img_path
        self.Out_Path = out_path
        ##
        self.data = None   # diffraction profile data: intensities
        self.qvals = None  # diffraction profile data: Q/2Theta
        self.kbx = None    # diffraction spot position: cs-kbx
        self.kby = None    # diffraction spot position: cs-kby
        self.theta = None  # diffraction spot position: GTS theta
        self.indx = projection_index   # imaging data: projection index
        self.proj = None               # imaging data: projection data
        ##
        self.pixel_size = 0.54
        self.binning = 1
        self.x_range_img = 2560
        self.y_range_img = 2160
        self.aspect_ratio = 2560.0/2160.0
        self.selection_range = 10
        ##
        self.fig_width = 10
        self.img_height = None
        self_fig_height = None
        self.fig = None
        self.gs = None
        self.gss_l = None
        self.gss_r = None
        self.ax1 = None
        self.ax2 = None
        self.ax3_l = None
        self.ax3_r = None
        self.ax4 = None
        self.ax5 = None
        self.ax6_l = None
        self.ax6_r = None
        self.save_img_button_all = None
        self.save_img_button_sel = None
        ##
        self.img_array = None
        self.temp = []
        self.scatter_plots = []
        self.xrd = []
        ##
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
            self.data=f['processed/result/data'][()]
            self.qvals=f['processed/result/q'][()]
            self.kbx = f['entry/diffraction/kb_cs_x'][()]
            self.kby = f['entry/diffraction/kb_cs_y'][()]
            self.theta=round(f['entry/diffraction_sum/gts_theta'][()].max(),2)
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

        self.img_array = np.array(self.proj) # Convert image to numpy array for matplotlib

    def scale_x(self, value, tick_number):
        x = ((value * self.binning) - 1280) * self.pixel_size
        return f'{x:.{0}f}'
    def scale_y(self, value, tick_number):
        y = ((value * self.binning) - 1080) * self.pixel_size
        return f'{y:.{0}f}'

    def plot_scatter(self, axs, kbx, kby):
        combinedsc = [sc for sc in self.scatter_plots]
        for i in range(len(self.kbx)):
            for j in range(len(self.kby)):
                #temp.append([kbx[i], kby[j], np.sum(data[j,i,summation_range[0]:summation_range[1]])])
                self.temp.append([self.kbx[i], self.kby[j]])
                sc = axs.scatter(self.kbx[i], self.kby[j], c='r', s=10, alpha=0.5)
                self.scatter_plots.append(sc)

    def save_img_plot(self, event, all_spots):
        img_fig = plt.subplots()
        img_fig_plot = img_fig[1].imshow(self.img_array, cmap='Greys', aspect='equal')

        img_fig[1].set_xlim(0, self.x_range_img)
        img_fig[1].set_ylim(self.y_range_img, 0)
        img_fig[1].set_xlabel("(um)")
        img_fig[1].set_ylabel("(um)")

        img_fig[1].xaxis.set_major_locator(MultipleLocator(256/self.binning))
        img_fig[1].yaxis.set_major_locator(MultipleLocator(216/self.binning))

        img_fig[1].xaxis.set_major_formatter(FuncFormatter(self.scale_x))
        img_fig[1].yaxis.set_major_formatter(FuncFormatter(self.scale_y))

        if(all_spots): self.plot_scatter(img_fig[1], self.kbx, self.kby)

        df = pd.DataFrame(self.xrd, columns=['kbx', 'kby'])
        df = df.drop_duplicates(subset=['kbx', 'kby'])

        cmap = cm.get_cmap('viridis')
        norm = plt.Normalize(0, len(self.xrd))

        for no, row in df.iterrows():
            text = "(" + str(row['kbx']+1) + " " + str(row['kby']+1) + ")"
            color = cmap(norm(no))

            img_fig[1].scatter(self.kbx[row['kbx']], self.kby[row['kby']], c=color, s=10)

        path = self.Out_Path + "_img.tiff"
        img_fig[0].savefig(path, format='tiff')

    def save_img_plot_all(self, event):
        self.save_img_plot(event, True)
    def save_img_plot_sel(self, event):
        self.save_img_plot(event, False)

    def update_diff_plot(self, axs, legend):
        df = pd.DataFrame(self.xrd, columns=['kbx', 'kby'])
        df = df.drop_duplicates(subset=['kbx', 'kby'])
        c = 0
        y_range = 10.0
        axs.clear()
        #axs.set_title('Diffraction dataset: ' + str(diffileno), fontsize=10)
        axs.set_xlabel("Scattering Momentum/Angle")
        axs.set_ylabel("Intensity (counts)")
        axs.set_xlim(min(self.qvals), max(self.qvals))
        axs.set_ylim(0, 1.0)

        cmap = cm.get_cmap('viridis')
        norm = plt.Normalize(0, len(self.xrd))

        for no, row in df.iterrows():
            text = "(" + str(row['kby']+1) + " " + str(row['kbx']+1) + ")"
            color = cmap(norm(no))

            self.ax1.scatter(self.kbx[row['kbx']], self.kby[row['kby']], color=color, s=10)

            if(legend): axs.plot(self.qvals, self.data[row['kby'],row['kbx']]+c, color=color, label=text)
            else:       axs.plot(self.qvals, self.data[row['kby'],row['kbx']]+c, color=color)
            y_range = max(c+max(self.data[row['kby'],row['kbx']])*1.050, y_range)
            c += 10
        axs.set_ylim(0, y_range)
        axs.legend(loc='upper left', fontsize=8)

    def clear_plots(self, event):
        # Reset dataset
        self.xrd.clear()
        # Reset figure diffraction
        self.ax4.clear()

        self.ax4.set_xlabel("Scattering Momentum/Angle")
        self.ax4.set_ylabel("Intensity (counts)")
        self.ax4.set_xlim(min(self.qvals), max(self.qvals))
        self.ax4.set_ylim(0, 1.0)

        # Reset figure imaging
        self.ax1.clear()
        img_plot = self.ax1.imshow(self.img_array, cmap='Greys', aspect='equal')
        self.ax1.set_xlim(0, self.x_range_img)
        self.ax1.set_ylim(self.y_range_img, 0)
        self.ax1.set_xlabel("(um)")
        self.ax1.set_ylabel("(um)")

        self.ax1.xaxis.set_major_locator(MultipleLocator(256/self.binning))
        self.ax1.yaxis.set_major_locator(MultipleLocator(216/self.binning))

        self.ax1.xaxis.set_major_formatter(FuncFormatter(self.scale_x))
        self.ax1.yaxis.set_major_formatter(FuncFormatter(self.scale_y))

        self.plot_scatter(self.ax1, self.kbx, self.kby)

        ## ## ## ## ## ## ## ## ##

        self.fig.canvas.draw_idle()

    def save_dif_plot(self, event, legend):
        dif_fig = plt.subplots()
        self.update_diff_plot(dif_fig[1], legend)
        path = self.Out_Path + "_dif.tiff"
        dif_fig[0].savefig(path, format='tiff')

    def save_dif_plot_leg(self, event):
        self.save_dif_plot(event, True)
    def save_dif_plot_cle(self, event):
        self.save_dif_plot(event, False)

    def onclick(self, event):
        if event.inaxes == self.ax1:  # Only respond to clicks in the first subplot
            if event.xdata is not None and event.ydata is not None:
                x = int(event.xdata)
                y = int(event.ydata)

                if 0 <= x < self.x_range_img and 0 <= y < self.y_range_img:
                    rgb_value = self.img_array[y, x]  # Note: y,x order for array indexing # Get the RGB value at the clicked position
                    kbx_i = np.where((self.kbx >= x-self.selection_range) & (self.kbx <= x+self.selection_range))[0] # Get kbx index at the clicked position
                    kby_i = np.where((self.kby >= y-self.selection_range) & (self.kby <= y+self.selection_range))[0] # Get kby index at the clicked position

                    # Update selection parameters in cell 1,0
                    self.ax2.clear()
                    self.ax2.set_xticks([])
                    self.ax2.set_yticks([])

                    if len(kbx_i) & len(kby_i) > 0:
                        kb_ix = kbx_i.item()
                        kb_iy = kby_i.item()
                        self.ax2.text(0.5, 0.5,
                            f"Selected pixel: X: {x}, Y: {y} RGB: {rgb_value:.3f}\nDiffraction Spot ID (row,column): {kb_iy+1} {kb_ix+1}",
                            horizontalalignment='center',
                            verticalalignment='center',
                            transform=self.ax2.transAxes)
                        self.xrd.append((kb_ix, kb_iy))
                        self.update_diff_plot(self.ax4, True)
                    else:
                        self.ax2.text(0.5, 0.5,
                            f"Selected pixel: X: {x}, Y: {y} RGB: {rgb_value:.3f}",
                            horizontalalignment='center',
                            verticalalignment='center',
                            transform=self.ax2.transAxes)

                    self.fig.canvas.draw_idle()

    def InputOutput(self):
        ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### File Path Selection

        if self.Dif_Path == None:
            dfile_chooser = FileChooser(self.VisitPath)
            dfile_chooser.title = 'Diffraction Data Reduction:'
            dfile_chooser.register_callback(self.load_diffraction)
            display(dfile_chooser)

        if self.Img_Path == None:
            ifile_chooser = FileChooser(self.VisitPath)
            ifile_chooser.title = 'Tomography Reconstruction:'
            ifile_chooser.register_callback(self.load_imaging)
            display(ifile_chooser)

        if self.Out_Path == None:
            ofile_chooser = FileChooser(self.VisitPath)
            ofile_chooser.title = 'Output File Path Root (without extension):'
            ofile_chooser.register_callback(self.load_output)
            display(ofile_chooser)

    def DataExplorer(self):
        self.import_diffraction_data()
        self.import_imaging_data()
        self.initialize_configuration()

        self.img_height = self.fig_width * self.aspect_ratio / 2
        self.fig_height = self.img_height + 0.2 + 0.2
        self.fig = plt.figure(figsize=(self.fig_width, self.fig_height))
        self.gs = GridSpec(3, 2, height_ratios=[self.img_height, 0.5, 0.5], width_ratios=[1, 1], figure=self.fig)

        ### Left Column ## ##

        # Cell (0,0): Image
        self.ax1 = self.fig.add_subplot(self.gs[0, 0])
        self.img_plot = self.ax1.imshow(self.img_array, cmap='Greys', aspect='equal')

        self.ax1.set_xlim(0, self.x_range_img)
        self.ax1.set_ylim(self.y_range_img, 0)
        self.ax1.set_xlabel("(um)")
        self.ax1.set_ylabel("(um)")

        self.ax1.xaxis.set_major_locator(MultipleLocator(256/self.binning))
        self.ax1.yaxis.set_major_locator(MultipleLocator(216/self.binning))

        self.ax1.xaxis.set_major_formatter(FuncFormatter(self.scale_x))
        self.ax1.yaxis.set_major_formatter(FuncFormatter(self.scale_y))

        self.plot_scatter(self.ax1, self.kbx, self.kby)

        # Cell (1,0): Text display
        self.ax2 = self.fig.add_subplot(self.gs[1, 0])
        self.ax2.text(0.5, 0.5, "No pixel selected yet",
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform=self.ax2.transAxes)
        self.ax2.set_xticks([])
        self.ax2.set_yticks([])
        self.ax2.set_frame_on(False)

        # Cell (2,0): Button
        self.gss_l = GridSpecFromSubplotSpec(1, 2, subplot_spec=self.gs[2, 0])
        self.ax3_l = self.fig.add_subplot(self.gss_l[0, 0])
        self.save_img_button_all = Button(self.ax3_l, 'Save with all spots')
        self.ax3_r = self.fig.add_subplot(self.gss_l[0, 1])
        self.save_img_button_sel = Button(self.ax3_r, 'Save with selected spots')

        #### Right Column ## ##

        # Cell (0,1): Second figure
        self.ax4 = self.fig.add_subplot(self.gs[0, 1])
        self.ax4.set_xlim(min(self.qvals), max(self.qvals))
        self.ax4.set_ylim(0, 1.0)
        #ax4.set_title('Diffraction dataset: ' + str(diffileno), fontsize=10)
        self.ax4.set_xlabel("Scattering Momentum/Angle")
        self.ax4.set_ylabel("Intensity (counts)")

        # Cell (1,1): Button
        self.ax5 = self.fig.add_subplot(self.gs[1, 1])
        pos = self.ax5.get_position()  # Get the current position of the subplot
        new_pos = [pos.x0, pos.y0 - 0.10, pos.width, pos.height]  # Adjust the y-position
        self.ax5.set_position(new_pos)  # Set the new position
        self.clear_dif_button = Button(self.ax5, 'Clear Diffraction Plot')

        # Cell (2,1): Button
        self.gss_r = GridSpecFromSubplotSpec(1, 2, subplot_spec=self.gs[2, 1])
        self.ax6_l = self.fig.add_subplot(self.gss_r[0, 0])
        self.save_dif_button_leg = Button(self.ax6_l, 'Save with legend')
        self.ax6_r = self.fig.add_subplot(self.gss_r[0, 1])
        self.save_dif_button_cle = Button(self.ax6_r, 'Save without legend')

        #### Connect the events
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.clear_dif_button.on_clicked(self.clear_plots)
        self.save_img_button_all.on_clicked(self.save_img_plot_all)
        self.save_img_button_sel.on_clicked(self.save_img_plot_sel)
        self.save_dif_button_leg.on_clicked(self.save_dif_plot_leg)
        self.save_dif_button_cle.on_clicked(self.save_dif_plot_cle)

        #### Enable interactive mode
        plt.ion()

        #### Adjust layout to prevent overlapping
        plt.tight_layout(pad=0)
        plt.subplots_adjust(top=0.95)  # Adjust the top padding as needed
        plt.show()
