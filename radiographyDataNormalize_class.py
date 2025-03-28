from IPython.display import display
from ipyfilechooser import FileChooser
import h5py
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import IntSlider, Text, HBox, VBox, Output, Button, Layout, Label

class RadiographyDataNormalize:
    def __init__(self, raw_data_path=None, flat_data_path=None, dark_data_path=None, dest_root_path=None):
        self.raw_data_path = raw_data_path
        self.flat_data_path = flat_data_path
        self.dark_data_path = dark_data_path
        self.dest_root_path = dest_root_path

    def load_raw_data_path(self, chooser):
        if chooser.selected:
            self.raw_data_path = chooser.selected

    def load_flat_data_path(self, chooser):
        if chooser.selected:
            self.flat_data_path = chooser.selected

    def load_dark_data_path(self, chooser):
        if chooser.selected:
            self.dark_data_path = chooser.selected

    def load_dest_root_path(self, chooser):
        if chooser.selected:
            self.dest_root_path = chooser.selected

    def InputOutput(self):
        if self.raw_data_path is None:
            dfile_chooser = FileChooser()
            dfile_chooser.title = 'Raw Radiography Data:'
            dfile_chooser.register_callback(self.load_raw_data_path)
            display(dfile_chooser)

        if self.flat_data_path is None:
            ifile_chooser = FileChooser()
            ifile_chooser.title = 'Flat Field Radiography Data:'
            ifile_chooser.register_callback(self.load_flat_data_path)
            display(ifile_chooser)

        if self.dark_data_path is None:
            ofile_chooser = FileChooser()
            ofile_chooser.title = 'Dark Field Radiography Data):'
            ofile_chooser.register_callback(self.load_dark_data_path)
            display(ofile_chooser)

        if self.dest_root_path is None:
            destfile_chooser = FileChooser()
            destfile_chooser.title = 'Destination Root File Path:'
            destfile_chooser.register_callback(self.load_dest_root_path)
            display(destfile_chooser)

    def ReadData(self):
        # Load flat data
        with h5py.File(self.flat_data_path, 'r') as f:
            image_keys = f['/entry/instrument/imaging/image_key'][()]
            indices = (image_keys == 1)
            flat_data = np.mean(f['/entry/imaging/data'][indices], axis=0)

        # Load dark data
        with h5py.File(self.dark_data_path, 'r') as f:
            image_keys = f['/entry/instrument/imaging/image_key'][()]
            indices = (image_keys == 2)
            dark_data = np.mean(f['/entry/imaging/data'][indices], axis=0)

        # Load and process raw data
        with h5py.File(self.raw_data_path, 'r') as f:
            image_keys = f['/entry/instrument/imaging/image_key'][()]
            indices = (image_keys == 0)
            self.raw_data = f['/entry/imaging/data'][indices]
            self.raw_data_normalized = (self.raw_data - dark_data) / (flat_data - dark_data)

    def save_snapshot(self, frame):
        plt.figure(figsize=(10, 8))
        plt.imshow(self.raw_data_normalized[frame], cmap='gray')
        plt.axis('off')
        plt.savefig(f'{self.dest_root_path}_{frame}.png', bbox_inches='tight', pad_inches=0)
        plt.close()

    def on_save_button_clicked(self, b):
        frame = self.slider.value
        self.save_snapshot(frame)

    def save_all_frames(self):
        with h5py.File(f'{self.dest_root_path}_all_frames.h5', 'w') as f:
            f.create_dataset('normalized_frames', data=self.raw_data_normalized)

    def RadiographyExplorer(self):
        self.ReadData()  # Read and normalize the data
        
        # Create a slider for frame selection
        self.slider = IntSlider(min=0, max=len(self.raw_data_normalized)-1, layout=Layout(width='200px'), readout=False)
        
        # Create a text input for frame number and initialize it with the slider's initial value
        self.text = Text(value=str(self.slider.value), layout=Layout(width='100px'))
        
        # Create an output widget for displaying plots
        self.output = Output()
        
        # Create a button to save the current snapshot
        save_button = Button(description="Save Snapshot", layout=Layout(width='120px'))
        save_button.on_click(self.on_save_button_clicked)
        
        # Create a button to save all frames
        save_all_button = Button(description="Save All Frames", layout=Layout(width='120px'))
        save_all_button.on_click(lambda b: self.save_all_frames())
        
        # Link slider and text input
        def update_text(change):
            self.text.value = str(change['new'])
            self.update_plot(change['new'])
        
        def update_slider(change):
            self.slider.value = int(change['new'])
            self.update_plot(int(change['new']))
        
        self.slider.observe(update_text, names='value')
        self.text.observe(update_slider, names='value')
        
        # Display the title, slider, text input, save button, and save all button together
        display(VBox([HBox([Label('Frame:'), self.slider, self.text, save_button, save_all_button]), self.output]))
        
        # Initial plot
        self.update_plot(self.slider.value)
    
    #def update_plot(self, frame):
    #    with self.output:
    #        self.output.clear_output()
    #        plt.figure(figsize=(10, 8))
    #        plt.imshow(self.raw_data_normalized[frame], cmap='gray')
    #        plt.axis('off')
    #        plt.show()

    def update_plot(self, frame):
        with self.output:
            self.output.clear_output()  # Clear the previous output
            fig, axes = plt.subplots(1, 2, figsize=(15, 8))  # Create two subplots side by side
    
            # Display the normalized radiography image
            axes[0].imshow(self.raw_data_normalized[frame], cmap='gray')
            axes[0].set_title("Normalized Radiography")
            axes[0].axis('off')
    
            # Display the original raw radiography data
            axes[1].imshow(self.raw_data[frame], cmap='gray')
            axes[1].set_title("Original Raw Radiography")
            axes[1].axis('off')
    
            plt.show()  # Show the updated figure
