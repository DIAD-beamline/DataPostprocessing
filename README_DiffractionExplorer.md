# Diffraction Data Explorer

**DiffractionDataExplorer** is a Python class designed for interactive exploration and visualization of diffraction and imaging data, particularly from HDF5 files. It is tailored for use in Jupyter notebooks with support for interactive widgets and plots.

## Features

- Load and visualize diffraction and tomography/radiography data from HDF5 files.
- Interactive selection of diffraction spots on imaging data.
- Plot and compare diffraction profiles.
- Save annotated images and diffraction plots.
- GUI-based file selection and parameter input using `ipywidgets`.


## Requirements

Install the required dependencies using pip:

```bash
pip install numpy pandas matplotlib h5py ipywidgets ipympl ipyfilechooser Pillow scipy


numpy
pandas
matplotlib
h5py
ipywidgets
ipympl
ipyfilechooser
Pillow
scipy


## To enable interactive plotting in Jupyter:

jupyter nbextension enable --py widgetsnbextension
jupyter nbextension enable --py ipympl


## Usage


from diffractionDataExplorer_class import DiffractionDataExplorer

explorer = DiffractionDataExplorer()
explorer.InputOutput()  # GUI for file selection
explorer.DataExplorer() # Launch the interactive visualization

