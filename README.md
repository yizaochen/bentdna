# bentdna
Calculating DNA bending properties by using coarse-grained model

# Create a virtual environment
`conda create --name bentdna python=3.8`

# Activate virtual environment
`conda activate bentdna`

# Deactivate
`conda deactivate`

# Install bentdna package
`pip install -e .`

# Upgrade package
`pip install -e . --upgrade`

# Location of site-packages
/home/yizaochen/miniconda3/envs/bentdna/lib/python3.8/site-packages

# Protocol
### Part 1: θ(s) plot
`notebooks/theta_s_plot.ipynb`
### Part 2: Fourier Decomposition
`notebooks/fourier_analysis.ipynb`
### Part 3: Use CURVES+ to find helical axis
`notebooks/curve_interface.ipynb`
### Part 3(Optional): Process a lot of structures(like trajectory)
`python_scripts/initialize_allsystems.py`
`python_scripts/curve_batch.py`

### Part 4: Make Dataframe for $|l_{i}|$, $|l_{j}|$, $\theta_{i,j}$ of all frames
`notebooks/make_dataframe_from_curve.ipynb`

# For plotly

### Install Node.js
`conda install -c conda-forge nodejs`
### JupyterLab
`jupyter labextension install jupyterlab-plotly@4.9.0`
