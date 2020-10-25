# Alphabeta
Foward-backward algorithm on FRET

# Create a virtual environment
`conda create --name Alphabeta python=3.8`

# Activate virtual environment
`conda activate Alphabeta`

# Protocol
### Check grid size of FEM
`notebooks/FEM_grid_size.ipynb`

### Check eigenvalues and eigenvectors
`notebooks/inspect_eigenvalues.ipynb`

### Simulation
`notebooks/simulation.ipynb`

### EM
`notebooks/EM_1.ipynb`

### Assemble Gaussian by Eigenvectors
`notebooks/assemble_gaussian_test.ipynb`

### Smooth
#### Using central finite difference to find first derivative and second derivative
- By [FiniteDifferences.jl](https://github.com/JuliaDiff/FiniteDifferences.jl)
- `notebooks/finitediff_test.ipynb`