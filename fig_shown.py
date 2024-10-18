import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

# Data
models = ['FusionNet', 'TDNet', 'PMRF', 'Proposed']
Q8 = [0.8936, 0.8936, 0.9005, 0.9387]
FLOPs = [9.92, 19.98, 32.62, 6.07]
NoP = [0.23, 0.49, 0.39, 0.1]

# Convert data to numpy arrays
Q8 = np.array(Q8)
FLOPs = np.array(FLOPs)
NoP = np.array(NoP)

# Create grid values first.
FLOPs_grid, NoP_grid = np.meshgrid(np.linspace(min(FLOPs), max(FLOPs), 100),
                                   np.linspace(min(NoP), max(NoP), 100))

# Interpolate the Q8 values on the grid
Q8_grid = griddata((FLOPs, NoP), Q8, (FLOPs_grid, NoP_grid), method='cubic')

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(FLOPs_grid, NoP_grid, Q8_grid, cmap='viridis', edgecolor='none')

# Scatter plot
ax.scatter(FLOPs, NoP, Q8, color='r', s=50)

# Labels
for i, model in enumerate(models):
    ax.text(FLOPs[i], NoP[i], Q8[i], model, color='red')

ax.set_xlabel('FLOPs (G)')
ax.set_ylabel('NoP (M)')
ax.set_zlabel('Q8')

plt.title('3D Surface Plot of Model Performance Metrics')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)  # Add a color bar which maps values to colors.
plt.show()
