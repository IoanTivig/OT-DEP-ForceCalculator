import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import plotly.graph_objects as go

# Step 1: Load Data
# Assuming the file "data_opendep.csv" is in the same directory as this script
data = pd.read_csv("data_opendep.csv")

# Extract columns into arrays
x = data['x'].values       # Array of x values
z = data['z'].values       # Array of z values
E = data['E'].values       # Array of E values

# Step 2: Define the Rational Function Model
def rational_fit(coords, a, b, c, d, e, f):
    x, z = coords
    return (a * x + b * z + c) / (d * x + e * z + f)

# Combine x and z into a single array for fitting
coords = np.vstack((x, z))

# Fit the Rational Function Model
params_rational, _ = curve_fit(rational_fit, coords, E, maxfev=10000)

# Extract coefficients
coefficients = {
    'a': params_rational[0],
    'b': params_rational[1],
    'c': params_rational[2],
    'd': params_rational[3],
    'e': params_rational[4],
    'f': params_rational[5]
}

# Print coefficients for copying
print("Coefficients:")
for key, value in coefficients.items():
    print(f"{key} = {value:.6f}")

# Generate formula with coefficients integrated
formula = f"E(x, z) = ({coefficients['a']:.6f} * x + {coefficients['b']:.6f} * z + {coefficients['c']:.6f}) / " \
          f"({coefficients['d']:.6f} * x + {coefficients['e']:.6f} * z + {coefficients['f']:.6f})"

print("\nIntegrated Formula:")
print(formula)

# Predict the values using the fitted model
E_pred_rational = rational_fit(coords, *params_rational)

# Step 3: Calculate R-Squared
r2_rational = r2_score(E, E_pred_rational)
print(f"\nR² (Coefficient of Determination): {r2_rational:.4f}")

# Step 4: Visualize the Results
# Create a grid for visualization
x_unique = np.linspace(np.min(x), np.max(x), 50)
z_unique = np.linspace(np.min(z), np.max(z), 50)
X, Z = np.meshgrid(x_unique, z_unique)
coords_grid = np.vstack((X.ravel(), Z.ravel()))

# Evaluate the fitted function on the grid
E_fit_rational = rational_fit(coords_grid, *params_rational).reshape(X.shape)

# Use Plotly for interactive visualization
fig = go.Figure()

# Add the surface plot
fig.add_trace(go.Surface(z=E_fit_rational, x=X, y=Z, colorscale='Viridis', showscale=True, name='Fitted Surface'))

# Add scatter points for original data
fig.add_trace(go.Scatter3d(x=x, y=z, z=E, mode='markers',
                           marker=dict(size=4, color='blue'), name='Original Data'))

# Update layout
fig.update_layout(scene=dict(
    xaxis_title='x (Distance from Electrode)',
    yaxis_title='z (Distance from Surface)',
    zaxis_title='E (Electric Field)'
), title=f'3D Fit of Electric Field (Rational Function Model, R² = {r2_rational:.4f})')

# Show the plot
fig.show()
