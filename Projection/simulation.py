'''
Simple simulated data with melanin components and non-melanin components.
'''


import numpy as np
import matplotlib.pyplot as plt

# Define melanin spectrum
wavelengths = np.array([ 700.,  730.,  760.,  800.,  850.,  910.,  930.,  950.,  980.,
       1030., 1080., 1100.])
melanin = 519 * (wavelengths / 500) ** (-3.5) * 0.4
plt.plot(wavelengths, melanin / np.linalg.norm(melanin)**2)
plt.show()

# Initialise data matrix
z, x, y = len(wavelengths), 400, 400
data = np.zeros((z, x, y))

# Define rectangle (melanin spectrum)
rect_top, rect_bottom = 50, 70
rect_left, rect_right = 150, 250
for i in range(z):
    data[i, rect_top:rect_bottom, rect_left:rect_right] = melanin[i] * 2

# Define left circle (melanin spectrum)
xx, yy = np.meshgrid(np.arange(x), np.arange(y), indexing='ij')
circle_left_center = (320, 130)
circle_radius = 20
mask_left = (xx - circle_left_center[0])**2 + (yy - circle_left_center[1])**2 <= circle_radius**2
for i in range(z):
    data[i][mask_left] = melanin[i]

# Define right circle (constant across z)
circle_right_center = (320, 270)
mask_right = (xx - circle_right_center[0])**2 + (yy - circle_right_center[1])**2 <= circle_radius**2
data[:, mask_right] = 50  # constant value across z

# Visualise a few slices
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for i, idx in enumerate([0, 6, 11]):
    ax[i].imshow(data[idx], cmap='viridis', clim=(0, 100))
    ax[i].set_title(f"Slice {idx} ({int(wavelengths[idx])} nm)")

np.save('Projection/simulations/simulated_data.npy', data)
plt.tight_layout()
plt.show()
