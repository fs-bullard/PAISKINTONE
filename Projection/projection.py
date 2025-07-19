import patato as pat
import matplotlib.pyplot as plt
import numpy as np

padata = pat.PAData.from_hdf5("Data/23_5.hdf5")[:, :-1]

image_data = np.squeeze(
    padata.get_scan_reconstructions()["Model Based", "0"].raw_data
)

image_data = np.load('Projection/simulations/simulated_data.npy')


wavelengths = padata.get_wavelengths()
melanin = 519 * (wavelengths / 500) ** (-3.5) * 0.4

n, x, y = image_data.shape
assert n == len(wavelengths)

image_flat = image_data.reshape(n, -1)

# Normalize each pixel spectrum
pixel_norms = np.linalg.norm(image_flat, axis=0, keepdims=True)
image_unit = np.divide(image_flat, pixel_norms, where=pixel_norms != 0)

# Normalize melanin spectrum
melanin_norm = melanin / np.linalg.norm(melanin)

# Compute cosine similarity
cos_sim = np.clip(np.dot(melanin_norm, image_unit).reshape(x, y), 0, 1)


# coeffs = np.dot(melanin, image_flat) / np.linalg.norm(melanin)**2
projection_flat = np.outer(melanin, cos_sim)

projection = projection_flat.reshape(n, x, y)
# coeffs_img = coeffs.reshape(x, y)

# Calculate subtraction (in theory, PAI without mUS clutter)
pai = np.clip(image_data - projection, 0, None)

fig, ax = plt.subplots(3, 3, figsize=(12, 12))

# Compute common colour scale for whole figure
vmin = min(image_data[:].min(), projection[:].min(), pai[:].min())
vmax = max(image_data[:].max(), projection[:].max(), pai[:].max())

j = 0
for i in range(n):
    if i in (2, 5, 10):
        # Plot PAI full
        im0 = ax[j, 0].imshow(image_data[i], cmap='viridis', vmin=vmin, vmax=vmax)
        ax[j, 0].set_title(f"Initial Reconstruction {int(wavelengths[i])} nm")

        # Plot mUS only
        im1 = ax[j, 1].imshow(projection[i], cmap='viridis', vmin=vmin, vmax=vmax)
        ax[j, 1].set_title(f"mUS only {int(wavelengths[i])} nm")

        # Plot PAI only
        im2 = ax[j, 2].imshow(pai[i], cmap='viridis', vmin=vmin, vmax=vmax)
        ax[j, 2].set_title(f"PAI only {int(wavelengths[i])} nm")

        j += 1

plt.tight_layout()
plt.show()
