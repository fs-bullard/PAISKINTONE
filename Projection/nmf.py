'''
Separating the melanin and PA signals with non-negative matrix factorisation
'''

import patato as pat
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import NMF

padata = pat.PAData.from_hdf5("Data/19_5.hdf5")[:, :-1]

image_data = np.squeeze(
    padata.get_scan_reconstructions()["Model Based", "0"].raw_data
)

# image_data = np.load('Projection/simulations/simulated_data.npy')


wavelengths = padata.get_wavelengths()
melanin = 519 * (wavelengths / 500) ** (-3.5) * 0.4

n, x, y = image_data.shape
assert n == len(wavelengths)

D = np.clip(image_data.reshape(n, -1), 0, None)

n_components = 4

nmf = NMF(n_components=n_components, init='nndsvd', random_state=0)
W = nmf.fit_transform(D)  # shape: (n, n_components) → learned spectra
H = nmf.components_       # shape: (n_components, pixels) → spatial weights

print(f'W shape: {W.shape}, H shape: {H.shape}')

for i in range(n_components):
    # plt.plot(wavelengths, W[:,i])
    plt.imshow(H[i].reshape(x, y))
    plt.show()