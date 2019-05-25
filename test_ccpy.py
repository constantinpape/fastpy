import numpy as np
from skimage import feature
from skimage.data import astronaut

from ccxt import connected_components


data = astronaut()[:256, :256]
edges = feature.canny(data[..., 0] / 255.)
binary = edges < .5

print("Run cc ....")
ccp = connected_components(binary)
print("Done ....")
print(ccp.shape)
print(np.unique(ccp))
