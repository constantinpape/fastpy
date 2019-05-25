import numpy as np
from scipy.ndimage import distance_transform_edt, label
from skimage import feature
from skimage import filters
from skimage.data import astronaut

# from wspy import watershed
from wsxt import watershed


data = astronaut()[:256, :256]
edges = feature.canny(data[..., 0] / 255.)
distances = distance_transform_edt(edges < .25)
distances = filters.gaussian(distances, 2.)

seeds = feature.peak_local_max(distances, indices=False, footprint=np.ones((3, 3)))
seeds, _ = label(seeds)

print("Run ws ....")
wsp = watershed(edges, seeds)
print("Done ....")
print(wsp.shape)
print(np.unique(wsp))
