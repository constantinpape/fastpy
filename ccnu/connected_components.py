import numpy as np
from numba import jit
from .union_find import UnionFind


@jit
def connected_components(image):
    shape = image.shape
    n_pixels = image.size
    ufd = UnionFind(range(n_pixels))
    im_flat = image.flatten()

    neighbor_shifts = ((-1, 0), (0, -1), (1, 0), (0, 1))
    for u in range(n_pixels):

        val = im_flat[u]
        if val == 0:
            continue

        # find the representative of the current node
        ru = ufd.find(u)

        # get the coordinate of this node
        coordinate = np.unravel_index(u, shape)
        neighbor_coords = np.array([[coordinate[0] - si[0], coordinate[1] - si[1]]
                                    for si in neighbor_shifts], dtype='int')
        # FIXME this only works for square images
        valid_coords = np.logical_and((neighbor_coords >= 0).all(axis=1),
                                      (neighbor_coords < shape[0]).all(axis=1))
        neighbor_coords = neighbor_coords[valid_coords].T
        neighbors = np.ravel_multi_index(neighbor_coords, shape)
        for v in neighbors:

            if im_flat[v] == 0:
                continue

            # get the neighbor representative
            rv = ufd.find(v)
            # check if the representatives agree - don't need to do anyhing
            if ru == rv:
                continue

            # merge the node's representatives and propagate the seeds
            ufd.union(ru, rv)

    # write the flat segmentation
    components = np.array([ufd.find(u) for u in range(n_pixels)],
                          dtype='uint64')
    return components.reshape(shape)
