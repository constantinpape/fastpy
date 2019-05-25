import numpy as np
from .union_find import UnionFind


def watershed(weights, seeds):
    assert weights.shape == seeds.shape
    shape = weights.shape

    # flatten the height map and the weights
    weights = weights.ravel()
    seeds = seeds.ravel()
    # sort the nodes by weight
    sorted_nodes = np.argsort(weights)

    n_pixels = seeds.size
    ufd = UnionFind(range(n_pixels))

    neighbor_shifts = ((-1, 0), (0, -1), (1, 0), (0, 1))
    for u in sorted_nodes:

        # find the representative of the current node
        ru = ufd.find(u)

        # is the representative seeded ?
        seed_u = seeds[ru]
        # yes: make sure that node has no seed or the same seed
        if seed_u != 0:
            assert seeds[u] == 0 or seeds[u] == seed_u, "%i, %i : %i, %i" % (ru, u, seed_u, seeds[u])
        # no: check if we have encoutered a new seed and update the representative seed if so
        else:
            seed_u = seeds[u]
            if seed_u != 0:
                seeds[ru] = seed_u

        # get the coordinate of this node
        coordinate = np.unravel_index(u, shape)
        # print(u)
        # print(coordinate)
        neighbor_coords = np.array([[coordinate[0] - si[0], coordinate[1] - si[1]]
                                    for si in neighbor_shifts], dtype='int')

        # FIXME this only works for square images
        valid_coords = np.logical_and((neighbor_coords >= 0).all(axis=1),
                                      (neighbor_coords < shape[0]).all(axis=1))
        neighbor_coords = neighbor_coords[valid_coords].T
        neighbors = np.ravel_multi_index(neighbor_coords, shape)
        # print(neighbor_coords)
        # print(neighbors)
        # iterate over neighboring nodes
        for v in neighbors:
            # get the neighbor representative
            rv = ufd.find(v)
            # check if the representatives agree - don't need to do anyhing
            if ru == rv:
                continue

            # get the neighbor seed
            seed_v = seeds[rv]

            # don't link the nodes if both are seeded and have different seeds
            if seed_v != 0 and seed_u != 0 and seed_u != seed_v:
                print(u, v)
                print(ru, rv)
                print(seed_u, seed_v)
                print()
                continue

            # merge the node's representatives and propagate the seeds
            ufd.union(ru, rv)
            nseed = seed_u if seed_u != 0 else seed_v if seed_v != 0 else 0
            seeds[ru] = nseed
            seeds[rv] = nseed

    # write the flat segmentation
    segmentation = np.array([ufd.find(u) for u in range(n_pixels)],
                            dtype=seeds.dtype)
    # segmentation = np.array([seeds[ufd.find(u)] for u in range(n_pixels)],
    #                         dtype=seeds.dtype)
    return segmentation.reshape(shape)
