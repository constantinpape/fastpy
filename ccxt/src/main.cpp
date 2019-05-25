#include "pybind11/pybind11.h"

#include "xtensor/xmath.hpp"
#include "xtensor/xarray.hpp"
#include <boost/pending/disjoint_sets.hpp>

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pytensor.hpp"
#include "xtensor-python/pyarray.hpp"

#include <iostream>
#include <numeric>
#include <cmath>

namespace py = pybind11;


xt::pyarray<uint64_t> connected_components(const xt::pytensor<bool, 2> & image)
{
    typedef typename xt::pytensor<bool, 2>::shape_type IndexType;
    const auto & shape = image.shape();
    const std::size_t n_nodes = shape[0] * shape[1];

    // make union find and map seeds to reperesentatives
    std::vector<uint64_t> ranks(n_nodes);
    std::vector<uint64_t> parents(n_nodes);
    boost::disjoint_sets<uint64_t*, uint64_t*> ufd(&ranks[0], &parents[0]);
    for(uint64_t node = 0; node < n_nodes; ++node) {
        ufd.make_set(node);
    }

    // flatten the image
    auto flat_image = xt::flatten(image);

    const int n_ngbs = 4;
    std::vector<int> shifts_x = {-1, 1, 0, 0};
    std::vector<int> shifts_y = {0, 0, -1, 1};

    // run connected components
    for(uint64_t u = 0; u < n_nodes; ++u) {
        if(flat_image[u] == 0) {
            continue;
        }

        // get representative
        const uint64_t ru = ufd.find_set(u);

        const auto coordinate = xt::unravel_index(u, shape);
        std::vector<IndexType> neighbor_coords;

        // make the neighbors
        for(unsigned ngb = 0; ngb < n_ngbs; ++ngb) {
            const int sx = shifts_x[ngb];
            const int sy = shifts_y[ngb];
            const int64_t x = coordinate[0] + sx;
            const int64_t y = coordinate[1] + sy;

            // bounds check
            if(sx < 0 || sx >= shape[0]) {
                continue;
            }
            if(sy < 0 || sy >= shape[1]) {
                continue;
            }

            neighbor_coords.emplace_back(IndexType({x, y}));
        }

        const auto neighbors = xt::ravel_indices(neighbor_coords, shape);
        // iterate over the neighbors
        for(const uint64_t v: neighbors) {
            if(flat_image[v] == 0) {
                continue;
            }

            const uint64_t rv = ufd.find_set(v);
            if(ru == rv) {
                continue;
            }

            ufd.link(ru, rv);
        }
    }

    xt::pyarray<uint64_t> seg = xt::zeros<uint64_t>({n_nodes});
    for(uint64_t u = 0; u < n_nodes; ++u) {
        seg[u] = ufd.find_set(u);    
    }
    seg.reshape(shape);
    return seg;
}



PYBIND11_MODULE(ccxt, m)
{
    xt::import_numpy();

    m.doc() = R"pbdoc(
        ccxt

        .. currentmodule:: ccxt

        .. autosummary::
           :toctree: _generate

           connected_components
    )pbdoc";
    m.def("connected_components", connected_components, "compute connected_components");
}
