#include "pybind11/pybind11.h"

#include "xtensor/xmath.hpp"
#include "xtensor/xarray.hpp"
#include <boost/pending/disjoint_sets.hpp>

#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pytensor.hpp"

#include <iostream>
#include <numeric>
#include <cmath>

namespace py = pybind11;


xt::pytensor<uint64_t, 2> watershed(const xt::pytensor<float, 2> & weights,
                                    xt::pytensor<uint64_t, 2> & seeds)
{
    typedef typename xt::pytensor<float, 2>::shape_type IndexType;
    const auto & shape = weights.shape();
    const std::size_t n_nodes = shape[0] * shape[1];

    // make union find and map seeds to reperesentatives
    std::vector<uint64_t> ranks(n_nodes);
    std::vector<uint64_t> parents(n_nodes);
    boost::disjoint_sets<uint64_t*, uint64_t*> ufd(&ranks[0], &parents[0]);
    for(uint64_t node = 0; node < n_nodes; ++node) {
        ufd.make_set(node);
    }

    // argsort the edges by edge weight
    auto flat_weights = xt::flatten(weights);
    auto flat_seeds = xt::flatten(seeds);

    std::vector<std::size_t> argsorted(n_nodes);
    std::iota(argsorted.begin(), argsorted.end(), 0);
    std::sort(argsorted.begin(), argsorted.end(), [&](const std::size_t a,
                                                      const std::size_t b){
        return flat_weights[a] < flat_weights[b];}
    );

    const int n_ngbs = 4;
    std::vector<int> shifts_x = {-1, 1, 0, 0};
    std::vector<int> shifts_y = {0, 0, -1, 1};

    // run kruskal
    for(const uint64_t u : argsorted) {
        // get representative
        const uint64_t ru = ufd.find_set(u);

        // get seed
        uint64_t seed_u = flat_seeds[ru];
        if(seed_u == 0) {
            seed_u = seeds[u];
            flat_seeds[ru] = seed_u;
        } else {
            const uint64_t seed_uu = seeds[u];
            if(seed_uu != 0 && seed_uu != seed_u) {
                std::cout << u << ", " << ru << " : " << seed_uu << ", " << seed_u << std::endl;
                throw std::runtime_error("Seeds disagree!");
            }
        }

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
            const uint64_t rv = ufd.find_set(v);
            if(ru == rv) {
                continue;
            }

            const uint64_t seed_v = flat_seeds[v];
            if(seed_u != 0 && seed_v != 0 && (seed_v != seed_u)) {
                continue;
            }

            ufd.link(ru, rv);
            if(seed_u != 0) {
                flat_seeds[ru] = seed_u;
                flat_seeds[rv] = seed_u;
            }
            else if(seed_v != 0) {
                flat_seeds[ru] = seed_v;
                flat_seeds[rv] = seed_v;
            }
        }
    }

    xt::pytensor<uint64_t, 2> seg = xt::zeros<uint64_t>(shape);
    return seg;
}

PYBIND11_MODULE(wsxt, m)
{
    xt::import_numpy();

    m.doc() = R"pbdoc(
        wsxt

        .. currentmodule:: wsxt

        .. autosummary::
           :toctree: _generate

           watershed
    )pbdoc";

    m.def("watershed", watershed, "compute watershed");
}
