#include <iostream>
#include <sstream>
#include <vector>
#include <stack>


#include "../../../mitsuba/src/integrators/pmm_guiding/octree.h"
#include "../../../mitsuba/src/integrators/pmm_guiding/envs/2d_env.h"
#include "../../../mitsuba/src/integrators/pmm_guiding/visualizer/AABB.hpp"
#include "../../../mitsuba/src/integrators/pmm_guiding/visualizer/math.hpp"

using Scalar = double;
static constexpr int Dimensionality = 2;


int main() {
    using Tree = pmm_focal::Octree<Env2D>;
    // create an octree
    Tree tree;
    Env2D::AABB newAABB;
    newAABB.min = Env2D::Vector{-3, -3};
    newAABB.max = Env2D::Vector{3, 3};

    // std::cout << newAABB.volume() << std::endl;

    tree.setAABB(newAABB);
    tree.configuration.threshold = 0.0001;
    tree.build();
    // print out
    std::cout << tree.toStringVerbose() << std::endl;
}