/** Based on Rath. et. al. 2020 */
/**
 * An adaptive spatial density represented by a hyperoctree (quad-tree in 2-D, octree in 3-D).
 * Leaves are sub-divided when enough energy is present (similar to the D-Tree of Müller et al. [2017]).
 */

namespace pmm_focal {

template<typename Env>
class Octree {
    static constexpr int Dimensionality = Env::Dimensionality;

    using Float = typename Env::Float;
    using Vector = typename Env::Vector;
    using Point = typename Env::Point;
    using AABB = typename Env::AABB;
    using PRNG = typename Env::PRNG;

        /**
     * Represents a piece of constant density of this density.
     * Used for visualization of densities.
     */
    struct Patch {
        AABB domain;
        Float density{};

        Patch() = default;
        Patch(const AABB &domain, Float density) : domain(domain), density(density) {}
    };

    AABB m_aabb;

public:
    struct Configuration {
        /// For compatibility with older file formats.
        bool unused{false};

        /// The spatial threshold for splitting regions of the octree. For more details, refer to our paper.
        Float threshold{0.001};

        /// Exponential decay that is applied after each training iteration, so that information is not lost completely.
        Float decay{0.5};

        /// The minimum depth of the data structure (e.g., for pre-splitting).
        int minDepth{0};

        /// The maximum depth of the data structure, which limits the maximum resolution (and computational overhead).
        int maxDepth{14};

        /// Whether to merge nodes with little density variation among their children, useful in the last iteration of training.
        bool pruning{false};
    };

    Configuration configuration;

    Octree() : builder(*this) {
        Octree::clear();
    }

    Octree &operator=(const Octree &other) {
        m_nodes = other.m_nodes;
        return *this;
    }

    const AABB &aabb() const {
        return m_aabb;
    }

    /**
     * Propagates all sample weight accumulated in the leaf nodes up the entire tree and returns the @b absolute splitting threshold
     * that the weight in a leaf needs to exceed to be split.
     */
    Float sumDensities() {
        builder.sumDensities();
        return builder.splittingThreshold;
    }

    /**
     * Sums up the weights accumulated in the leaf nodes and updates the spatial density.
     */
    void build() {
        builder.sumDensities();
        builder.build();
    }

    /**
     * Updates the spatial density given a provided @b absolute splitting threshold, does @b not propagate the weight accumulated
     * in children up the tree.
     * This method is useful if multiple spatial structures should sum up their root weight for determining the splitting
     * threshold (most notably when a converging and a diverging field are used in tandem).
     * In that case, call @c Orthree::sumDensities on all spatial structures, sum up the absolute splitting thresholds returned
     * and call this method on all structures with the summed up thresholds.
     */
    void build(Float threshold) {
        builder.splittingThreshold = threshold;
        builder.build();
    }

    void clear() {
        m_nodes.clear();
        m_nodes.emplace_back();
        build();
    }

    void splat(const Point &origin, const Vector &direction, Float distance, std::vector<AABB>& leafAABBs) {
        Float alpha = 1;

        Traversal traversal{*this, origin, direction};
        distance = std::min(distance, traversal.maxT());

        traversal.traverse(distance, [&](
            NodeIndex nodeIndex, StratumIndex stratum, Float tNear, Float tFar
        ) {
            auto &child = m_nodes[nodeIndex].children[stratum];
            if (child.isLeaf()) {
                leafAABBs.push_back(child.computeLeafAABB(nodeIndex, stratum));
            }
        });
    }

    [[nodiscard]] std::vector<Patch> visualize() const {
        std::vector<Patch> result;

        struct StackEntry {
            AABB domain;
            NodeIndex nodeIndex;
        };

        std::stack<StackEntry> stack;
        stack.push({
            this->aabb(),
            0
        });

        while (!stack.empty()) {
            const StackEntry stackEntry = stack.top();
            stack.pop();

            for (StratumIndex stratum = 0; stratum < Arity; stratum++) {
                auto &child = m_nodes[stackEntry.nodeIndex].children[stratum];
                AABB childDomain{};
                for (int dim = 0; dim < Dimensionality; dim++) {
                    const Float min = stackEntry.domain.min[dim];
                    const Float max = stackEntry.domain.max[dim];
                    const Float mid = (min + max) / 2;

                    if ((stratum >> dim) & 1) {
                        childDomain.min[dim] = mid;
                        childDomain.max[dim] = max;
                    } else {
                        childDomain.min[dim] = min;
                        childDomain.max[dim] = mid;
                    }
                }

                if (child.isLeaf()) {
                    result.push_back({
                        childDomain,
                        child.density
                    });
                } else {
                    stack.push({
                        childDomain,
                        child.index
                    });
                }
            }
        }

        return result;
    }

private:
    static constexpr int Arity = 1 << Dimensionality;

    using NodeIndex = uint32_t;
    using StratumIndex = uint8_t;

    struct Node {
        struct Child {
            NodeIndex index{0};
            Float accumulator{};

            union {
                // we differentiate between the two depending on context,
                // densityTimesVolume is only used while building
                Float density{};
                Float densityTimesVolume;
            };

            [[nodiscard]] bool isLeaf() const {
                return index == 0;
            }

            AABB computeLeafAABB(NodeIndex nodeIndex, StratumIndex stratum) {
                const Point origin = {0.0f, 0.0f, 0.0f};
                Point minPoint, maxPoint;
                Float currentStepsize = 1;

                for (int dim = 0; dim < Dimensionality; ++dim) {
                    if (stratum & (1 << dim)) {
                        minPoint[dim] = origin[dim] + (currentStepsize / 2);
                        maxPoint[dim] = origin[dim] + currentStepsize;
                    } else {
                        minPoint[dim] = origin[dim];
                        maxPoint[dim] = origin[dim] + (currentStepsize / 2);
                    }
                }

                return AABB{minPoint, maxPoint}; // note: this will return AABB in [0,1] scale, each point should be scaled by tree AABB to achieve true coordinates
            }
        };

        Child children[Arity];

        /**
         * Looks up which child index (stratum) a point in [0,1)^n lies in,
         * and renormalizes the position so that it spans the containing child domain.
         */
        static StratumIndex lookup(Point &pos) {
            StratumIndex stratum = 0;
            for (int dim = 0; dim < Dimensionality; dim++) {
                const int bit = pos[dim] >= 0.5f;
                stratum |= bit << dim;
                pos[dim] = pos[dim] * 2 - Float(bit);
            }
            return stratum;
        }

        /**
         * Samples a point using hierarchical sample warping [McCool and Harwood 1997].
         */
        NodeIndex sample(Vector &sample, Point &origin, Float stepsize) const {
            int childIndex = 0;

            // sample each axis individually to determine sampled child
            for (int dim = 0; dim < Dimensionality; ++dim) {
                // marginalize over remaining dimensions {dim+1..Dimension-1}
                Float p[2] = {0, 0};
                for (int child = 0; child < (1 << (Dimensionality - dim)); ++child) {
                    // we are considering only children that match all our
                    // chosen dimensions {0..dim-1} so far.
                    // we are collecting the sum of density for children with
                    // x[dim] = 0 in p[0], and x[dim] = 1 in p[1].
                    const int ci = (child << dim) | childIndex;
                    p[child & 1] += children[ci].density;
                }

                assert(p[0] >= 0 && p[1] >= 0);
                assert((p[0] + p[1]) > 0);

                p[0] /= p[0] + p[1];

                const int slab = sample[dim] >= p[0];
                childIndex |= slab << dim;

                if (slab) {
                    origin[dim] += stepsize / 2;
                    sample[dim] = (sample[dim] - p[0]) / (1 - p[0]);
                } else {
                    sample[dim] = sample[dim] / p[0];
                }

                if (sample[dim] >= 1)
                    sample[dim] = std::nextafterf(1, 0);

                assert(sample[dim] >= 0);
                assert(sample[dim] < 1);
            }

            return childIndex;
        }
    };

    /**
     * Based on "An Efficient Parametric Algorithm for Octree Traversal" [Revelles et al. 2000].
     */
    struct Traversal {
    private:
        const Octree &tree;
        StratumIndex a; // bitmask indicating which dimensions are reversed
        Vector tNear, tFar;

        [[nodiscard]] static StratumIndex firstNode(const Vector &tNear, const Vector &tMid) {
            const int maxDimension = Env::argmin(tNear);
            const Float maxValue = tNear[maxDimension];

            StratumIndex result = 0;
            for (int dim = 0; dim < Dimensionality; dim++) {
                if (dim == maxDimension) continue;
                if (tMid[dim] < maxValue) result |= 1 << ((Dimensionality - 1) - dim);
            }
            return result;
        }

        [[nodiscard]] static StratumIndex newNode(StratumIndex currNode, const Vector &tFar) {
            const int exitDimension = Env::argmin(tFar);
            const StratumIndex flag = 1 << exitDimension;
            if (currNode & flag)
                return Arity; // END
            return currNode | flag;
        }

        template<typename F>
        void traverse(NodeIndex nodeIndex, const Vector &tNear, const Vector &tFar, Float tMax, F &&processTerminal) const {
            if (Env::min(tFar) < 0) return;
            if (Env::max(tNear) > tMax) return;

            const Vector tMid = (tNear + tFar) / 2;
            StratumIndex currNode = firstNode(tNear, tMid);
            do {
                Vector tChildNear;
                Vector tChildFar;
                for (int dim = 0; dim < Dimensionality; dim++) {
                    if ((currNode >> dim) & 1) {
                        tChildNear[dim] = tMid[dim];
                        tChildFar[dim] = tFar[dim];
                    } else {
                        tChildNear[dim] = tNear[dim];
                        tChildFar[dim] = tMid[dim];
                    }
                }

                auto &child = tree.m_nodes[nodeIndex].children[a ^ currNode];
                if (child.isLeaf()) {
                    const Float t0 = Env::max(tChildNear);
                    const Float t1 = Env::min(tChildFar);
                    if (t1 >= 0 && t0 < t1 && t0 < tMax)
                        processTerminal(nodeIndex, a ^ currNode, std::max(t0, Float(0)), std::min(t1, tMax));
                } else {
                    traverse(child.index, tChildNear, tChildFar, tMax, processTerminal);
                }
                currNode = newNode(currNode, tChildFar);
            } while (currNode < Arity);
        }

    public:
        explicit Traversal(const Octree &tree, Point origin, Vector direction) : tree(tree) {
            a = 0;
            for (int dim = 0; dim < Dimensionality; dim++) {
                if (direction[dim] == 0) direction[dim] = 1e-10; // hack
                if (direction[dim] > 0)
                    continue;

                origin[dim] = (tree.aabb().max[dim] + tree.aabb().min[dim]) - origin[dim];
                direction[dim] = -direction[dim];
                a |= 1 << dim;
            }

            tNear = Env::divide(tree.aabb().min - origin, direction);
            tFar = Env::divide(tree.aabb().max - origin, direction);
        }

        float minT() const { return Env::max(tNear); }
        float maxT() const { return Env::min(tFar); }

        template<typename F>
        void traverse(Float tMax, F &&processTerminal) {
            if (Env::max(tNear) < Env::min(tFar) && Env::max(tNear) < tMax)
                traverse(0, tNear, tFar, tMax, processTerminal);
        }
    };

    struct Builder {
        explicit Builder(Octree &tree) : tree(tree) {}

        void sumDensities() {
            if (Env::volume(tree.aabb()) == 0) {
                printf("empty volume\n");
                return;
            }

            rootChildVolume = std::min(std::abs(Env::volume(tree.aabb())), Float(1e+20)) / Arity;

            keepNodes.resize(tree.m_nodes.size());
            std::fill(keepNodes.begin(), keepNodes.end(), true);

            maxDensities.resize(tree.m_nodes.size());

            Float rootAccumulator;
            rootWeight = sumDensities(0, rootAccumulator, rootChildVolume);
            splittingThreshold = tree.configuration.threshold * rootWeight;

            printf("root weight: %.3e\n", rootWeight);
        }

        void build() {
            if (rootChildVolume == 0) {
                printf("trying to build tree without samples!\n");
                return;
            }

            const auto nodesBeforeSplit = NodeIndex(tree.m_nodes.size());
            build(0, 0, rootChildVolume);
            const auto nodesAfterSplit = NodeIndex(tree.m_nodes.size());
            pruneTree();
            const auto nodesAfterPrune = NodeIndex(tree.m_nodes.size());

            printf("node count: %d -> %d -> %d\n", nodesBeforeSplit, nodesAfterSplit, nodesAfterPrune);
        }

        Octree &tree;
        Float rootChildVolume{};
        Float rootWeight{};
        Float splittingThreshold{};
        std::vector<bool> keepNodes;
        std::vector<Float> maxDensities;

        Float sumDensities(const NodeIndex index, Float &accumulator, Float childVolume = 1.f) {
            Float sum = 0;
            accumulator = 0;
            Float nodeMaxDensity = 0;
            for (auto &child: tree.m_nodes[index].children) {
                child.accumulator = std::max(child.accumulator, Float(1e-20)); // hack to avoid numerical issues
                accumulator += child.accumulator;

                // we are now switching to densityTimesVolume
                child.densityTimesVolume = child.accumulator;

                Float childMaxDensity = child.densityTimesVolume / childVolume;
                if (!child.isLeaf()) {
                    child.densityTimesVolume = sumDensities(
                        child.index,
                        child.accumulator,
                        childVolume / Arity);
                    childMaxDensity = maxDensities[child.index];
                }
                assert(!std::isinf(child.densityTimesVolume));
                assert(!std::isnan(child.densityTimesVolume));
                assert(child.densityTimesVolume > 0);
                sum += child.densityTimesVolume;
                nodeMaxDensity = std::max(nodeMaxDensity, childMaxDensity);
            }
            maxDensities[index] = nodeMaxDensity;
            return sum;
        }

        void build(NodeIndex index, int currentDepth, Float childVolume) {
            for (StratumIndex stratum = 0; stratum < Arity; stratum++) {
                // we use a lambda to capture the child because the vector might be re-allocated in the following code
                auto child = [&]() -> typename Node::Child & {
                    return tree.m_nodes[index].children[stratum];
                };

                const Float accumulator = child().accumulator;
                const Float densityTimesVolume = child().densityTimesVolume;
                const Float density = densityTimesVolume / childVolume;
                const bool wasLeafBefore = child().isLeaf();
                const bool isLeafNow = currentDepth >= tree.configuration.minDepth && (
                    (currentDepth >= tree.configuration.maxDepth) ||
                    (tree.configuration.pruning ?
                     wasLeafBefore || (maxDensities[child().index] < Float(2) * density) :
                     densityTimesVolume <= splittingThreshold)
                );

                if (wasLeafBefore && !isLeafNow) {
                    // need to split node
                    const auto newNodeIndex = NodeIndex(tree.m_nodes.size());
                    tree.m_nodes.emplace_back();
                    keepNodes.push_back(true);
                    maxDensities.push_back(accumulator);
                    for (auto &childStratum: tree.m_nodes[newNodeIndex].children) {
                        // initialize children weight
                        childStratum.accumulator = accumulator / Arity;
                        childStratum.densityTimesVolume = densityTimesVolume / Arity;
                    }
                    child().index = newNodeIndex;
                }

                if (!wasLeafBefore && isLeafNow) {
                    // need to collapse node
                    keepNodes[child().index] = false;
                    child().index = 0;
                }

                if (!isLeafNow) {
                    assert(child().index > 0);
                    build(
                        child().index,
                        currentDepth + 1,
                        childVolume / Arity);
                }

                // we are done with densityTimesVolume, use density again
                child().density = density / rootWeight;
                child().accumulator *= tree.configuration.decay;

                assert(!std::isinf(child().density));
                assert(!std::isnan(child().density));
                assert(child().density >= 0);
            }
        }

        std::vector<NodeIndex> buildIndexRemapping() {
            std::vector<NodeIndex> result;
            result.reserve(keepNodes.size());

            NodeIndex currentIndex = 0;
            for (auto keep: keepNodes) {
                result.push_back(currentIndex);
                if (keep)
                    currentIndex++;
            }
            return result;
        }

        /**
         * Removes children that have been marked as collapsed from the data structure.
         * Not to be confused with the similarly badly named @c Configuration::pruning , which
         * collapses nodes when they have little variation among their children.
         * Note that even if pruning is disabled, nodes can still be collapsed if their weight
         * does not exceed the @c Configuration::threshold .
         */
        void pruneTree() {
            auto remapping = buildIndexRemapping();

            auto newNode = tree.m_nodes.begin();
            for (NodeIndex oldNodeIndex = 0; oldNodeIndex < NodeIndex(keepNodes.size()); oldNodeIndex++) {
                if (!keepNodes[oldNodeIndex])
                    // node was marked for deletion
                    continue;

                *newNode = tree.m_nodes[oldNodeIndex];
                for (auto &stratum: newNode->children) {
                    // remap child indices
                    stratum.index = remapping[stratum.index];
                }

                newNode++;
            }

            tree.m_nodes.erase(newNode, tree.m_nodes.end());
        }
    };

    std::vector<Node> m_nodes;

    /**
     * Looks up a given position in [0,1)^n in the octree and returns the node index and stratum of the containing node,
     * while also renormalizing the position to span the domain of the containing child node.
     */
    void lookup(Point &pos, NodeIndex &nodeIndex, StratumIndex &stratumIndex) const {
        NodeIndex candidate = 0;
        do {
            auto &node = m_nodes[candidate];
            nodeIndex = candidate;
            stratumIndex = Node::lookup(pos);
            candidate = node.children[stratumIndex].index;
        } while (candidate);
    }

protected:
    static constexpr uint16_t TypeSizes[] = {
        sizeof(Configuration),
        sizeof(Node),
    };

    Builder builder;
};

};