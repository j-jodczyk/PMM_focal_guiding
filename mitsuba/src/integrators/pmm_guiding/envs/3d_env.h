#include "../visualizer/AABB.hpp"
#include "../visualizer/math.hpp"

// for now only as much as GMM tests require
class Env3D {
public:
    static constexpr int Dimensionality = 3;

    using Float = ::Float;
    using Vector = ::Vector<Float, Dimensionality>;
    using AABB = ::AABB<Vector>;
};
