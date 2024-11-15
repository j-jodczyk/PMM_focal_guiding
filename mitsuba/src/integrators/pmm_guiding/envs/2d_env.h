#include "../visualizer/AABB.hpp"
#include "../visualizer/math.hpp"
#include "../visualizer/PRNG.hpp"

class Env2D {
public:
    static constexpr int Dimensionality = 2;

    using Float = ::Float;
    using Vector = ::Vector<Float, Dimensionality>;
    using Point = ::Vector<Float, Dimensionality>;
    using AABB = ::AABB<Vector>;
    using PRNG = ::PRNG;

    static Vector divide(const Vector &a, const Vector &b) { return a / b; }

    static Float min(const Vector &v) { return v.min(); }

    static Float max(const Vector &v) { return v.max(); }

    static int argmin(const Vector &v) { return v.argmin(); }

    static int argmax(const Vector &v) { return v.argmax(); }

    static void extend(AABB &aabb, const Point &point) { aabb.extend(point); }

    static void atomicAdd(Float &dest, Float delta) {
        dest += delta;
    }

    static Vector normalize(const Vector &vec) {
        return vec.normalized();
    }

    static Float volume(const AABB &aabb) {
        return aabb.volume();
    }

    static Float segment(Float tNear, Float tFar) {
        return Float(2.0 * M_PI * M_PI) * (tFar * tFar - tNear * tNear);
    }

    static Point absolute(const AABB &aabb, const Point &relative) {
        return aabb.absolute(relative);
    }

    static Point relative(const AABB &aabb, const Point &absolute) {
        return aabb.relative(absolute);
    }
};
