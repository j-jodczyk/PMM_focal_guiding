/** File containing environments for octree */
#include <mitsuba/render/renderproc.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/statistics.h>


// based on focal guiding
struct EnvMitsuba3D {
    using AABB = mitsuba::AABB;
    using Float = mitsuba::Float;
    using Vector = mitsuba::Vector;
    using Point = mitsuba::Point;

    static constexpr int Dimensionality = 3;

    /**
     * Tiny wrapper to interface Mitsuba's PRNG with our focal guiding code-base.
     */
    struct PRNG {
        PRNG() : rRec(nullptr) {}
        // RadienceQueryRecord is a data structure used by SamplingIntegrator.
        PRNG(mitsuba::RadianceQueryRecord &rRec) : rRec(&rRec) {}

        mitsuba::Float operator()() {
            return rRec->nextSample1D(); // generate 1D (floating-point number) sample
        }

    private:
        mitsuba::RadianceQueryRecord *rRec;
    };

    /// Return the value of the largest component of a vector.
    static Float max(const Vector &a) { return a.x > a.y ? (a.x > a.z ? a.x : a.z) : (a.y > a.z ? a.y : a.z); }

    /// Return the value of the smallest component of a vector.
    static Float min(const Vector &a) { return a.x < a.y ? (a.x < a.z ? a.x : a.z) : (a.y < a.z ? a.y : a.z); }

    /// Return the index of the smallest component of a vector.
    static int argmin(const Vector &a) { return a.x < a.y ? (a.x < a.z ? 0 : 2) : (a.y < a.z ? 1 : 2); }

    /// Divide a vector component-wise.
    static Vector divide(const Vector &a, const Vector &b) {
        Vector c;
        for (int dim = 0; dim < Dimensionality; dim++)
            c[dim] = a[dim] / b[dim];
        return c;
    }

    /// Return the volume of an axis-aligned bounding box.
    static Float volume(const AABB &aabb) { return aabb.getVolume(); }

    /**
     * Computes the integral \int_{t_0}^{t_1} t^2 dt, which is needed to compute the directional PDF resulting
     * from a piece-wise constant spatial density. For more details, please refer to our paper.
     * This needs to be specified as our guiding library can also run in other dimensions (most notably 2-D).
     * EQUATION (7)
     */
    static Float segment(Float tNear, Float tFar) {
        assert(tNear <= tFar);
        return (tFar * tFar * tFar - tNear * tNear * tNear) * (Float(1) / Float(3));
    }
};
