#include <mitsuba/render/scene.h>
#include <mitsuba/core/statistics.h>
#include <mitsuba/core/logger.h>

#include "gaussian_mixture_model.h"
#include "gaussian_component.h"
#include "octree.h"

#include "./envs/mitsuba_env.h"

// todo:
// after each ray is traced go though the tree and collect samples (probably like the middle of the leaf)
// use those samples to feed GMM
// make a better project structure


MTS_NAMESPACE_BEGIN

static StatsCounter avgPathLength("Path tracer", "Average path length", EAverage);

/* Based on MIPathTracer */
class PMMFocalGuidingIntegrator : public MonteCarloIntegrator {
    using Tree = pmm_focal::Octree<EnvMitsuba3D>;

    mutable Tree m_octree;
    // todo
    // probably not double
    using Scalar = double;
    using Vectord = Eigen::Matrix<Scalar, 3, 1>;
    // what about number of components -- is there a way to intelligently modify
    // dims -- probably should be in template... -- visualizer???
    mutable pmm_focal::GaussianMixtureModel<3, 4, Scalar, pmm_focal::GaussianComponent, EnvMitsuba3D> m_gmm;

public:
    PMMFocalGuidingIntegrator(const Properties &props)
        : MonteCarloIntegrator(props) {
            m_octree.configuration.threshold = props.getFloat("orth.threshold", 1e-3);
            m_octree.configuration.minDepth = props.getInteger("orth.minDepth", 0);
            m_octree.configuration.maxDepth = props.getInteger("orth.maxDepth", 14);
            m_octree.configuration.decay = props.getFloat("orth.decay", 0.5f);
        }

    PMMFocalGuidingIntegrator(Stream *stream, InstanceManager *manager)
        : MonteCarloIntegrator(stream, manager) { }

    bool render(Scene *scene, RenderQueue *queue, const RenderJob *job, int sceneResID, int sensorResID, int samplerResID) {
        ref<Scheduler> sched = Scheduler::getInstance();
        ref<Sensor> sensor = static_cast<Sensor *>(sched->getResource(sensorResID));
        ref<Film> film = sensor->getFilm();

        size_t nCores = sched->getCoreCount();
        const Sampler *sampler = static_cast<const Sampler *>(sched->getResource(samplerResID, 0));
        size_t sampleCount = sampler->getSampleCount();

        m_octree.setAABB(scene->getAABB());
        m_gmm.initialize(scene->getAABB());
        Log(EInfo, m_gmm.toString().c_str());
        Log(EInfo, m_octree.toString().c_str());
        Log(EDebug, m_octree.toStringVerbose().c_str());

        Log(EInfo, "Starting render job (%ix%i, " SIZE_T_FMT " %s, " SIZE_T_FMT
            " %s, " SSE_STR ") ..", film->getCropSize().x, film->getCropSize().y,
            sampleCount, sampleCount == 1 ? "sample" : "samples", nCores,
            nCores == 1 ? "core" : "cores");

        /* This is a sampling-based integrator - parallelize */
        ref<ParallelProcess> proc = new BlockedRenderProcess(job,
            queue, scene->getBlockSize());
        int integratorResID = sched->registerResource(this);
        proc->bindResource("integrator", integratorResID);
        proc->bindResource("scene", sceneResID);
        proc->bindResource("sensor", sensorResID);
        proc->bindResource("sampler", samplerResID);
        scene->bindUsedResources(proc);
        bindUsedResources(proc);
        sched->schedule(proc);

        m_process = proc;
        sched->wait(proc);
        m_process = NULL;
        sched->unregisterResource(integratorResID);

        return proc->getReturnStatus() == ParallelProcess::ESuccess;
    }


    Spectrum sampleFromGMM(
        const BSDF* bsdf,
        BSDFSamplingRecord& bRec,
        Float& woPdf,
        Float& bsdfPdf,
        Float& gmmPdf,
        Float bsdfSamplingFraction,
        RadianceQueryRecord rRec,
        bool& isGuidedSample
    ) const {
        mitsuba::Point2 sample = rRec.nextSample2D(); // this is direction (azimutal and polar coords)
        // not sure why we need this.

        auto type = bsdf->getType();

        // EDelta means discrete number of directions - unsuitable for guiding or importance sampling based on density distributions that work over ranges of directions
        if ((type & BSDF::EDelta) == (type & BSDF::EAll)) { // || m_iteration == 0) {
            auto result = bsdf->sample(bRec, bsdfPdf, sample);
            woPdf = bsdfPdf;
            gmmPdf = 0;
            return result;
        }
        Vectord gmmSample;
        Spectrum result;
        if (sample.x < bsdfSamplingFraction) { // bsdfSamplingFraction is from MIS
            // sample BSDF
            sample.x /= bsdfSamplingFraction;
            result = bsdf->sample(bRec, bsdfPdf, sample);
            if (result.isZero()) {
                woPdf = bsdfPdf = gmmPdf = 0;
                return Spectrum{0.0f};
            }

            // If we sampled a delta component, then we have a 0 probability
            // of sampling that direction via guiding, thus we can return early.
            if (bRec.sampledType & BSDF::EDelta) {
                gmmPdf = 0;
                woPdf = bsdfPdf * bsdfSamplingFraction;
                return result / bsdfSamplingFraction;
            }

            result *= bsdfPdf;

        } else {
            // sample guiding distribution
            sample.x = (sample.x - bsdfSamplingFraction) / (1 - bsdfSamplingFraction);

            gmmSample = m_gmm.sample();
            // mitsuba::Point endPoint(gmmSample[0], gmmSample[1], gmmSample[2]);
            // // wo is outgoing direction
            // bRec.wo = normalize(endPoint - bRec.its.p);
            // bRec.wo = bRec.its.toLocal(bRec.wo);

            // // idk - they say it's hack
            // bRec.eta = 1; // eta is Relative index of refraction in the sampled direction. Refractive index determines how much the path of light is bent, or refracted, when entering a material.
            // bRec.sampledType = BSDF::EDiffuse;

            // result = bsdf->eval(bRec);

            // if (result.isZero()) {
            //     // invalid (aka zero contribution) direction
            //     return result;
            // }

            isGuidedSample = true;
        }

        pdfCombined(woPdf, bsdfPdf, gmmPdf, bsdfSamplingFraction, bsdf, bRec, gmmSample);
        if (woPdf == 0) {
            return Spectrum{0.0f};
        }

        return result / woPdf;
    }

    void pdfCombined(
        Float& woPdf, Float& bsdfPdf, Float& gmmPdf, Float bsdfSamplingFraction,
        const BSDF* bsdf, const BSDFSamplingRecord& bRec, Vectord& gmmSample
    ) const {
        gmmPdf = 0;

        auto type = bsdf->getType();
        if ((type & BSDF::EDelta) == (type & BSDF::EAll)) {// || m_iteration == 0) {
            woPdf = bsdfPdf = bsdf->pdf(bRec);
            return;
        }

        bsdfPdf = bsdf->pdf(bRec);
        assert(std::isfinite(bsdfPdf));

        gmmPdf = m_gmm.pdf(gmmSample); // idk - the original with tree did a splat from itersection point towards wo
        assert(std::isfinite(gmmPdf));

        // Multiple Importance Sampling - "While our guiding density excels at sampling focal effects, its performance on other light transport can be poor.
        // It is therefore advisable to combine it with other sampling strategies using multiple importance sampling"
        woPdf = bsdfSamplingFraction * bsdfPdf + (1 - bsdfSamplingFraction) * gmmPdf;
    }


    Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const {
        /* Some aliases and local variables */
        const Scene *scene = rRec.scene;
        Intersection &its = rRec.its;
        RayDifferential ray(r);
        Spectrum Li(0.0f);
        bool scattered = false;

        Float bsdfSamplingFraction = 0.5f; // MIS

        /* Perform the first ray intersection (or ignore if the
           intersection has already been provided). */
        rRec.rayIntersect(ray);
        ray.mint = Epsilon;

        Spectrum throughput(1.0f);
        Float eta = 1.0f;

        while (rRec.depth <= m_maxDepth || m_maxDepth < 0) {
            if (!its.isValid()) {
                /* If no intersection could be found, potentially return
                   radiance from a environment luminaire if it exists */
                if ((rRec.type & RadianceQueryRecord::EEmittedRadiance)
                    && (!m_hideEmitters || scattered))
                    Li += throughput * scene->evalEnvironment(ray);
                break;
            }

            const BSDF *bsdf = its.getBSDF(ray);

            /* Possibly include emitted radiance if requested */
            if (its.isEmitter() && (rRec.type & RadianceQueryRecord::EEmittedRadiance)
                && (!m_hideEmitters || scattered))
                Li += throughput * its.Le(-ray.d);

            /* Include radiance from a subsurface scattering model if requested */
            if (its.hasSubsurface() && (rRec.type & RadianceQueryRecord::ESubsurfaceRadiance))
                Li += throughput * its.LoSub(scene, rRec.sampler, -ray.d, rRec.depth);

            if ((rRec.depth >= m_maxDepth && m_maxDepth > 0)
                || (m_strictNormals && dot(ray.d, its.geoFrame.n)
                    * Frame::cosTheta(its.wi) >= 0)) {

                /* Only continue if:
                   1. The current path length is below the specifed maximum
                   2. If 'strictNormals'=true, when the geometric and shading
                      normals classify the incident direction to the same side */
                break;
            }

            /* ==================================================================== */
            /*                     Direct illumination sampling                     */
            /* ==================================================================== */

            /* Estimate the direct illumination if this is requested */
            DirectSamplingRecord dRec(its);

            // this is nee
            // if (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance &&
            //     (bsdf->getType() & BSDF::ESmooth)) {
            //     Spectrum value = scene->sampleEmitterDirect(dRec, rRec.nextSample2D());
            //     if (!value.isZero()) {
            //         const Emitter *emitter = static_cast<const Emitter *>(dRec.object);

            //         /* Allocate a record for querying the BSDF */
            //         BSDFSamplingRecord bRec(its, its.toLocal(dRec.d), ERadiance);

            //         /* Evaluate BSDF * cos(theta) */
            //         const Spectrum bsdfVal = bsdf->eval(bRec);

            //         /* Prevent light leaks due to the use of shading normals */
            //         if (!bsdfVal.isZero() && (!m_strictNormals
            //                 || dot(its.geoFrame.n, dRec.d) * Frame::cosTheta(bRec.wo) > 0)) {

            //             /* Calculate prob. of having generated that direction
            //                using BSDF sampling */
            //             Float woPdf = 0, bsdfPdf = 0, gmmPdf = 0;
            //             if (emitter->isOnSurface() && dRec.measure == ESolidAngle) {
            //                 pdfCombined(woPdf, bsdfPdf, gmmPdf, bsdfSamplingFraction, bsdf, bRec);
            //             }

            //             /* Weight using the power heuristic */
            //             Float misWeight = miWeight(dRec.pdf, woPdf);
            //             // LrEstimate += bsdfVal * value * misWeight;
            //             Li += throughput * value * bsdfVal * misWeight;
            //         }
            //     }
            // }

            /* ==================================================================== */
            /*                            BSDF sampling                             */
            /* ==================================================================== */

            /* Sample BSDF * cos(theta) */
            Float woPdf, bsdfPdf, gmmPdf;

            do {
                BSDFSamplingRecord bRec(its, rRec.sampler, ERadiance);
                bool isGuidedSample = false;
                Spectrum bsdfWeight = sampleFromGMM(bsdf, bRec, woPdf, bsdfPdf, gmmPdf, bsdfSamplingFraction, rRec, isGuidedSample);
                if (bsdfWeight.isZero())
                    break;

                const Vector wo = its.toWorld(bRec.wo);
                Float woDotGeoN = dot(its.geoFrame.n, wo);
                if (m_strictNormals && woDotGeoN * Frame::cosTheta(bRec.wo) <= 0)
                    break;

                bool hitEmitter = false;
                Spectrum value;

                /* Trace a ray in this direction */
                ray = Ray(its.p, wo, ray.time);

                if (scene->rayIntersect(ray, its)) {
                    /* Intersected something - check if it was a luminaire */
                    if (its.isEmitter()) {
                        value = its.Le(-ray.d);
                        dRec.setQuery(ray, its);
                        hitEmitter = true;
                    }
                } else {
                    /* Intersected nothing -- perhaps there is an environment map? */
                    const Emitter *env = scene->getEnvironmentEmitter();

                    if (env) {
                        value = env->evalEnvironment(ray);
                        if (!env->fillDirectSamplingRecord(dRec, ray))
                            break;
                        hitEmitter = true;
                    } else {
                        break;
                    }
                }

                /* If a luminaire was hit, estimate the local illumination and
                    weight using the power heuristic */
                if (hitEmitter &&
                    (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance)) {
                    /* Compute the prob. of generating that direction using the
                        implemented direct illumination sampling technique */
                    Float lumPdf = (!(bRec.sampledType & BSDF::EDelta)) ? scene->pdfEmitterDirect(dRec) : 0;
                    Float misWeight = miWeight(woPdf, lumPdf);
                    Li += bsdfWeight * value * misWeight;
                }

                /* ==================================================================== */
                /*                         Indirect illumination                        */
                /* ==================================================================== */

                /* Set the recursive query type. Stop if no surface was hit by the
                    BSDF sample or if indirect illumination was not requested */
                if (!its.isValid() || !(rRec.type & RadianceQueryRecord::EIndirectSurfaceRadiance))
                    break;
                rRec.type = RadianceQueryRecord::ERadianceNoEmission & ~RadianceQueryRecord::EIntersection;
                rRec.depth++;

                Spectrum outputNested = this->Li(ray, rRec);
                Li += bsdfWeight * outputNested;

                // splat
                const BSDF *endpointBSDF = its.getBSDF();
                Float endpointRoughness = std::numeric_limits<Float>::infinity();
                if (endpointBSDF) {
                    for (int comp = 0; comp < endpointBSDF->getComponentCount(); ++comp) {
                        endpointRoughness = std::min(endpointRoughness, endpointBSDF->getRoughness(its, comp));
                    }
                }
                const bool endpointIsGlossy = endpointRoughness < 0.3f; // [Ruppert et al. 2020]
                const Float splatDistance = endpointIsGlossy ?
                    std::numeric_limits<Float>::infinity() : /// virtual image possible, need to splat entire ray
                    its.t;

                std::vector<AABB> leafAABBs;
                m_octree.splat(ray.o, ray.d, splatDistance, leafAABBs);
                Log(EDebug, "Collected %d samples", leafAABBs.size());

                std::vector<Eigen::Matrix<Scalar, 3, 1>> samples;
                for (size_t i=0; i < leafAABBs.size(); i++) {
                    auto leaf = leafAABBs[i];
                    Log(EDebug, leaf.toString().c_str());
                    Eigen::Matrix<Scalar, 3, 1> center;
                    center << (leaf.min[0] + leaf.max[0]) / 2, (leaf.min[1] + leaf.max[1]) / 2, (leaf.min[2] + leaf.max[2]) / 2;
                    samples.push_back(center);
                }
                m_gmm.fit(samples);
                Log(EDebug, "Fitted GMM");
                Log(EDebug, m_gmm.toString().c_str());

            } while(false);
        }

        /* Store statistics */
        avgPathLength.incrementBase();
        avgPathLength += rRec.depth;

        // Log(EInfo, m_gmm.toString().c_str());

        return Li;
    }

    inline Float miWeight(Float pdfA, Float pdfB) const {
        pdfA *= pdfA;
        pdfB *= pdfB;
        return pdfA / (pdfA + pdfB);
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        MonteCarloIntegrator::serialize(stream, manager);
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "PMMFocalGuidingIntegrator" << endl
            << "  maxDepth = " << m_maxDepth << "," << endl
            << "  rrDepth = " << m_rrDepth << "," << endl
            << "  strictNormals = " << m_strictNormals << endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
};

MTS_IMPLEMENT_CLASS_S(PMMFocalGuidingIntegrator, false, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(PMMFocalGuidingIntegrator, "PMM focal guiding path tracer")
MTS_NAMESPACE_END
