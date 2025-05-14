/*
    This file is part of Mitsuba, a physically based rendering system.

    Copyright (c) 2007-2014 by Wenzel Jakob and others.

    Mitsuba is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Mitsuba is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

/// we support outputting several AOVs that can be helpful for research and debugging.
/// since they are computationally expensive, we disable them by default.
/// uncomment the following line to enable outputting AOVs:
//#define FOCAL_INCLUDE_AOVS

#include <mitsuba/render/renderproc.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/statistics.h>

#include <array>
#include <atomic>
#include <chrono>
#include <fstream>
#include <functional>
#include <iomanip>
#include <sstream>
#include <mutex>
#include <Eigen/Dense>
#include "iterable_blocked_vector.h"
#include "gaussian_mixture_model.h"
#include "octree.h"
#include "weighted_sample.h"
#include "stats_recursive.h"

namespace pmm_focal {
    thread_local StatsRecursiveImageBlockCache *StatsRecursiveImageBlockCache::instance = nullptr;
    thread_local StatsRecursiveDescriptorCache *StatsRecursiveDescriptorCache::instance = nullptr;
    thread_local StatsRecursiveValuesCache *StatsRecursiveValuesCache::instance = nullptr;
}

MTS_NAMESPACE_BEGIN


/**
 * Based on the recursive path tracer from EARS [Rath et al. 2022].
 */
class MIFocalGuidingPathTracer : public MonteCarloIntegrator {
private:
    struct LiInput {
        int pixelX, pixelY;
        Spectrum weight;
        Spectrum absoluteWeight; /// only relevant for AOVs
        RayDifferential ray;
        RadianceQueryRecord rRec;
        bool scattered { false };
        Float eta { 1.f };
    };

    struct LiOutput {
        Spectrum reflected { 0.f };
        Spectrum emitted { 0.f };

        int numSamples { 0 };
        Float depthAcc { 0.f };
        Float depthWeight { 0.f };

        void markAsLeaf(int depth) {
            depthAcc = depth;
            depthWeight = 1;
        }

        Float averagePathLength() const {
            return depthWeight > 0 ? depthAcc / depthWeight : 0;
        }

        Float numberOfPaths() const {
            return depthWeight;
        }

        Spectrum totalContribution() const {
            return reflected + emitted;
        }
    };

    using Distribution = pmm_focal::Octree<EnvMitsuba3D>;
    using Scalar = float;
    using Vectord = Eigen::Matrix<Scalar, 3, 1>;
    mutable pmm_focal::GaussianMixtureModel<Scalar, EnvMitsuba3D> m_gmm;
    mutable std::vector<IterableBlockedVector<pmm_focal::WeightedSample>> perThreadSamples;
    mutable PrimitiveThreadLocal<bool> threadLocalInitialized;
    mutable std::atomic<size_t> maxThreadId;

public:
    MIFocalGuidingPathTracer(const Properties &props)
    : MonteCarloIntegrator(props) {
        m_converging.configuration.threshold = props.getFloat("orth.threshold", 1e-3);
        m_converging.configuration.minDepth = props.getInteger("orth.minDepth", 0);
        m_converging.configuration.maxDepth = props.getInteger("orth.maxDepth", 14);
        m_converging.configuration.decay = props.getFloat("orth.decay", 0.5f);
        m_diverging.configuration = m_converging.configuration;

        m_budget = props.getFloat("budget", 120.0f);
        m_iterationBudget = props.getFloat("iterationBudget", 6.0f);
        m_iterationCount = props.getInteger("iterationCount", 15);

        m_gmm.setAlpha(props.getFloat("gmm.alpha", 0.25));
        m_gmm.setSplittingThreshold(props.getFloat("gmm.splittingThreshold", 7.0));
        m_gmm.setMergingThreshold(props.getFloat("gmm.mergingThreshold", 0.25));
        m_gmm.setMinNumComp(props.getInteger("gmm.minNumComp", 10));
        m_gmm.setMaxNumComp(props.getInteger("gmm.maxNumComp", 15));
        m_gmm.setInitMethod(props.getString("gmm.initMethod", "Random"));

        // print all properties to logs (and hence EXRs), useful for making sense of old renders.
        for (const auto &name : props.getPropertyNames()) {
            Log(EInfo, "%s: %s", name.c_str(), props.getAsString(name).c_str());
        }
    }

    ref<BlockedRenderProcess> renderPass(Scene *scene,
        RenderQueue *queue, const RenderJob *job,
        int sceneResID, int sensorResID, int samplerResID, int integratorResID) {

        /* This is a sampling-based integrator - parallelize */
        ref<BlockedRenderProcess> proc = new BlockedRenderProcess(job,
            queue, scene->getBlockSize());

        // proc->disableProgress();

        proc->bindResource("integrator", integratorResID);
        proc->bindResource("scene", sceneResID);
        proc->bindResource("sensor", sensorResID);
        proc->bindResource("sampler", samplerResID);

        scene->bindUsedResources(proc);
        bindUsedResources(proc);

        return proc;
    }

    bool renderIterationTime(Float until, int &passesRenderedLocal, Scene *scene, RenderQueue *queue, const RenderJob *job,
        int sceneResID, int sensorResID, int samplerResID, int integratorResID) {

        ref<Scheduler> sched = Scheduler::getInstance();
        ref<Sensor> sensor = static_cast<Sensor *>(sched->getResource(sensorResID));
        ref<Film> film = sensor->getFilm();

        Log(EInfo, "ITERATION %d, until %.1f seconds", m_iteration, until);

        passesRenderedLocal = 0;

        bool result = true;
        while (true) {
            ref<BlockedRenderProcess> process = renderPass(scene, queue, job, sceneResID, sensorResID, samplerResID, integratorResID);
            sched->schedule(process);
            sched->wait(process);

            ++passesRenderedLocal;
            ++m_passesRenderedGlobal;

            const Float progress = computeElapsedSeconds(m_startTime);
            m_progress->update(progress);
            if (progress > until) {
                break;
            }

            if (process->getReturnStatus() != ParallelProcess::ESuccess) {
                result = false;
                break;
            }
        }

        Log(EInfo, "  %.2f seconds elapsed, passes this iteration: %d, total passes: %d",
            computeElapsedSeconds(m_startTime), passesRenderedLocal, m_passesRenderedGlobal);

        return result;
    }

    static Float computeElapsedSeconds(std::chrono::steady_clock::time_point start) {
        auto current = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(current - start);
        return (Float)ms.count() / 1000;
    }

    bool renderTime(Scene *scene, RenderQueue *queue, const RenderJob *job,
        int sceneResID, int sensorResID, int samplerResID, int integratorResID) {

        ref<Scheduler> sched = Scheduler::getInstance();
        ref<Sensor> sensor = static_cast<Sensor *>(sched->getResource(sensorResID));
        ref<Film> film = sensor->getFilm();

        m_progress = std::unique_ptr<ProgressReporter>(new ProgressReporter("Rendering", (int)m_budget, job));

        /// we begin with iterative narrowing disabled.
        m_converging.configuration.splattingStrategy = Distribution::SPLAT_RAY;
        m_diverging.configuration = m_converging.configuration;

        /// we start with iterative narrowing once ~66% of the the training iterations have finished.
        const int startWeightingIteration = m_iterationCount * 2 / 3;

        int spp;
        Float until = 0;
        for (m_iteration = 0;; m_iteration++) {
            if (m_iteration == startWeightingIteration) {
                SLog(EInfo, "starting iterative narrowing");
                m_converging.configuration.splattingStrategy = Distribution::SPLAT_RAY_WEIGHTED;

                /// we perform stronger exponential decay of our tree to forget spurious focal points more quickly.
                m_converging.configuration.decay = 0.25f;
                m_diverging.configuration = m_converging.configuration;
            }

            const Float timeBeforeIter = computeElapsedSeconds(m_startTime);
            if (timeBeforeIter >= m_budget) {
                /// note that we always do at least one sample per pixel per training iteration,
                /// which can sometimes be significantly longer than the budget for that iteration.
                /// this means we can exhaust the training budget before all iterations have finished
                break;
            }

            until += m_iterationBudget;
            if (m_iteration == m_iterationCount) {
                SLog(EInfo, "final iteration");

                /// before building the spatial density for the last time, we enable merging of nodes with little variation.
                m_converging.configuration.pruning = true;
                /// we also disable the minimum tree depth, which is only useful during training, and thereby improve performance.
                m_converging.configuration.minDepth = 0;

                m_diverging.configuration = m_converging.configuration;

                until = m_budget;
            }

            if (m_iteration > 0) {
                const size_t numValidSamples = std::accumulate(perThreadSamples.begin(), perThreadSamples.end(), 0UL,
                    [](size_t sum, const IterableBlockedVector<pmm_focal::WeightedSample>& samples) -> size_t { return sum+samples.size(); });

                if (numValidSamples < 128) {
                    Log(EInfo, "skipping fit due to insufficient sample data (got %zu/%d valid samples).", numValidSamples, 128);
                    return true;
                }

                std::vector<pmm_focal::WeightedSample> iterationSamples;
                // reserve to prevent bad alloc
                iterationSamples.reserve(numValidSamples);
                for (auto& vec : perThreadSamples) {
                    iterationSamples.insert(iterationSamples.end(), vec.begin(), vec.end());
                    vec.clear();
                }

                Log(EInfo, "Got %i non-zero samples", iterationSamples.size());
                m_gmm.processMegaBatch(iterationSamples);
                Log(EInfo, m_gmm.toString().c_str());
                /// update the guiding densities
                // diverging meaning behind the surface we are currently sampling from.
                const Float convThreshold = m_converging.sumDensities();
                const Float divThreshold = m_diverging.sumDensities();
                const Float threshold = convThreshold + divThreshold;

                /// the probabilty for diverging is set to the share of weight the diverging field has in the total weight.
                m_divergeProbability = divThreshold == 0 ? 0 : divThreshold / (divThreshold + convThreshold);

                printf("building converging\n  ");
                m_converging.build(threshold);
                printf("building diverging\n  ");
                m_diverging.build(threshold);

                printf("diverge probability: %.3f\n", m_divergeProbability);
                printf("\n");
            }

            film->clear();

            if (!renderIterationTime(until, spp, scene, queue, job, sceneResID, sensorResID, samplerResID, integratorResID)) {
                return false;
            }

            m_finalImage.clear();
            m_finalImage.add(film, spp, 1);
        }

        return true;
    }

    bool render(Scene *scene, RenderQueue *queue, const RenderJob *job,
        int sceneResID, int sensorResID, int samplerResID) {

        ref<Scheduler> sched = Scheduler::getInstance();

        size_t nCores = sched->getCoreCount();
        perThreadSamples.resize(nCores);
        maxThreadId.store(0);
        ref<Sensor> sensor = static_cast<Sensor *>(sched->getResource(sensorResID));
        ref<Film> film = sensor->getFilm();

        m_converging.setAABB(scene->getAABB());
        m_converging.clear();
        m_diverging.setAABB(scene->getAABB());
        m_diverging.clear();
        m_gmm.setAABB(scene->getAABB());
        m_divergeProbability = 0.5f;

        m_startTime = std::chrono::steady_clock::now();

        Log(EInfo, "Starting render job (%ix%i, " SIZE_T_FMT " %s, " SSE_STR ") ..", film->getCropSize().x, film->getCropSize().y, nCores, nCores == 1 ? "core" : "cores");

        Thread::initializeOpenMP(nCores);

        int integratorResID = sched->registerResource(this);
        bool result = true;

        m_passesRenderedGlobal = 0;
        m_finalImage.clear();

        result = renderTime(scene, queue, job, sceneResID, sensorResID, samplerResID, integratorResID);

        Vector2i size = film->getSize();
        ref<Bitmap> image = new Bitmap(Bitmap::EPixelFormat::ESpectrum, Bitmap::EComponentFormat::EFloat32, size);
        film->develop(Point2i(0, 0), size, Point2i(0, 0), image);

        ref<Bitmap> finalBitmap = new Bitmap(Bitmap::EPixelFormat::ESpectrum, Bitmap::EComponentFormat::EFloat32, film->getSize());
        m_finalImage.develop(finalBitmap.get());
        film->setBitmap(finalBitmap);

        sched->unregisterResource(integratorResID);
        m_progress = nullptr;

        return result;
    }

    void renderBlock(const Scene *scene, const Sensor *sensor, // sensor = camera
        Sampler *sampler, ImageBlock *block, const bool &stop,
        const std::vector< TPoint2<uint8_t> > &points) const {

        bool needsApertureSample = sensor->needsApertureSample();
        bool needsTimeSample = sensor->needsTimeSample();

        RadianceQueryRecord rRec(scene, sampler);
        Point2 apertureSample(0.5f);
        Float timeSample = 0.5f;
        RayDifferential sensorRay;

        block->clear();

        pmm_focal::StatsRecursiveValues stats;

        uint32_t queryType = RadianceQueryRecord::ESensorRay;

        if (!sensor->getFilm()->hasAlpha()) // Don't compute an alpha channel if we don't have to
            queryType &= ~RadianceQueryRecord::EOpacity;

        // CORE RENDERING LOOP
        for (size_t i = 0; i < points.size(); ++i) {
            Point2i offset = Point2i(points[i]) + Vector2i(block->getOffset()); // offset of the point in the context of the whole image

            constexpr int sppPerPass = 1;
            for (int j = 0; j < sppPerPass; j++) {
                stats.reset();

                rRec.newQuery(queryType, sensor->getMedium());
                Point2 samplePos(Point2(offset) + Vector2(rRec.nextSample2D()));

                if (needsApertureSample)
                    apertureSample = rRec.nextSample2D();
                if (needsTimeSample)
                    timeSample = rRec.nextSample1D();

                Spectrum spec = sensor->sampleRayDifferential(
                    sensorRay, samplePos, apertureSample, timeSample);

                LiInput input;
                input.pixelX = offset.x;
                input.pixelY = offset.y;
                input.absoluteWeight = spec;
                input.weight = spec;
                input.ray = sensorRay;
                input.rRec = rRec;

                LiOutput output = Li(input, stats); // Light transport simulation
                block->put(samplePos, spec * output.totalContribution(), input.rRec.alpha);
                sampler->advance();

                stats.avgPathLength.add(output.averagePathLength());
                stats.numPaths.add(output.numberOfPaths());
            }
        }

    }

    Spectrum sampleMat(
        const BSDF* bsdf, BSDFSamplingRecord& bRec, Float& woPdf, Float& bsdfPdf, Float& gmmPdf,
        Float bsdfSamplingFraction, RadianceQueryRecord& rRec, bool& isGuidedSample
    ) const {
        isGuidedSample = false;

        Point2 sample = rRec.nextSample2D();

        auto type = bsdf->getType();
        // EDelta = reflection into a discrete set of directions
        // the following check cheks if this is the only type of BSDF
        if ((type & BSDF::EDelta) == (type & BSDF::EAll) || m_iteration == 0) {
            auto result = bsdf->sample(bRec, bsdfPdf, sample);
            woPdf = bsdfPdf;
            gmmPdf = 0;
            return result;
        }

        Spectrum result;
        mitsuba::Vector dir;
        bool isDiverging = false;
        isDiverging = (sample.x < m_divergeProbability);

        if (sample.x < bsdfSamplingFraction) {
            /// sample the BSDF

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
            /// sample the guiding distribution

            sample.x = (sample.x - bsdfSamplingFraction) / (1 - bsdfSamplingFraction);

            /// decide whether to sample the converging or the diverging focal field.
            if (isDiverging) {
                sample.x /= m_divergeProbability;
            } else {
                sample.x = (sample.x - m_divergeProbability) / (1 - m_divergeProbability);
            }

            Eigen::VectorXd gmmSample = m_gmm.sample(rRec);
            mitsuba::Point endPoint(gmmSample[0], gmmSample[1], gmmSample[2]);
            dir = endPoint - bRec.its.p;
            bRec.wo = normalize(dir);
            bRec.wo = bRec.its.toLocal(bRec.wo);
            bRec.eta = 1; /// hack
            bRec.sampledType = BSDF::EDiffuse; /// hack
            if (isDiverging) {
                /// if we sample the diverging focal field, the direction points @b away from the sampled point
                bRec.wo *= -1;
            }
            result = bsdf->eval(bRec);

            if (result.isZero()) {
                /// no need to compute any PDFs, our guiding produced an invalid (zero contribution) direction.
                return result;
            }

            isGuidedSample = true;
        }

        pdfMat(woPdf, bsdfPdf, gmmPdf, bsdfSamplingFraction, bsdf, bRec, isDiverging);
        if (woPdf == 0) {
            return Spectrum{0.0f};
        }

        return result / woPdf;
    }

    /* calculates the probability densities (pdf) for a ray direction based on both the BSDF (Bidirectional Scattering Distribution Function) and a directional guiding tree */
    void pdfMat(
        Float& woPdf, Float& bsdfPdf, Float& gmmPdf, Float bsdfSamplingFraction,
        const BSDF* bsdf, const BSDFSamplingRecord& bRec, bool isDiverging
    ) const {
        gmmPdf = 0;

        auto type = bsdf->getType();
        if ((type & BSDF::EDelta) == (type & BSDF::EAll) || m_iteration == 0 // EDelta = scattering into a discrete set of directions; Eall = any kind of scattering
        //    || m_iteration != m_iterationCount
        ) {
            woPdf = bsdfPdf = bsdf->pdf(bRec);
            return;
        }

        bsdfPdf = bsdf->pdf(bRec);
        assert(std::isfinite(bsdfPdf));

        // HERE PDF SAMPLING FROM TREE
        gmmPdf = (1 - m_divergeProbability) * m_converging.splatPdf(bRec.its.p, bRec.its.toWorld(bRec.wo), m_gmm) + m_divergeProbability * m_diverging.splatPdf(bRec.its.p, -bRec.its.toWorld(bRec.wo), m_gmm);
        assert(std::isfinite(gmmPdf));

        // Multiple Importance Sampling - "While our guiding density excels at sampling focal effects, its performance on other light transport can be poor.
        // It is therefore advisable to combine it with other sampling strategies using multiple importance sampling"
        woPdf = bsdfSamplingFraction * bsdfPdf + (1 - bsdfSamplingFraction) * gmmPdf;
    }

    Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const {
        Assert(false);
        return Spectrum { 0.f };
    }

    LiOutput Li(LiInput &input, pmm_focal::StatsRecursiveValues &stats) const {
        static thread_local IterableBlockedVector<pmm_focal::WeightedSample>* samples {nullptr};

        // init thread variables
        if (!threadLocalInitialized.get()) {
            const size_t threadId = maxThreadId.fetch_add(1, std::memory_order_relaxed);
            // Log(EInfo, "threadId: %d", threadId);
            samples = &perThreadSamples.at(threadId);
            threadLocalInitialized.get() = true;
        }

        LiOutput output;
        Float bsdfSamplingFraction = 0.5f; // MIS

        if (m_maxDepth >= 0 && input.rRec.depth > m_maxDepth) {
            // maximum depth reached
            output.markAsLeaf(input.rRec.depth);
            return output;
        }

        /* Some aliases and local variables */
        RadianceQueryRecord &rRec = input.rRec;
        Intersection &its = rRec.its;
        const Scene *scene = rRec.scene;
        RayDifferential ray(input.ray);

        /* Perform the first ray intersection (or ignore if the
           intersection has already been provided). */
        if (rRec.type & RadianceQueryRecord::EIntersection) {
            rRec.rayIntersect(ray);
        }

        if (!its.isValid()) {
            /* If no intersection could be found, potentially return
                radiance from a environment luminaire if it exists */
            if ((rRec.type & RadianceQueryRecord::EEmittedRadiance)
                && (!m_hideEmitters || input.scattered))
                output.emitted += scene->evalEnvironment(ray); // Return the environment radiance for a ray that did not intersect any of the scene objects.
            stats.emitted.add(rRec.depth-1, input.absoluteWeight * output.emitted, 0);
            output.markAsLeaf(rRec.depth);
            return output;
        }

        const BSDF *bsdf = its.getBSDF();

        /* Possibly include emitted radiance if requested */
        if (its.isEmitter() && (rRec.type & RadianceQueryRecord::EEmittedRadiance)
            && (!m_hideEmitters || input.scattered))
            output.emitted += its.Le(-ray.d);

        /* Include radiance from a subsurface scattering model if requested - whaaat? */
        if (its.hasSubsurface() && (rRec.type & RadianceQueryRecord::ESubsurfaceRadiance))
            output.emitted += its.LoSub(scene, rRec.sampler, -ray.d, rRec.depth);

        stats.emitted.add(rRec.depth-1, input.absoluteWeight * output.emitted, 0);

        const Float wiDotGeoN = -dot(its.geoFrame.n, ray.d);
        const Float wiDotShN = Frame::cosTheta(its.wi);
        if ((rRec.depth >= m_maxDepth && m_maxDepth > 0)
            || (m_strictNormals && wiDotGeoN * wiDotShN < 0)) { // This condition ensures that the ray is interacting with the correct side of the surface.

            /* Only continue if:
                1. The current path length is below the specifed maximum
                2. If 'strictNormals'=true, when the geometric and shading
                    normals classify the incident direction to the same side */
            output.markAsLeaf(rRec.depth);
            return output;
        }

        /* ==================================================================== */
        /*                 Compute reflected radiance estimate                  */
        /* ==================================================================== */

        /// compute splitting factor
        const Float splittingFactor = 1; // No Russian Roulette or Splitting for now

        Spectrum learnedContribution(0.f);

        /// actual number of samples is the stochastic rounding of our splittingFactor
        const int numSamples = int(splittingFactor + rRec.nextSample1D());
        output.numSamples = numSamples;
        for (int sampleIndex = 0; sampleIndex < numSamples; ++sampleIndex) {
            Spectrum LrEstimate(0.f);

            /* ==================================================================== */
            /*                     Direct illumination sampling                     */
            /* ==================================================================== */
            // estimates the light directly reaching the surface from an emitter (like a light source)
            // without considering indirect reflections or refractions.

            DirectSamplingRecord dRec(its);

            /* Estimate the direct illumination if this is requested */
            if ((rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance) &&
                (bsdf->getType() & BSDF::ESmooth)) {

                Spectrum value = scene->sampleEmitterDirect(dRec, rRec.nextSample2D());
                if (!value.isZero()) {
                    const Emitter *emitter = static_cast<const Emitter *>(dRec.object);

                    /* Allocate a record for querying the BSDF */
                    BSDFSamplingRecord bRec(its, its.toLocal(dRec.d), ERadiance);

                    /* Evaluate BSDF * cos(theta) */
                    Spectrum bsdfVal = bsdf->eval(bRec);

                    /* Prevent light leaks due to the use of shading normals */
                    if (!bsdfVal.isZero() && (!m_strictNormals
                        || dot(its.geoFrame.n, dRec.d) * Frame::cosTheta(bRec.wo) > 0)) {

                        /* Calculate prob. of having generated that direction
                            using BSDF sampling */
                        Float woPdf = 0, bsdfPdf = 0, dTreePdf = 0;
                        if (emitter->isOnSurface() && dRec.measure == ESolidAngle) {
                            // first time guiding is used
                            pdfMat(woPdf, bsdfPdf, dTreePdf, bsdfSamplingFraction, bsdf, bRec, false);
                        }

                        /* Weight using the power heuristic */
                        Float misWeight = miWeight(dRec.pdf, woPdf); // multiple importance sampling weight

                        LrEstimate += bsdfVal * value * misWeight; // direct illumination estimate
                        // we do not learn to guide direct light, so no update to learnedContribution here

                        stats.emitted.add(rRec.depth, input.absoluteWeight * bsdfVal * value * misWeight / splittingFactor, 0);
                    }
                }
            }

            /* ==================================================================== */
            /*                            BSDF sampling                             */
            /* ==================================================================== */

            Spectrum bsdfWeight(0.f);
            Float woPdf, bsdfPdf, dTreePdf;
            Spectrum LiEstimate(0.f);
            LiInput inputNested = input;
            bool sampledDiffuse = false;

            do {
                inputNested.weight *= 1.f / splittingFactor;
                inputNested.absoluteWeight *= 1.f / splittingFactor;
                inputNested.rRec.its = rRec.its;

                RadianceQueryRecord &rRec = inputNested.rRec;
                Intersection &its = rRec.its;
                RayDifferential &ray = inputNested.ray;

                /* Sample BSDF * cos(theta) */
                bool isGuidedSample = false;
                BSDFSamplingRecord bRec(its, rRec.sampler, ERadiance);
                // this is the second time guiding is used
                bsdfWeight = sampleMat(bsdf, bRec, woPdf, bsdfPdf, dTreePdf, bsdfSamplingFraction, rRec, isGuidedSample);
                if (bsdfWeight.isZero())
                    break;

                inputNested.scattered |= bRec.sampledType != BSDF::ENull;
                sampledDiffuse = bRec.sampledType & BSDF::ESmooth;

                /* Prevent light leaks due to the use of shading normals */
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
                        if (m_hideEmitters && !inputNested.scattered)
                            break;

                        value = env->evalEnvironment(ray);
                        if (!env->fillDirectSamplingRecord(dRec, ray))
                            break;
                        hitEmitter = true;
                    } else {
                        break;
                    }
                }

                /* Keep track of the throughput, medium, and relative
                refractive index along the path */
                inputNested.weight *= bsdfWeight;
                inputNested.absoluteWeight *= bsdfWeight;
                inputNested.eta *= bRec.eta; // Relative index of refraction in the sampled direction.

                /* If a luminaire was hit, estimate the local illumination and
                    weight using the power heuristic */
                if (hitEmitter &&
                    (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance)) {
                    /* Compute the prob. of generating that direction using the
                        implemented direct illumination sampling technique */
                    Float lumPdf = (!(bRec.sampledType & BSDF::EDelta)) ? scene->pdfEmitterDirect(dRec) : 0;
                    Float misWeight = miWeight(woPdf, lumPdf);
                    LrEstimate += bsdfWeight * value * misWeight;
                    learnedContribution += bsdfWeight * value * misWeight;
                    stats.emitted.add(rRec.depth, inputNested.absoluteWeight * value * misWeight, 0);
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

                LiOutput outputNested = this->Li(inputNested, stats); // recursive
                LrEstimate += bsdfWeight * outputNested.totalContribution();
                learnedContribution += bsdfWeight * outputNested.totalContribution();

                output.depthAcc += outputNested.depthAcc;
                output.depthWeight += outputNested.depthWeight;

                // splat
                float contribution = (input.weight * learnedContribution).average();
                if (sampledDiffuse && contribution > 0 && m_iteration != m_iterationCount) {
                    // Log(EInfo, "contribution greater than 0: %f", contribution);
                    const auto &ray = inputNested.ray;

                    /// Determine if focal point could lie outside the [0,its.t] interval (t is distance traveled along the ray)
                    /// Since this is only possible for virtual images, we check whether the endpoint of the path segment
                    /// is glossy -- if it is not, it cannot produce a virtual image
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
                        its.t; /// virtual image is not possible since the endpoint is diffuse, splatting segment is sufficient

                    /// To arrive at the correct weights for iterative narrowing, we need to multiply the PDF contribution
                    /// from each region by (one minus) the diverge probability. Equivalently, we can also divide @b woPdf by
                    /// the same quantity. This way the converging field does not try to learn focal points that are better
                    /// handled by the diverging focal field and vice versa.
                    m_converging.splat(ray.o, ray.d, splatDistance, contribution, samples, woPdf / (1 - m_divergeProbability));
                    if (endpointIsGlossy) {
                        m_diverging.splat(ray.o, -ray.d, splatDistance, contribution, samples, woPdf / m_divergeProbability);
                    }
                }
            } while (false);

            output.reflected += LrEstimate / splittingFactor;
        }

        if (output.depthAcc == 0) {
            /// all BSDF samples have failed :-(
            output.markAsLeaf(rRec.depth);
        }

        return output;
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
        oss << "MIFocalGuidingPathTracer[" << endl
            << "  maxDepth = " << m_maxDepth << "," << endl
            << "  rrDepth = " << m_rrDepth << "," << endl
            << "  strictNormals = " << m_strictNormals << endl
            << "]";
        return oss.str();
    }

private:
    std::unique_ptr<pmm_focal::StatsRecursiveImageBlocks> m_statsImages;
    mutable ref<ImageBlock> m_debugImage;
    mutable ref<Film> m_debugFilm;

    struct WeightedBitmapAccumulator {
        void clear() {
            m_scrap = nullptr;
            m_bitmap = nullptr;
            m_spp = 0;
            m_weight = 0;
        }

        bool hasData() const {
            return m_weight > 0;
        }

        void add(const ref<Film> &film, int spp, Float avgVariance = 1) {
            if (avgVariance == 0 && m_weight > 0) {
                SLog(EError, "Cannot add an image with unknown variance to an already populated accumulator");
                return;
            }

            const Vector2i size = film->getSize();
            const long floatCount = size.x * size.y * long(SPECTRUM_SAMPLES);

            if (!m_scrap) {
                m_scrap = new Bitmap(Bitmap::EPixelFormat::ESpectrum, Bitmap::EComponentFormat::EFloat32, size);
            }
            film->develop(Point2i(0, 0), size, Point2i(0, 0), m_scrap);

            ///

            if (!m_bitmap) {
                m_bitmap = new Bitmap(Bitmap::EPixelFormat::ESpectrum, Bitmap::EComponentFormat::EFloat32, size);

                Float *m_bitmapData = m_bitmap->getFloat32Data();
                for (long i = 0; i < floatCount; ++i) {
                    m_bitmapData[i] = 0;
                }
            }

            Float *m_bitmapData = m_bitmap->getFloat32Data();
            if (avgVariance > 0 && m_weight == 0 && m_spp > 0) {
                /// reweight previous frames that had unknown variance with our current variance estimate
                const Float reweight = 1 / avgVariance;
                for (long i = 0; i < floatCount; ++i) {
                    m_bitmapData[i] *= reweight;
                }
                m_weight += m_spp * reweight;
            }

            const Float weight = avgVariance > 0 ? spp / avgVariance : spp;
            const Float *m_scrapData = m_scrap->getFloat32Data();
            for (long i = 0; i < floatCount; ++i) {
                m_bitmapData[i] += m_scrapData[i] * weight;
            }

            m_weight += avgVariance > 0 ? weight : 0;
            m_spp += spp;
        }

        void develop(Bitmap *dest) const {
            if (!m_bitmap) {
                SLog(EWarn, "Cannot develop bitmap, as no data is available");
                return;
            }

            const Vector2i size = m_bitmap->getSize();
            const long floatCount = size.x * size.y * long(SPECTRUM_SAMPLES);

            const Float weight = m_weight == 0 ? m_spp : m_weight;
            Float *m_destData = dest->getFloat32Data();
            const Float *m_bitmapData = m_bitmap->getFloat32Data();
            for (long i = 0; i < floatCount; ++i) {
                m_destData[i] = weight > 0 ? m_bitmapData[i] / weight : 0;
            }
        }

        void develop(const fs::path &path) const {
            if (!m_scrap) {
                SLog(EWarn, "Cannot develop bitmap, as no data is available");
                return;
            }

            develop(m_scrap.get());
            m_scrap->write(path);
        }

    private:
        mutable ref<Bitmap> m_scrap;
        ref<Bitmap> m_bitmap;
        Float m_weight;
        int m_spp;
    } m_finalImage;

    int m_iteration;
    int m_passesRenderedGlobal;

    /// The total allocated render time in seconds, including the training phase.
    Float m_budget;

    /// The time allocated for each training iteration in seconds.
    Float m_iterationBudget;

    /// The number of training iterations to be performed.
    int m_iterationCount;

    /// Whether to dump the geometry of the scene for visualization purposes.
    bool m_dumpScene;

    /**
     * The converging guiding distribution handles cases where the focal point lies in positive direction of the ray.
     * Some focal points can only be converging (e.g., occlusion focal points), while others can also occur in negative direction
     * of the ray (e.g., a virtual image projected behind a surface).
     * For the converging field, we choose the ray direction to point towards the sampled point.
     */
    mutable Distribution m_converging;

    /**
     * The diverging guiding distribution handles cases where focal points lie in negative direction of the ray.
     * An example of a diverging focal point is a defocused lens that creates its virtual image behind the sensor:
     * In this case, our rays do not converge in the focal point (as it occurs for a negative ray parameter t),
     * instead the rays diverge from it.
     * For the diverging field, we choose the ray direction to point away from the sampled point.
     */
    mutable Distribution m_diverging;

    /**
     * In our paper we handle diverging focal points by partitioning the region of interest by an additional set of
     * spatial regions that choose a diverging direction when sampled.
     * A simple way to implement this is by storing two octrees ( @c m_convering and @c m_diverging ) and choosing randomly
     * between the two depending on the total weight in each tree.
     * This variable contains the probability of sampling the diverging guiding distribution.
     */
    Float m_divergeProbability;

    mutable std::unique_ptr<ProgressReporter> m_progress;
    std::chrono::steady_clock::time_point m_startTime;

public:
    MTS_DECLARE_CLASS()
};

MTS_IMPLEMENT_CLASS(MIFocalGuidingPathTracer, false, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(MIFocalGuidingPathTracer, "MI focal guiding path tracer");
MTS_NAMESPACE_END
