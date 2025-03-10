#include <mitsuba/mitsuba.h>
#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/fstream.h>
#include <mitsuba/core/logger.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/sched.h>
#include <mitsuba/core/statistics.h>
#include <mitsuba/core/tls.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/film.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/sampler.h>
#include <mitsuba/render/scene.h>
#include <mitsuba/render/sensor.h>
#include <mitsuba/render/renderproc.h>
#include <deque>
#include <fstream>
#include <filesystem>
#include <sstream>

#include "gaussian_mixture_model.h"
#include "octree.h"
#include "weighted_sample.h"

#include "./envs/mitsuba_env.h"

MTS_NAMESPACE_BEGIN

static StatsCounter avgPathLength("Path tracer", "Average path length", EAverage);
static StatsCounter samplesFromGMMCount("Path tracer", "Sampled from GMM", ENumberValue);

#include <vector>
#include <Eigen/Dense>
#include <mutex>
#include <sstream>
#include <string>

struct ContributionAndThroughput {
    Spectrum contribution {0.0f};
    Spectrum throughput   {1.0f};

    explicit ContributionAndThroughput(const Spectrum& contribution, const Spectrum& throughput)
        : contribution{contribution}, throughput{throughput} {}

    Spectrum getClamped(const float maxThroughput)
    {
        if (throughput.max() > maxThroughput)
            return contribution*throughput*(maxThroughput/throughput.max());
        return contribution*throughput;
    }
};

struct IntersectionData {
    //direct light from next its, bsdf*miWeight
    ContributionAndThroughput bsdfDirectLight;
    //direct light from next event estimation, bsdf*miWeight
    ContributionAndThroughput neeDirectLight;
    //self emission
    ContributionAndThroughput emission;
    float bsdfMiWeight {1.0f};

    IntersectionData()
        : bsdfDirectLight(Spectrum(0.0f), Spectrum(1.0f)),
          neeDirectLight(Spectrum(0.0f), Spectrum(1.0f)),
          emission(Spectrum(0.0f), Spectrum(1.0f)) {}
};

/* Based on MIPathTracer */
class PMMFocalGuidingIntegrator : public MonteCarloIntegrator {
    using Tree = pmm_focal::Octree<EnvMitsuba3D>;

    mutable Tree m_octree;
    // todo
    // probably not double
    using Scalar = double;
    using Vectord = Eigen::Matrix<Scalar, 3, 1>;
    mutable pmm_focal::GaussianMixtureModel<Scalar, EnvMitsuba3D> m_gmm;
    ref<Timer> m_timer;
    uint32_t m_renderMaxSeconds;
    AABB sceneSpace;
    mutable std::vector<std::vector<pmm_focal::WeightedSample>> perThreadSamples;
    mutable std::vector<std::vector<mitsuba::RayDifferential>> perThreadZeroValuedSamples;
    mutable std::atomic<size_t> maxThreadId;

    mutable PrimitiveThreadLocal<bool> threadLocalInitialized;
    uint32_t minSamplesToStartFitting;
    uint32_t samplesPerIteration;
    uint32_t trainingIterations;
    uint32_t trainingSamples;
    uint32_t allCollectedSamplesCount = 0;
    int maxDepth;

    bool training; // only collecting samples while training
    bool shouldUseGuiding = false; // no use in using guiding for the first iteration


public:
    PMMFocalGuidingIntegrator(const Properties &props)
        : MonteCarloIntegrator(props) {
            m_octree.configuration.threshold = props.getFloat("orth.threshold", 1e-3);
            m_octree.configuration.minDepth = props.getInteger("orth.minDepth", 0);
            m_octree.configuration.maxDepth = props.getInteger("orth.maxDepth", 40);
            m_octree.configuration.decay = props.getFloat("orth.decay", 0.5f);

            m_gmm.setAlpha(props.getFloat("gmm.alpha", 0.25));
            m_gmm.setSplittingThreshold(props.getFloat("gmm.splittingThreshold", 7.0));
            m_gmm.setMergingThreshold(props.getFloat("gmm.mergingThreshold", 0.25));
            Log(EInfo, "merging threshold: %f, splitting theshold: %f", m_gmm.getMergingThreshold(), m_gmm.getSplittingThreshold());
            m_gmm.setMinNumComp(props.getInteger("gmm.minNumComp", 10));
            m_gmm.setMaxNumComp(props.getInteger("gmm.maxNumComp", 15));

            minSamplesToStartFitting = static_cast<uint32_t>(props.getSize("minSamplesToStartFitting", 12)); // todo: use
            samplesPerIteration = static_cast<uint32_t>(props.getSize("samplesPerIteration", 4));

            Log(EInfo, "minSamplesToStartFitting = %zu", minSamplesToStartFitting);

            m_renderMaxSeconds = static_cast<uint32_t>(props.getSize("renderMaxSeconds", 300)); // 5 min
            this->maxDepth = props.getInteger("maxDepth", 40);
            this->trainingIterations = static_cast<uint32_t>(props.getSize("iterationCount"), 3);
            this->trainingSamples = static_cast<uint32_t>(props.getSize("trainingSamples", 4));
            m_timer = new Timer{false};
            Log(EInfo, this->toString().c_str());
        }

    PMMFocalGuidingIntegrator(Stream *stream, InstanceManager *manager)
        : MonteCarloIntegrator(stream, manager) { }

    bool preprocess(const Scene *scene, RenderQueue *queue, const RenderJob *job, int sceneResID, int sensorResID, int samplerResID) {
        Log(EInfo, "Starting preprocess");

        sceneSpace = scene->getAABB();

        m_octree.setAABB(scene->getAABB());
        Log(EInfo, m_octree.toStringVerbose().c_str());

        // we initialize with an avarage of max and min
        size_t initialComponentCount = (m_gmm.getMaxNumComp() + m_gmm.getMinNumComp()) / 2;
        m_gmm.init(initialComponentCount, 3, scene->getAABB());
        Log(EInfo, m_gmm.toString().c_str());

        // per thread storage for sample data
        ref<Scheduler> sched = Scheduler::getInstance();
        const size_t nCores = sched->getCoreCount();
        // m_perThreadIntersectionData.resize(nCores);
        perThreadSamples.resize(nCores);
        perThreadZeroValuedSamples.resize(nCores);
        maxThreadId.store(0);

        train(static_cast<Scene*>(sched->getResource(sceneResID)), queue, job, sensorResID);

        ref<Sensor> sensor = static_cast<Sensor *>(sched->getResource(sensorResID));
        iterationPreprocess(sensor->getFilm());

        training = false;

        return true;
    }

    void train(Scene *scene, RenderQueue *queue, const RenderJob *job, int sensorResID) {
        training = true;

        ref<Scheduler> sched = Scheduler::getInstance();
        ref<Sensor> sensor = static_cast<Sensor *>(sched->getResource(sensorResID));
        ref<Film> film = sensor->getFilm();

        // create copy of the scene
        ref<Scene> trainingScene = new Scene(scene);
        const int trainingSceneResID = sched->registerResource(trainingScene);

        Log(EInfo, "Starting training...");

        for (size_t i = 0; i < this->trainingIterations; ++i) {
            Log(EInfo, "Rendering %i iteration", i);

            Properties trainingSamplerProps = scene->getSampler()->getProperties();
            trainingSamplerProps.removeProperty("sampleCount");
            trainingSamplerProps.setSize("sampleCount", trainingSamples);
            ref<Sampler> trainingSampler = static_cast<Sampler*>(PluginManager::getInstance()->createObject(MTS_CLASS(Sampler), trainingSamplerProps));
            trainingSampler->configure();

            trainingScene->setSampler(trainingSampler);

            /* Create a sampler instance for every core */
            std::vector<SerializableObject *> samplers(sched->getCoreCount());
            for (size_t i=0; i<sched->getCoreCount(); ++i) {
                ref<Sampler> clonedSampler = trainingSampler->clone();
                clonedSampler->incRef();
                samplers[i] = clonedSampler.get();
            }

            int trainingSamplerResID = sched->registerMultiResource(samplers);

            for (size_t i=0; i<sched->getCoreCount(); ++i)
                samplers[i]->decRef();

            iterationPreprocess(film);
            // fully render training scene
            // okay, it does make sence - we update gmm after every render, so each training iteration, gmm should be getting smarter
            SamplingIntegrator::render(trainingScene, queue, job, trainingSceneResID, sensorResID, trainingSamplerResID);
            iterationPostprocess(film, samplesPerIteration, job); // this is where we update the gmm

            sched->unregisterResource(trainingSamplerResID);
            shouldUseGuiding = true;
        }

        Log(EInfo, "Finished training phase. Starting real rendering.");
    }

    void iterationPreprocess(ref<Film> film)
    {
        film->clear();
        Statistics::getInstance()->resetAll();

        m_timer->reset();
    }

    void iterationPostprocess(const ref<Film> film, uint32_t numSPP, const RenderJob *job) {
        const Float renderTime = m_timer->stop();
        // Log(EInfo, "Post processing of the rendering iteration");
        Log(EInfo, "iteration render time: %s", timeString(renderTime, true).c_str());

        if (!training)
            return;

        const size_t numValidSamples = std::accumulate(perThreadSamples.begin(), perThreadSamples.end(), 0UL,
            [](size_t sum, const std::vector<pmm_focal::WeightedSample>& samples) -> size_t { return sum+samples.size(); });

        allCollectedSamplesCount += numValidSamples;
        if (allCollectedSamplesCount == 0) {
            Log(EInfo, "No samples found during training, resetting gmm.");
            m_gmm.init(m_gmm.getMinNumComp(), 3, sceneSpace);
        }


        if (numValidSamples < minSamplesToStartFitting) {
            Log(EInfo, "skipping fit due to insufficient sample data (got %zu/%d valid samples).", numValidSamples, minSamplesToStartFitting);
            return;
        }

        m_timer->reset();
        // flatten the samples vector
        std::vector<pmm_focal::WeightedSample> iterationSamples = std::accumulate(perThreadSamples.begin(), perThreadSamples.end(), std::vector<pmm_focal::WeightedSample>{},
        [](std::vector<pmm_focal::WeightedSample>& acc, const std::vector<pmm_focal::WeightedSample>& vec) {
            acc.insert(acc.end(), vec.begin(), vec.end());
            return acc;
        });
        // for (auto sample : iterationSamples) {
        //     Log(EInfo, sample.toString().c_str());
        // }

        // flatten the zero-samples vector
        std::vector<RayDifferential> iterationZeroValuedSamples = std::accumulate(perThreadZeroValuedSamples.begin(), perThreadZeroValuedSamples.end(), std::vector<RayDifferential>{},
        [](std::vector<RayDifferential>& acc, const std::vector<RayDifferential>& vec) {
            acc.insert(acc.end(), vec.begin(), vec.end());
            return acc;
        });

        Log(EInfo, "Got %i non-zero samples and %i zero samples", iterationSamples.size(), iterationZeroValuedSamples.size());
        m_gmm.processBatch(iterationSamples);
        const Float postprocessingTime = m_timer->stop();
        Log(EInfo, "iteration postprocessing time: %s", timeString(postprocessingTime, true).c_str());
        m_timer->reset();
        Log(EInfo, m_gmm.toString().c_str());
        Log(EInfo, m_octree.toString().c_str());
        Log(EDebug, m_octree.toStringVerbose().c_str());
    }

    bool render(Scene *scene, RenderQueue *queue, const RenderJob *job, int sceneResID, int sensorResID, int samplerResID) {
        ref<Timer> renderTimer = new Timer{};

        ref<Scheduler> sched = Scheduler::getInstance();
        ref<Sensor> sensor = static_cast<Sensor *>(sched->getResource(sensorResID));
        ref<Film> film = sensor->getFilm();

        size_t nCores = sched->getCoreCount();
        const Sampler *sampler = static_cast<const Sampler *>(sched->getResource(samplerResID, 0));
        size_t sampleCount = sampler->getSampleCount();

        Log(EInfo, "Starting render job (%ix%i, " SIZE_T_FMT " %s, " SIZE_T_FMT
            " %s, " SSE_STR ") ..", film->getCropSize().x, film->getCropSize().y,
            sampleCount, sampleCount == 1 ? "sample" : "samples", nCores,
            nCores == 1 ? "core" : "cores");

        int integratorResID = sched->registerResource(this);

        // Log(EInfo, m_gmm.toString().c_str());
        // Log(EInfo, m_octree.toString().c_str());
        // Log(EDebug, m_octree.toStringVerbose().c_str());

        Float totalRenderTime;
        Float iterationRenderTime;
        ParallelProcess::EStatus status;
        size_t iterations = 0;
        do { // it's going to be at least one iteration
            /* This is a sampling-based integrator - parallelize */
            ref<ParallelProcess> proc = new BlockedRenderProcess(job,
                queue, scene->getBlockSize());
            proc->bindResource("integrator", integratorResID);
            proc->bindResource("scene", sceneResID);
            proc->bindResource("sensor", sensorResID);
            proc->bindResource("sampler", samplerResID);
            scene->bindUsedResources(proc);
            sched->schedule(proc);

            m_process = proc;
            sched->wait(proc);
            m_process = nullptr;

            status = proc->getReturnStatus();

            iterationRenderTime = renderTimer->lap();
            totalRenderTime = renderTimer->getSeconds();
            ++iterations;
        } while(status == ParallelProcess::ESuccess && totalRenderTime + iterationRenderTime < m_renderMaxSeconds);

        Log(EInfo, "rendered %zu samples per pixel in %s.", iterations*sampler->getSampleCount(), timeString(renderTimer->getMilliseconds()/1000.0f, true).c_str());
        sched->unregisterResource(integratorResID);

        return status == ParallelProcess::ESuccess;
    }


    Spectrum sampleFromGMM(
        const BSDF* bsdf,
        BSDFSamplingRecord& bRec,
        Float& woPdf,
        Float& bsdfPdf,
        Float bsdfSamplingFraction,
        RadianceQueryRecord rRec
    ) const {
        mitsuba::Point2 sample = rRec.nextSample2D(); // this is direction (azimutal and polar coords)

        auto type = bsdf->getType();

        // EDelta means discrete number of directions
        // - unsuitable for guiding or importance sampling based on density distributions
        // that work over ranges of directions
        if ((type & BSDF::EDelta) == (type & BSDF::EAll) || !shouldUseGuiding) { // first iteration we should not use guiding
            auto result = bsdf->sample(bRec, bsdfPdf, sample); // this sets bsdfPdf
            woPdf = bsdfPdf;
            return result;
        }

        Spectrum result;
        if (sample.x < bsdfSamplingFraction) { // bsdfSamplingFraction is from MIS
            // sample BSDF
            sample.x /= bsdfSamplingFraction;
            result = bsdf->sample(bRec, bsdfPdf, sample);
            if (result.isZero()) {
                woPdf = bsdfPdf = 0;
                return Spectrum{0.0f};
            }

            // If we sampled a delta component, then we have a 0 probability
            // of sampling that direction via guiding, thus we can return early.
            if (bRec.sampledType & BSDF::EDelta) {
                woPdf = bsdfPdf * bsdfSamplingFraction;
                return result / bsdfSamplingFraction; // woPdf and bsdfPdf set - it's fine
            }

            result *= bsdfPdf;

        } else {
            // sample guiding distribution
            avgPathLength++; // add to statistics

            sample.x = (sample.x - bsdfSamplingFraction) / (1 - bsdfSamplingFraction);
            std::random_device rd;
            std::mt19937 gen(rd());

            Eigen::VectorXd gmmSample = m_gmm.sample(gen);
            mitsuba::Point endPoint(gmmSample[0], gmmSample[1], gmmSample[2]);
            // wo is outgoing direction
            bRec.wo = normalize(endPoint - bRec.its.p);
            bRec.wo = bRec.its.toLocal(bRec.wo);

            // // idk - they say it's hack
            bRec.eta = 1; // eta is Relative index of refraction in the sampled direction. Refractive index determines how much the path of light is bent, or refracted, when entering a material.
            bRec.sampledType = BSDF::EDiffuse;

            result = bsdf->eval(bRec);

            if (result.isZero()) {
                // invalid (aka zero contribution) direction - so we don't set wo and bdfPdf I guess
                return result;
            }
        }

        pdfMat(woPdf, bsdfPdf, bsdf, bRec); // woPdf and bsdfPdf set - it's fine
        if (woPdf == 0) {
            return Spectrum{0.0f};
        }
        return result / woPdf;
        // from what I can tell right now, always: woPdf == bsdfPdf
    }

    void pdfMat(
        Float& woPdf, Float& bsdfPdf, // Float bsdfSamplingFraction,
        const BSDF* bsdf, const BSDFSamplingRecord& bRec
    ) const {
        woPdf = bsdfPdf = bsdf->pdf(bRec); // leaving this for now
    }

    Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const {
        Spectrum Li(0.0f);
        IntersectionData intersectionData;

        static thread_local std::vector<pmm_focal::WeightedSample>* samples {nullptr};
        static thread_local std::vector<RayDifferential>* zeroValuedSamples {nullptr}; // todo: might not be WeigthedSample

        // init thread variables
        if (training && !threadLocalInitialized.get()) {
            const size_t threadId = maxThreadId.fetch_add(1, std::memory_order_relaxed);
            samples = &perThreadSamples.at(threadId);
            zeroValuedSamples = &perThreadZeroValuedSamples.at(threadId);
            threadLocalInitialized.get() = true;
        }

        /* Some aliases and local variables */
        const Scene *scene = rRec.scene;
        Intersection &its = rRec.its;
        RayDifferential ray(r);

        bool scattered = false;

        /* Perform the first ray intersection (or ignore if the
           intersection has already been provided). */
        rRec.rayIntersect(ray);
        ray.mint = Epsilon;

        Spectrum throughput(1.0f);
        Float eta = 1.0f;

        while (rRec.depth <= maxDepth || maxDepth < 0) {
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
                && (!m_hideEmitters || scattered)) {
                const Spectrum emittedRadiance = its.Le(-ray.d);
                Li += throughput * emittedRadiance;
                intersectionData.emission.contribution = emittedRadiance;
            }


            /* Include radiance from a subsurface scattering model if requested */
            if (its.hasSubsurface() && (rRec.type & RadianceQueryRecord::ESubsurfaceRadiance)) {
                const Spectrum subsurfaceScatteredRadiance = its.LoSub(scene, rRec.sampler, -ray.d, rRec.depth);
                Li += throughput * subsurfaceScatteredRadiance;

                intersectionData.emission.contribution += subsurfaceScatteredRadiance;
            }

            if ((rRec.depth >= maxDepth && maxDepth > 0)
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

            if (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance &&
                (bsdf->getType() & BSDF::ESmooth)) {
                Spectrum value = scene->sampleEmitterDirect(dRec, rRec.nextSample2D());
                if (!value.isZero()) {
                    // so this is almost never true for our scenes - no wonder, it's very difficult to hit the emitter
                    const Emitter *emitter = static_cast<const Emitter *>(dRec.object);

                    /* Allocate a record for querying the BSDF */
                    BSDFSamplingRecord bRec(its, its.toLocal(dRec.d), ERadiance);

                    /* Evaluate BSDF * cos(theta) */
                    const Spectrum bsdfVal = bsdf->eval(bRec);

                    /* Prevent light leaks due to the use of shading normals */
                    if (!bsdfVal.isZero() && (!m_strictNormals
                            || dot(its.geoFrame.n, dRec.d) * Frame::cosTheta(bRec.wo) > 0)) {

                        /* Calculate prob. of having generated that direction
                            using BSDF sampling */
                            // it both other solutions here is where pdf is calculated from guided distribution
                        Float bsdfPdf = (emitter->isOnSurface() && dRec.measure == ESolidAngle)
                            ? bsdf->pdf(bRec) : 0; // should this be also guided somehow?

                        /* Weight using the power heuristic */
                        Float weight = miWeight(dRec.pdf, bsdfPdf);
                        // Log(EInfo, ("throughput = " + throughput.toString() + " value = " + value.toString() + " bsdfVal = " + bsdfVal.toString() + " weight = %f").c_str(), weight);
                        Li += throughput * value * bsdfVal * weight;
                        intersectionData.neeDirectLight = ContributionAndThroughput{value, bsdfVal*weight};
                    }
                }
            }

            /* ==================================================================== */
            /*                            BSDF sampling                             */
            /* ==================================================================== */

            /* Sample BSDF * cos(theta) */
            Float bsdfPdf, woPdf;
            Float bsdfSamplingFraction = 0.5;

            BSDFSamplingRecord bRec(its, rRec.sampler, ERadiance);

            // here is where sampling takes place -- should sample guided
            Spectrum bsdfWeight = sampleFromGMM(bsdf, bRec, woPdf, bsdfPdf, bsdfSamplingFraction, rRec);
            if (bsdfWeight.isZero()) // this is why woPdf and bsdfPdf doesn't matter here
                break;

            scattered |= bRec.sampledType != BSDF::ENull;

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
                    Log(EInfo, ("hit luminaire, value = " + value.toString()).c_str());
                    dRec.setQuery(ray, its);
                    hitEmitter = true;
                }
            } else {
                /* Intersected nothing -- perhaps there is an environment map? */
                const Emitter *env = scene->getEnvironmentEmitter();

                if (env) {
                    Log(EInfo, "Environment map");
                    if (m_hideEmitters && !scattered)
                        break;

                    value = env->evalEnvironment(ray);
                    if (!env->fillDirectSamplingRecord(dRec, ray))
                        break;
                    hitEmitter = true;
                } else {
                    break;
                }
            }

            /* Keep track of the throughput and relative
                refractive index along the path */
            throughput *= bsdfWeight;
            eta *= bRec.eta;

            /* If a luminaire was hit, estimate the local illumination and
                weight using the power heuristic */
            if (hitEmitter &&
                (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance)) {
                /* Compute the prob. of generating that direction using the
                    implemented direct illumination sampling technique */
                const Float lumPdf = (!(bRec.sampledType & BSDF::EDelta)) ?
                    scene->pdfEmitterDirect(dRec) : 0;
                Float weight = miWeight(woPdf, lumPdf);
                Li += throughput * value * weight;

                // Log(EInfo, ("Throughput = " + throughput.toString() + " value = " + value.toString() + " weight = %f").c_str(), weight);
                intersectionData.bsdfDirectLight = ContributionAndThroughput{value, bsdfWeight*weight};
                intersectionData.bsdfMiWeight =weight;
            }

            /* ==================================================================== */
            /*                         Indirect illumination                        */
            /* ==================================================================== */

            /* Set the recursive query type. Stop if no surface was hit by the
                BSDF sample or if indirect illumination was not requested */
            if (!its.isValid() || !(rRec.type & RadianceQueryRecord::EIndirectSurfaceRadiance))
                break;

            rRec.type = RadianceQueryRecord::ERadianceNoEmission;

            if (rRec.depth++ >= m_rrDepth) {
                /* Russian roulette: try to keep path weights equal to one,
                    while accounting for the solid angle compression at refractive
                    index boundaries. Stop with at least some probability to avoid
                    getting stuck (e.g. due to total internal reflection) */

                Float q = std::min(throughput.max() * eta * eta, (Float) 0.95f);
                if (rRec.nextSample1D() >= q)
                    break;
                throughput /= q;
            }

            if (training) {
                // collect samples while training
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

                    // todo: vmm had some more restrains for samples (sampel type and roughness)
                    Spectrum clampedLight {0.0f};
                    const float maxThroughput = 10.0f;

                clampedLight += intersectionData.bsdfDirectLight.getClamped(maxThroughput);
                clampedLight += intersectionData.neeDirectLight.getClamped(maxThroughput);
                clampedLight += intersectionData.emission.getClamped(maxThroughput);
                clampedLight += intersectionData.bsdfDirectLight.contribution*intersectionData.bsdfMiWeight;

                if (clampedLight.average() < 1e-6) {
                    // do nothing
                } else {
                    std::vector<Eigen::VectorXd> points;
                    m_octree.splat(ray.o, ray.d, splatDistance, points);
                    for(auto& point : points) {
                        if (point.size() == 0) {
                            SLog(EInfo, "Intersection was not found");
                            continue; // if for some reason the intersection was not found, don't save the point
                        }
                        samples->push_back({point, clampedLight.average()});
                        // Log(EInfo, "[%f, %f, %f]", point[0], point[1], point[2]);
                    }
                }
            }
        }

        /* Store statistics */
        avgPathLength.incrementBase();
        avgPathLength += rRec.depth;

        return Li;
    }

    bool fileExists(const std::string &filename) const {
        std::ifstream file(filename);
        return file.good(); // Returns true if file exists and is accessible
    }

    void dumpVectorToTextFile(const std::vector<Eigen::VectorXd> &data, const std::string &filename) const {
        bool file_exists = fileExists(filename);

        // Open the file for writing (overwriting or appending)
        std::ofstream file;
        if (file_exists) {
            file.open(filename, std::ios::app); // Append if the file already exists
        } else {
            file.open(filename); // Create a new file
        }

        for (const auto &vec : data) {
            for (int i = 0; i < vec.size(); ++i) {
                file << vec[i];
                if (i != vec.size() - 1) {
                    file << " ";
                }
            }
            file << "\n";
        }

        file.close();
    }

    void postprocess(const Scene *scene, RenderQueue *queue, const RenderJob *job, int sceneResID, int sensorResID, int samplerResID)
    {

        ref<Scheduler> sched = Scheduler::getInstance();
        ref<Sensor> sensor = static_cast<Sensor *>(sched->getResource(sensorResID));
        ref<Film> film = sensor->getFilm();

        iterationPostprocess(film, static_cast<uint32_t>(scene->getSampler()->getSampleCount()), job);
        perThreadSamples.clear();
        perThreadZeroValuedSamples.clear();

        Log(EInfo, m_gmm.toString().c_str());
    }

    inline Float miWeight(Float pdfA, Float pdfB) const {
        if (pdfA == 0.0 && pdfB == 0.0)
            return 0.0; // make sure you're not dividing by 0
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
            << "  maxDepth = " << maxDepth << "," << endl
            << "  rrDepth = " << m_rrDepth << "," << endl
            << "  strictNormals = " << m_strictNormals << endl
            << "  gmm = " << m_gmm.toString() << endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
};

MTS_IMPLEMENT_CLASS_S(PMMFocalGuidingIntegrator, false, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(PMMFocalGuidingIntegrator, "PMM focal guiding path tracer")
MTS_NAMESPACE_END
