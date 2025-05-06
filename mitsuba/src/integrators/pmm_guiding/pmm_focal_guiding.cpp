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
#include "iterable_blocked_vector.h"


MTS_NAMESPACE_BEGIN

static StatsCounter avgPathLength("Path tracer", "Average path length", EAverage);
static StatsCounter samplesFromGMMCount("Path tracer", "Sampled from GMM", ENumberValue);
static StatsCounter samplesFromDiverging("Path tracer", "Sampled from Diverging", ENumberValue);
static StatsCounter samplesFromConverging("Path tracer", "Sampled from Converging", ENumberValue);

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

    Spectrum getClamped(const float maxThroughput) const
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
    float woPdf {0.0f};
    float distance;
    float endpointRoughness;
    mitsuba::Spectrum throughputFactors;
    unsigned int sampledType;
    const Intersection& its;

    IntersectionData(const Intersection& its)
        : bsdfDirectLight(Spectrum(0.0f), Spectrum(1.0f)),
          neeDirectLight(Spectrum(0.0f), Spectrum(1.0f)),
          emission(Spectrum(0.0f), Spectrum(1.0f)), distance(its.t), its(its) {}

    IntersectionData& operator*=(Spectrum factor) {
        bsdfDirectLight.throughput *= factor;
        neeDirectLight.throughput  *= factor;
        emission.throughput        *= factor;

        return *this;
    }
};

/* Based on MIPathTracer */
class PMMFocalGuidingIntegrator : public MonteCarloIntegrator {
    using Tree = pmm_focal::Octree<EnvMitsuba3D>;

    mutable Tree m_octree;
    mutable Tree m_octreeDiverging;

    std::string importGmmFile;
    std::string saveGMMDirectory;
    // todo
    using Scalar = float;
    using Vectord = Eigen::Matrix<Scalar, 3, 1>;
    mutable pmm_focal::GaussianMixtureModel<Scalar, EnvMitsuba3D> m_gmm;
    ref<Timer> m_timer;
    uint32_t m_renderMaxSeconds;
    AABB sceneSpace;
    mutable std::vector<std::vector<IntersectionData>> perThreadIntersectionData;
    mutable std::vector<IterableBlockedVector<pmm_focal::WeightedSample>> perThreadSamples;
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
    float divergeProbability = 0.5f;
    float bsdfMISFraction = 0.5f;


public:
    PMMFocalGuidingIntegrator(const Properties &props)
        : MonteCarloIntegrator(props) {
            importGmmFile = props.getString("importGmmFile", "");
            saveGMMDirectory = props.getString("saveGMMDirectory", "");
            Log(EInfo, ("Saving to: " + saveGMMDirectory).c_str());

            bsdfMISFraction = props.getFloat("bsdfMISFraction", 0.5);
            Log(EInfo, "mis fraction: %f", bsdfMISFraction);

            m_octree.configuration.threshold = props.getFloat("tree.threshold", 1e-3);
            m_octree.configuration.minDepth = props.getInteger("tree.minDepth", 0);
            m_octree.configuration.maxDepth = props.getInteger("tree.maxDepth", 40);
            m_octree.configuration.decay = props.getFloat("tree.decay", 0.5f);

            m_octreeDiverging.configuration = m_octree.configuration;

            m_gmm.setAlpha(props.getFloat("gmm.alpha", 0.25));
            m_gmm.setSplittingThreshold(props.getFloat("gmm.splittingThreshold", 7.0));
            m_gmm.setMergingThreshold(props.getFloat("gmm.mergingThreshold", 0.25));
            m_gmm.setMinNumComp(props.getInteger("gmm.minNumComp", 10));
            m_gmm.setMaxNumComp(props.getInteger("gmm.maxNumComp", 15));
            m_gmm.setInitMethod(props.getString("gmm.initMethod", "Random"));

            Log(EInfo, ("GMM params: alpha = %f, split_th = %f, merge_th = %f, min_num_comp = %d, max_num_comp = %d, init = " + m_gmm.getInitMethod()).c_str(), m_gmm.getAlpha(), m_gmm.getSplittingThreshold(), m_gmm.getMergingThreshold(), m_gmm.getMinNumComp(), m_gmm.getMaxNumComp());

            minSamplesToStartFitting = static_cast<uint32_t>(props.getInteger("minSamplesToStartFitting", 12));
            samplesPerIteration = static_cast<uint32_t>(props.getSize("samplesPerIteration", 4));

            m_renderMaxSeconds = static_cast<uint32_t>(props.getInteger("renderMaxSeconds", 3000)); // 5 min
            this->maxDepth = props.getInteger("maxDepth", 40);
            this->trainingIterations = static_cast<uint32_t>(props.getInteger("iterationCount", 10));
            Log(EInfo, "iterationCount: %d", this->trainingIterations);
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
        m_octreeDiverging.setAABB(scene->getAABB());
        Log(EInfo, m_octree.toString().c_str());
        Log(EInfo, m_octreeDiverging.toString().c_str());

        if (!importGmmFile.empty()) {
            // initialize from file
            Log(EInfo, "Importing GMM from a file");
            ref<FileStream> gmmImportStream = new FileStream(importGmmFile, FileStream::EReadOnly);
            m_gmm.deserialize(gmmImportStream);
            gmmImportStream->close();
            shouldUseGuiding = true; // when we load, we can use guiding right away!
            divergeProbability = m_gmm.getDivergeProbability();
        } else {
            // do nothing - the gmm is initialize with the first training batch
        }
        m_gmm.setAABB(scene->getAABB());

        Log(EInfo, m_gmm.toString().c_str());

        // per thread storage for sample data
        ref<Scheduler> sched = Scheduler::getInstance();
        const size_t nCores = sched->getCoreCount();
        perThreadIntersectionData.resize(nCores);
        perThreadSamples.resize(nCores);
        maxThreadId.store(0);

        if (importGmmFile.empty()) {
            // only train if GMM was not imported
            train(static_cast<Scene*>(sched->getResource(sceneResID)), queue, job, sensorResID);
        }

        ref<Sensor> sensor = static_cast<Sensor *>(sched->getResource(sensorResID));
        iterationPreprocess(sensor->getFilm()); // clears the film

        if (!saveGMMDirectory.empty()) {
            const std::string gmmFile = saveGMMDirectory + "final_gmm.serialized";
            ref<FileStream> gmmSerializationStream = new FileStream(gmmFile, FileStream::ETruncWrite);
            m_gmm.serialize(gmmSerializationStream);
            gmmSerializationStream->close();
            Log(EInfo, "final guiding field written to %s", gmmFile.c_str());
        }

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

        Log(EInfo, "Starting training... (%d) iterations", trainingIterations);

        for (size_t i = 0; i < this->trainingIterations; ++i) {
            if (!training) {
                Log(EInfo, "GMM converged, finishing training early");
            }
            Log(EInfo, "Rendering %i iteration", i);
            if (i > this->trainingIterations * 0.66) {
                // iterative narrowing applied after we're done with 66% of the training
                Log(EInfo, "starting now, iterative narrowing is applied");
                m_octree.configuration.splattingStrategy = Tree::SPLAT_RAY_WEIGHTED;
                m_octreeDiverging.configuration.splattingStrategy = Tree::SPLAT_RAY_WEIGHTED;
            }

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

        // std::vector<mitsuba::Point> iterationSampledPoints = std::accumulate(perThreadSampledPoints.begin(), perThreadSampledPoints.end(), std::vector<mitsuba::Point>{},
        // [](std::vector<mitsuba::Point>& acc, const std::vector<mitsuba::Point>& vec) {
        //     acc.insert(acc.end(), vec.begin(), vec.end());
        //     return acc;
        // });
        // for(size_t i = 0; i < iterationSampledPoints.size(); ++i) {
        //     if (i%1000 == 0){
        //         Log(EInfo, iterationSampledPoints[i].toString().c_str());
        //     }
        // }

        if (!training)
            return;

        const size_t numValidSamples = std::accumulate(perThreadSamples.begin(), perThreadSamples.end(), 0UL,
            [](size_t sum, const IterableBlockedVector<pmm_focal::WeightedSample>& samples) -> size_t { return sum+samples.size(); });

        allCollectedSamplesCount += numValidSamples;
        if (allCollectedSamplesCount == 0) {
            Log(EInfo, "No samples found during training.");
        }

        if (numValidSamples < minSamplesToStartFitting) {
            Log(EInfo, "skipping fit due to insufficient sample data (got %zu/%d valid samples).", numValidSamples, minSamplesToStartFitting);
            return;
        }

        m_timer->reset();
        // flatten the samples vector

        std::vector<pmm_focal::WeightedSample> iterationSamples;
        // reserve to prevent bad alloc
        iterationSamples.reserve(std::accumulate(
            perThreadSamples.begin(), perThreadSamples.end(), 0ul,
            [](size_t acc, const IterableBlockedVector<pmm_focal::WeightedSample>& vec) {
                return acc + vec.size();
            })
        );
        for (auto& vec : perThreadSamples) {
            iterationSamples.insert(iterationSamples.end(), vec.begin(), vec.end());
            vec.clear();
        }

        Log(EInfo, "Got %i non-zero samples", iterationSamples.size());

        // FIRST APPROACH - chunks outside
        // size_t chunkSize = 1e6; // processing 1M samples at once - prevents bad alloc
        // for (size_t i = 0; i < iterationSamples.size(); i += chunkSize) {
        //     size_t end = std::min(i + chunkSize, iterationSamples.size());
        //     std::vector<pmm_focal::WeightedSample> chunk(iterationSamples.begin() + i, iterationSamples.begin() + end);
        //     m_gmm.processBatch(chunk);
        // }

        // second approach - chunks inside
        bool shouldTerminateEarly = m_gmm.processBatchParallel(iterationSamples);
        training = !shouldTerminateEarly;

        const Float convThreshold = m_octree.sumDensities();
        const Float divThreshold = m_octreeDiverging.sumDensities();
        const Float threshold = convThreshold + divThreshold;

        m_octree.build(threshold);
        m_octreeDiverging.build(threshold);

        divergeProbability = divThreshold / (divThreshold + convThreshold);
        m_gmm.setDivergeProbability(divergeProbability);
        Log(EInfo, "diverge probability: %.3f\n", divergeProbability);

        const Float postprocessingTime = m_timer->stop();
        Log(EInfo, "iteration postprocessing time: %s", timeString(postprocessingTime, true).c_str());
        m_timer->reset();
        Log(EInfo, m_gmm.toString().c_str());
        Log(EInfo, m_octree.toString().c_str());
        Log(EDebug, m_octree.toStringVerbose().c_str());

        Log(EInfo, m_octreeDiverging.toString().c_str());
        Log(EDebug, m_octreeDiverging.toStringVerbose().c_str());
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
            Log(EInfo, "Remaining rendering time: %d", (m_renderMaxSeconds - totalRenderTime));
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
        Float& gmmPdf,
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
            gmmPdf = 0;
            return result;
        }

        Spectrum result;
        mitsuba::Vector dir;
        bool isDiverging = false;
        bool isGuided = false;

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
                return result / bsdfSamplingFraction; // woPdf and bsdfPdf set - it's fine
            }

            result *= bsdfPdf;
            dir = bRec.its.toWorld(bRec.wo);
            // Log(EInfo, "bsdf (before /wo) %f", result);

        } else {
            // sample guiding distribution
            ++samplesFromGMMCount; // add to statistics
            isGuided = true;

            sample.x = (sample.x - bsdfSamplingFraction) / (1 - bsdfSamplingFraction);

            Eigen::VectorXd gmmSample = m_gmm.sample(rRec);
            mitsuba::Point endPoint(gmmSample[0], gmmSample[1], gmmSample[2]);
            dir = endPoint - bRec.its.p;
            bRec.wo = normalize(dir);
            bRec.wo = bRec.its.toLocal(bRec.wo);

            isDiverging = sample.x < divergeProbability;
            if (isDiverging) {
                ++samplesFromDiverging;
                bRec.wo *= -1;
            } else {
                ++samplesFromConverging;
            }

            bRec.eta = 1; // eta is Relative index of refraction in the sampled direction. Refractive index determines how much the path of light is bent, or refracted, when entering a material.
            bRec.sampledType = BSDF::EDiffuse;

            result = bsdf->eval(bRec);

            if (result.isZero()) {
                // invalid (aka zero contribution) direction
                woPdf = bsdfPdf = gmmPdf = 0;
                return result;
            }
        }

        pdfMat(woPdf, bsdfPdf, gmmPdf, bsdfSamplingFraction, bsdf, bRec, rRec.its.p, dir, isDiverging);
        if (woPdf == 0) {
            return Spectrum{0.0f};
        }

        result /= woPdf;
        // if (result.average() > 1)
        //     Log(EInfo, ("isGuided: %d, result: " + result.toString() + " woPdf: %f, bsdfPdf: %f, gmmPdf: %f").c_str(), isGuided, woPdf, bsdfPdf, gmmPdf);

        return result;
    }

    void pdfMat(
        Float& woPdf,
        Float& bsdfPdf,
        Float& gmmPdf,
        Float bsdfSamplingFraction,
        const BSDF* bsdf,
        const BSDFSamplingRecord& bRec,
        const Point &origin,
        mitsuba::Vector dir,
        bool isDiverging
    ) const {
        gmmPdf = 0.0f;

        auto type = bsdf->getType();
        if ((type & BSDF::EDelta) == (type & BSDF::EAll) || !shouldUseGuiding // EDelta = scattering into a discrete set of directions; Eall = any kind of scattering
        ) {
            woPdf = bsdfPdf = bsdf->pdf(bRec);
            return;
        }

        bsdfPdf = bsdf->pdf(bRec);
        assert(std::isfinite(bsdfPdf));

        gmmPdf = isDiverging ? m_octreeDiverging.splatPdf(origin, dir, m_gmm) : m_octree.splatPdf(origin, dir, m_gmm);
        assert(std::isfinite(gmmPdf));

        // MIS
        woPdf = bsdfSamplingFraction * bsdfPdf + (1 - bsdfSamplingFraction) * gmmPdf;
        woPdf = std::max(woPdf, 1e-6f); // clamping for numerical stability
        // Log(EInfo, "bsdfPdf: %f, gmmPdf: %f, woPdf: %f", bsdfPdf, gmmPdf, woPdf);
    }

    Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const {
        RayDifferential ray(r);
        Spectrum Li(0.0f);

        bool scattered = false;

        /* Perform the first ray intersection (or ignore if the
           intersection has already been provided). */
        rRec.rayIntersect(ray);
        ray.mint = Epsilon;

        Spectrum throughput(1.0f);
        Float eta = 1.0f;

        static thread_local std::vector<IntersectionData>* intersectionData {nullptr};
        static thread_local IterableBlockedVector<pmm_focal::WeightedSample>* samples {nullptr};

        // init thread variables
        if (training && !threadLocalInitialized.get()) {
            const size_t threadId = maxThreadId.fetch_add(1, std::memory_order_relaxed);
            intersectionData = &perThreadIntersectionData.at(threadId);
            samples = &perThreadSamples.at(threadId);
            threadLocalInitialized.get() = true;
        }

        IntersectionData dummyID{rRec.its};

        while (rRec.depth <= maxDepth || maxDepth < 0) {
            if (training)
                intersectionData->emplace_back(rRec.its);
            IntersectionData& currentIntersectionData = training ? intersectionData->back() : dummyID;

            if (!rRec.its.isValid()) {
                /* If no intersection could be found, potentially return
                    radiance from a environment luminaire if it exists */
                if ((rRec.type & RadianceQueryRecord::EEmittedRadiance)
                    && (!m_hideEmitters || scattered)) {
                    const Spectrum envmapRadiance = rRec.scene->evalEnvironment(ray);
                    Li += throughput * envmapRadiance;
                    if (Li.average() > 1e4)
                        Log(EInfo, ("1 " + Li.toString() + " " + throughput.toString()).c_str());
                    currentIntersectionData.emission = ContributionAndThroughput{envmapRadiance, Spectrum{1.0f}};
                }
                break;
            }

            const BSDF *bsdf = rRec.its.getBSDF(ray);

            /* Possibly include emitted radiance if requested */
            if (rRec.its.isEmitter() && (rRec.type & RadianceQueryRecord::EEmittedRadiance)
                && (!m_hideEmitters || scattered)) {
                const Spectrum emittedRadiance = rRec.its.Le(-ray.d);
                Li += throughput * emittedRadiance;
                if (Li.average() > 1e4)
                    Log(EInfo, ("2 " + Li.toString() + " " + throughput.toString()).c_str());

                currentIntersectionData.emission.contribution = emittedRadiance;
            }


            /* Include radiance from a subsurface scattering model if requested */
            if (rRec.its.hasSubsurface() && (rRec.type & RadianceQueryRecord::ESubsurfaceRadiance)) {
                const Spectrum subsurfaceScatteredRadiance = rRec.its.LoSub(rRec.scene, rRec.sampler, -ray.d, rRec.depth);
                Li += throughput * subsurfaceScatteredRadiance;
                if (Li.average() > 1e4)
                    Log(EInfo, ("3 " + Li.toString() + " " + throughput.toString()).c_str());

                currentIntersectionData.emission.contribution += subsurfaceScatteredRadiance;
            }

            if ((rRec.depth >= maxDepth && maxDepth > 0)
                || (m_strictNormals && dot(ray.d, rRec.its.geoFrame.n)
                    * Frame::cosTheta(rRec.its.wi) >= 0)) {

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
            DirectSamplingRecord dRec(rRec.its);

            if (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance &&
                (bsdf->getType() & BSDF::ESmooth)) {
                Spectrum value = rRec.scene->sampleEmitterDirect(dRec, rRec.nextSample2D());
                if (!value.isZero()) {
                    // so this is almost never true for our scenes - no wonder, it's very difficult to hit the emitter
                    const Emitter *emitter = static_cast<const Emitter *>(dRec.object);

                    /* Allocate a record for querying the BSDF */
                    BSDFSamplingRecord bRec(rRec.its, rRec.its.toLocal(dRec.d), ERadiance);

                    /* Evaluate BSDF * cos(theta) */
                    const Spectrum bsdfVal = bsdf->eval(bRec);

                    /* Prevent light leaks due to the use of shading normals */
                    if (!bsdfVal.isZero() && (!m_strictNormals
                            || dot(rRec.its.geoFrame.n, dRec.d) * Frame::cosTheta(bRec.wo) > 0)) {

                        /* Calculate prob. of having generated that direction
                            using BSDF sampling */
                            // it both other solutions here is where pdf is calculated from guided distribution
                        Float bsdfPdf = (emitter->isOnSurface() && dRec.measure == ESolidAngle)
                            ? bsdf->pdf(bRec) : 0; // should this be also guided somehow?

                        /* Weight using the power heuristic */
                        Float weight = miWeight(dRec.pdf, bsdfPdf);

                        Li += throughput * value * bsdfVal * weight;
                        if (Li.average() > 1e4)
                            Log(EInfo, ("4 " + Li.toString() + " " + throughput.toString()).c_str());
                        // if (shouldUseGuiding) {
                        //     Log(EInfo, "Li after direct sampling %f", Li.average());
                        // Log(EInfo, ("throughput = " + throughput.toString() + " value = " + value.toString() + " bsdfVal = " + bsdfVal.toString() + " weight = %f").c_str(), weight);
                        // }
                        currentIntersectionData.neeDirectLight = ContributionAndThroughput{value, bsdfVal*weight};
                    }
                }
            }

            /* ==================================================================== */
            /*                            BSDF sampling                             */
            /* ==================================================================== */

            /* Sample BSDF * cos(theta) */
            Float bsdfPdf, woPdf, gmmPdf;
            Float bsdfSamplingFraction = bsdfMISFraction;

            BSDFSamplingRecord bRec(rRec.its, rRec.sampler, ERadiance);

            // here is where sampling takes place
            Spectrum bsdfWeight = sampleFromGMM(bsdf, bRec, woPdf, bsdfPdf, gmmPdf, bsdfSamplingFraction, rRec);
            currentIntersectionData.woPdf = woPdf;

            if (bsdfWeight.isZero()) // this is why woPdf and bsdfPdf doesn't matter here
                break;

            scattered |= bRec.sampledType != BSDF::ENull;

            /* Prevent light leaks due to the use of shading normals */
            const Vector wo = rRec.its.toWorld(bRec.wo);
            currentIntersectionData.sampledType = bRec.sampledType;

            ++rRec.depth;

            Float woDotGeoN = dot(rRec.its.geoFrame.n, wo);
            if (m_strictNormals && woDotGeoN * Frame::cosTheta(bRec.wo) <= 0)
                break;

            bool hitEmitter = false;
            Spectrum value;

            /* Trace a ray in this direction */
            ray = Ray(rRec.its.p, wo, ray.time);
            if (rRec.scene->rayIntersect(ray, rRec.its)) {
                /* Intersected something - check if it was a luminaire */
                if (rRec.its.isEmitter()) {
                    value = rRec.its.Le(-ray.d);
                    // Log(EInfo, ("hit luminaire, value = " + value.toString()).c_str());
                    dRec.setQuery(ray, rRec.its);
                    hitEmitter = true;
                }
            } else {
                /* Intersected nothing -- perhaps there is an environment map? */
                const Emitter *env = rRec.scene->getEnvironmentEmitter();

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

            // Log(EInfo, "bsdfWeight: %f, throughput: %f", bsdfWeight, throughput);
            throughput *= bsdfWeight;
            currentIntersectionData.throughputFactors = bsdfWeight;
            eta *= bRec.eta;

            // currentIntersectionData.endpointRoughness =  rRec.its.getBSDF(ray)->getRoughness(rRec.its, bRec.sampledComponent);

            /* If a luminaire was hit, estimate the local illumination and
                weight using the power heuristic */
            if (hitEmitter &&
                (rRec.type & RadianceQueryRecord::EDirectSurfaceRadiance)) {
                /* Compute the prob. of generating that direction using the
                    implemented direct illumination sampling technique */
                const Float lumPdf = (!(bRec.sampledType & BSDF::EDelta)) ?
                    rRec.scene->pdfEmitterDirect(dRec) : 0;
                Float weight = miWeight(woPdf, lumPdf);
                Li += throughput * value * weight;
                if (Li.average() > 1e4)
                    Log(EInfo, ("5 " + Li.toString() + " " + throughput.toString()).c_str());

                currentIntersectionData.bsdfDirectLight = ContributionAndThroughput{value, bsdfWeight*weight};
                currentIntersectionData.bsdfMiWeight = weight;
            }

            /* ==================================================================== */
            /*                         Indirect illumination                        */
            /* ==================================================================== */

            /* Set the recursive query type. Stop if no surface was hit by the
                BSDF sample or if indirect illumination was not requested */
            if (!rRec.its.isValid() || !(rRec.type & RadianceQueryRecord::EIndirectSurfaceRadiance))
                break;

            rRec.type = RadianceQueryRecord::ERadianceNoEmission;

            if (maxDepth >= 0 && rRec.depth > maxDepth) {
                /* Russian roulette: try to keep path weights equal to one,
                    while accounting for the solid angle compression at refractive
                    index boundaries. Stop with at least some probability to avoid
                    getting stuck (e.g. due to total internal reflection) */

                Float q = std::min(throughput.max() * eta * eta, (Float) 0.95f);
                if (rRec.nextSample1D() >= q) {
                    if (training && !currentIntersectionData.bsdfDirectLight.contribution.isZero()) {
                        Log(EInfo, "bsdfDirectLight contribution is not zero");
                        intersectionData->emplace_back(rRec.its);
                    }
                    break;
                }
                throughput /= q;
                currentIntersectionData.throughputFactors /= q;
                // Log(EInfo, "invQ: %f, throughput: %f", 1/q, throughput);
            }
        }

        /* Store statistics */
        avgPathLength.incrementBase();
        avgPathLength += rRec.depth;

        if (training) {
            // Log(EInfo, "There are %d intersection samples to go through", intersectionData->size());
            // generate samples by traversing the path back to front
            for (int i=intersectionData->size() - 1; i >= 0; --i) {
                const IntersectionData& currentIntersectionData = (*intersectionData)[i];

                Spectrum clampedLight {0.0f};

                // for debugging purposes:
                // Spectrum bsdfLight {0.0f};
                // Spectrum neeLight {0.0f};
                // Spectrum emissionLight {0.0f};


                const float maxThroughput = 10.0f;
                const float minPdf = 0.1f;

                for (size_t j=i+1; j<intersectionData->size(); ++j)
                {
                    clampedLight += (*intersectionData)[j].bsdfDirectLight.getClamped(maxThroughput);
                    clampedLight += (*intersectionData)[j].neeDirectLight.getClamped(maxThroughput);
                    clampedLight += (*intersectionData)[j].emission.getClamped(maxThroughput);
                    // for debugging purposes:
                    // bsdfLight += (*intersectionData)[j].bsdfDirectLight.getClamped(maxThroughput);
                    // neeLight += (*intersectionData)[j].neeDirectLight.getClamped(maxThroughput);
                    // emissionLight += (*intersectionData)[j].emission.getClamped(maxThroughput);
                }
                // clampedLight += currentIntersectionData.bsdfDirectLight.contribution * currentIntersectionData.bsdfMiWeight; -- no direct contributions



                if (clampedLight.isZero()) {
                    // could collect to statistics...
                    // do nothing
                } else {
                    float clampedPdf = std::max(minPdf, currentIntersectionData.woPdf);
                    // float weight = clampedLight.average() / clampedPdf;
                    // Log(EInfo, "clampedLight: %f, clampedPdf: %f, weight: %f", clampedLight.average(), clampedPdf, weight);
                    // Log(EInfo, "bsdfDirect: %f, neeDirect: %f, emitted: %f, rest: %f", bsdfLight.average(), neeLight.average(), emissionLight.average(), (currentIntersectionData.bsdfDirectLight.contribution*currentIntersectionData.bsdfMiWeight).average());
                    const BSDF *endpointBSDF = currentIntersectionData.its.getBSDF();
                    Float endpointRoughness = std::numeric_limits<Float>::infinity();
                    if (endpointBSDF) {
                        for (int comp = 0; comp < endpointBSDF->getComponentCount(); ++comp)
                            endpointRoughness = std::min(endpointRoughness, endpointBSDF->getRoughness(currentIntersectionData.its, comp));
                    } else {
                        Log(EInfo, "no endpoitnBSDF");
                    }

                    const bool endpointIsGlossy = endpointRoughness < 0.3f; // [Ruppert et al. 2020]
                    const Float splatDistance = endpointIsGlossy ?
                    std::numeric_limits<Float>::infinity() : /// virtual image possible, need to splat entire ray
                        currentIntersectionData.distance;
                    if (currentIntersectionData.woPdf == 0) {
                        continue;
                    }
                    m_octree.splat(ray.o, ray.d, splatDistance, clampedLight.average(), samples,  clampedPdf / (1 - divergeProbability));
                    if (endpointIsGlossy && divergeProbability != 0) {
                        m_octreeDiverging.splat(ray.o, ray.d, splatDistance, clampedLight.average(), samples,  clampedPdf / divergeProbability);
                    }
                }

                if (i == 0)
                    break;

                for (size_t j=i+1; j<intersectionData->size(); ++j)
                    (*intersectionData)[j] *= currentIntersectionData.throughputFactors;
            }
            intersectionData->clear();
        }

        if (Li.average() > 1e4)
            Log(EInfo, Li.toString().c_str());

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
        perThreadIntersectionData.clear();
        perThreadSamples.clear();
        // perThreadSampledPoints.clear();

        Log(EInfo, m_gmm.toString().c_str());
        // Log(EInfo, m_octree.toStringVerbose().c_str());
        // Log(EInfo, m_octreeDiverging.toStringVerbose().c_str());
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
