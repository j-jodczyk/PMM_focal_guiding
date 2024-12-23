/* Adapted from Anna Dodik sdmm-mitsuba project (https://github.com/anadodik/sdmm-mitsuba/tree/main/mitsuba/src/integrators/dmm) */

#ifndef __MIXTURE_MODEL_H
#define __MIXTURE_MODEL_H

#include <vector>
#include <functional>
#include <numeric>
#include <cassert>
#include <atomic>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <deque>
#include <mutex>

#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

// #define MITSUBA_LOGGING = 1

#ifdef MITSUBA_LOGGING
#include <mitsuba/core/logger.h>
#endif

#include "distribution.h"
#include "util.h"

#define FAIL_ON_ZERO_CDF 0
#define USE_MAX_KEEP 0

namespace pmm_focal {

template<typename T>
using alignedVector = std::vector<T, Eigen::aligned_allocator<T>>;

template<
    size_t t_dims,
    size_t t_components,
    typename Scalar_t,
    template<size_t, typename> class Component_t,
    typename Env
>
class GaussianMixtureModel : public Distribution<t_dims, Scalar_t> {
public:
    using Scalar = Scalar_t;
    using AABB = typename Env::AABB;

    using Component = Component_t<t_dims, Scalar>;

    using Vectord = Eigen::Matrix<Scalar, t_dims, 1>; // todo: same in gaussian_compoennt - can be moved outside the class, into namespace
    using Matrixd = Eigen::Matrix<Scalar, t_dims, t_dims>;

    using Responsibilities = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, t_components, Eigen::Dynamic>;
    using SampleVector = std::deque<Vectord>;

    GaussianMixtureModel() :
        m_components(t_components),
        m_weights(t_components),
        m_cdf(t_components),
        m_samples(),
        m_paramCount(getParamCount(t_dims, t_components))
    {
#ifdef MITSUBA_LOGGING
        SLog(mitsuba::EInfo, "Created GMM with %i components", t_components);
#endif
    }

    std::string toString() const
    {
        std::ostringstream oss;
        oss << "GMM[" << std::endl;
        for(size_t k = 0; k < t_components; k++){
            oss << "[" << k << "]: " << "weight = " << m_weights[k] << " " << m_components[k].toString() <<std::endl;
        }
        oss << "]";
        return oss.str();
    }

    Scalar pdf(const Vectord& sample) const {
        Scalar pdfAccum = 0.f;
        for(size_t component_i = 0; component_i < m_numComponents; ++component_i) {
            if(m_weights[component_i] != 0) {
                pdfAccum += m_weights[component_i] * m_components[component_i].pdf(sample);
            }
        }
        return pdfAccum;
    }

    pmm_focal::alignedVector<Component>& components() { return m_components; }
    pmm_focal::alignedVector<Scalar>& weights() { return m_weights; }

    void save(const std::string& filename) const {
        // make an archive
        std::ofstream ofs(filename.c_str());
        boost::archive::binary_oarchive oa(ofs);
        oa << BOOST_SERIALIZATION_NVP(*this);
    }

    void load(const std::string& filename) {
        // open the archive
        std::ifstream ifs(filename.c_str());
        boost::archive::binary_iarchive ia(ifs);
        ia >> BOOST_SERIALIZATION_NVP(*this);
    }

    Vectord getComponentMean(size_t k) {
        return m_components[k].getMean();
    }

    Matrixd getComponentCovariance(size_t k) {
        return m_components[k].getCovariance();
    }

    size_t getParamCount(size_t dimentions, size_t components) {
        /* Mean vector has dim elements, Covariance 0.5 * dim * (dim + 1), and there are t_components */
        return 0.5 * components * ( dimentions * dimentions + 3 * dimentions + 2 );
    }

    /** Written primarily for testing program, might be better to later initialize with the first couple of samples */
    void initialize(SampleVector& samples) {
        m_samples = samples;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> sampleDist(0, samples.size() - 1);
        std::uniform_real_distribution<Scalar_t> weightDist(0.1, 1.0);

        Scalar_t weight_sum = 0.0;

        for (size_t k = 0; k < t_components; ++k) {
            m_components[k].setMean(samples[sampleDist(gen)]);
            m_components[k].setCovariance(Matrixd::Identity() * std::numeric_limits<Scalar>::epsilon());

            m_weights[k] = weightDist(gen);
            weight_sum += m_weights[k];
        }

        for (auto& weight : m_weights) {
            weight /= weight_sum;
        }
    }

    void initialize(const AABB& aabb) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<Scalar_t> weightDist(0.1, 1.0);

        auto min = aabb.min; ///< Component-wise minimum
        auto max = aabb.max; ///< Component-wise maximum

        for (size_t k = 0; k < t_components; ++k) {
            Vectord mean;
            for (size_t i=0; i < t_dims; i++) {
                std::uniform_real_distribution<Scalar_t> pointDist(min[i], max[i]);
                mean[i] = pointDist(gen);
            }

            m_components[k].setMean(mean);
            m_components[k].setCovariance(Matrixd::Identity() * weightDist(gen));

            m_weights[k] = 1.0 / t_components; // starting with equal weights for all
        }
    }

    template<typename TInIterator>
    Scalar calculateResponsibilitesForSample(TInIterator& sample, size_t i, Responsibilities& responsibilities) {
        Scalar partialSumResponsibility{0.0f};

        for (size_t k = 0; k < m_numComponents; ++k) {
            Scalar pdfValue = m_components[k].pdf(sample);
            Scalar responsibility = m_weights[k] * pdfValue;
            responsibilities(k, i) = responsibility;
            partialSumResponsibility += responsibilities(k, i);
        }

        // float mixturePDF = sum(partialSumResponsibility); -- uncomment for SIMD (to be implemented)
        float mixturePDF = partialSumResponsibility;
        const Scalar invMixturePDF = partialSumResponsibility > std::numeric_limits<Scalar>::epsilon()
                                    ? 1.0f / partialSumResponsibility
                                    : 1.0f / t_components;

        for(size_t k = 0; k < m_numComponents; ++k)
            responsibilities(k, i) *= invMixturePDF;

#ifdef MITSUBA_LOGGING
        SLog(mitsuba::EDebug, "Finished calculating responsibilities for %i sample.", i);
#endif
        return partialSumResponsibility;
    }

    std::string samplesToString(const SampleVector& samples) {
        std::ostringstream oss;
        oss << "[";
        for (auto& sample : samples) {
            oss << "[";
            for (size_t i = 0; i < t_dims; ++i) {
                oss << sample[i];
                if (i < t_dims - 1) {
                    oss << ", ";
                }
            }
            oss << "]";
        }

        oss << "]";
        return oss.str();
    }

    Scalar fit(SampleVector& newSamples) {
#ifdef MITSUBA_LOGGING
        SLog(mitsuba::EDebug, "Begining to fit samples.");
        SLog(mitsuba::EInfo, samplesToString(newSamples).c_str());
#endif
        size_t numSamples = newSamples.size();

        Scalar logLikelihood = 0;
        Responsibilities responsibilities(t_components, numSamples);
        responsibilities.setZero();

#ifdef MITSUBA_LOGGING
        SLog(mitsuba::EDebug, "Performing E-step of EM algorithm.");
#endif
        for (size_t col = 0; col < numSamples; ++col) {
            const Vectord sample = newSamples[col];

            Scalar partialLogLikelihood = calculateResponsibilitesForSample(sample, col, responsibilities); // todo: refactoring idea: move to a class, similarly to vmm
            partialLogLikelihood = std::max(partialLogLikelihood, std::numeric_limits<Scalar>::epsilon());

            logLikelihood += std::log(partialLogLikelihood);
        }

#ifdef MITSUBA_LOGGING
        SLog(mitsuba::EDebug, "Performing M-step of EM algorithm.");
#endif

        Scalar weightSum = 0.0;
        for (size_t k = 0; k < t_components; ++k) { // M-step -- todo: modularize
            Scalar responsibilitySum = responsibilities.row(k).sum();

            // std::cout << "responsibilitySum: " << responsibilitySum << std::endl;

            m_weights[k] = responsibilitySum / numSamples;

            // if(!std::isfinite(m_weights[k])) {
            //     SLog(mitsuba::EInfo, "Responsibility sum = %f", responsibilitySum);
            //     SLog(mitsuba::EInfo, this->toString().c_str());
            // }

            weightSum += responsibilitySum / numSamples;

            Vectord newMean = Vectord::Zero();
            for (size_t col = 0; col < numSamples; ++col) {
                const Vectord sample = newSamples[col];
                newMean += responsibilities(k, col) * sample;
            }
            newMean /= responsibilitySum;
            // SLog(mitsuba::EInfo, "NewMean is %d", newMean);
            m_components[k].setMean(newMean);

            Matrixd newCovariance = Matrixd::Zero();
            for (size_t col = 0; col < numSamples; ++col) {
                const Vectord sample = newSamples[col];
                Vectord diff = sample - newMean;
                newCovariance += responsibilities(k, col) * (diff * diff.transpose());
            }
            newCovariance /= responsibilitySum;
            m_components[k].setCovariance(newCovariance);
        }

        if (weightSum == 1.0) {
            // weights sum up to 1 - can return early
            return logLikelihood;
        }
        // weights don't sum up to 1 - normalize
        for (size_t k = 0; k < t_components; ++k)
            m_weights[k] = m_weights[k] / weightSum;

#ifdef MITSUBA_LOGGING
        SLog(mitsuba::EDebug, "Finished fitting samples.");
#endif
        // todo: check for convergence outside of function (?)
        return logLikelihood;
    }

    /* Vestion with repetition until max iterations */
    void fit(SampleVector& samples, int maxIterations, Scalar tolerance = 1e-6) {
        Scalar logLikelihood = 0;
        Scalar previousLogLikelihood = -std::numeric_limits<Scalar>::infinity();
        for (int iter = 0; iter < maxIterations; ++iter) {
            logLikelihood = fit(samples);
            if (std::abs(logLikelihood - previousLogLikelihood) < tolerance)
                break;
            previousLogLikelihood = logLikelihood;
        }
    }

    Vectord sample() const {
        // first we have to sample the component according to weights
        std::vector<Scalar> cdf(m_weights.size(), 0.0f);
        cdf[0] = m_weights[0];

        for (size_t i = 1; i < m_weights.size(); i++) {
            cdf[i] = cdf[i - 1] + m_weights[i];
        }

        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        std::mt19937 rng(std::random_device{}());
        float randomValue = dist(rng);

        Component component;

        for (size_t i = 0; i < cdf.size(); ++i) {
            if (randomValue <= cdf[i]) {
                component = m_components[i];
                break;
            }
        }

        // now we sample from the component
        return component.sample();
    }


private:
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const size_t version) {
        ar  & m_components;
        ar  & m_weights;
        ar  & m_numComponents;
        ar  & m_heuristicWeight;
        if(version > 0) {
            ar  & m_normalization;
        }
    }

    pmm_focal::alignedVector<Component> m_components;
    pmm_focal::alignedVector<Scalar> m_weights;
    pmm_focal::alignedVector<Scalar> m_cdf;
    std::mutex m_mutex;
    size_t m_numComponents = t_components;
    SampleVector m_samples;
    size_t m_paramCount;

    Scalar m_heuristicWeight = 0.5f;
    Scalar m_mixtureThreshold = 0.f; // 1.f / (Scalar) t_components;
    Scalar m_normalization = 1.f;
    Scalar m_surfacesize_tegral = 1.f;
    Scalar m_surfaceArea = 0.f;
    Scalar m_modelError = 0.f;
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}

#endif /* __MIXTURE_MODEL_H */