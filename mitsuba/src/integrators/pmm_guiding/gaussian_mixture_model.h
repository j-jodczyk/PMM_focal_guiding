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

#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#define MITSUBA_LOGGING = 1

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
    using SampleVector = Eigen::Matrix<Scalar, t_dims, Eigen::Dynamic>;

    GaussianMixtureModel() :
        m_components(t_components),
        m_weights(t_components),
        m_cdf(t_components),
        m_samples()
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

    /** Written primarily for testing program, might be better to later initialize with the first couple of samples */
    void initialize(SampleVector& samples) {
        m_samples = samples;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> sampleDist(0, samples.cols() - 1);
        std::uniform_real_distribution<Scalar_t> weightDist(0.1, 1.0);

        Scalar_t weight_sum = 0.0;

        for (size_t k = 0; k < t_components; ++k) {
            m_components[k].setMean(samples.col(sampleDist(gen)));
            m_components[k].setCovariance(Matrixd::Identity() * weightDist(gen));

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

        Scalar_t weight_sum = 0.0;

        for (size_t k = 0; k < t_components; ++k) {
            Vectord mean;
            for (size_t i=0; i < t_dims; i++) {
                std::uniform_real_distribution<Scalar_t> pointDist(min[i], max[i]);
                mean[i] = pointDist(gen);
            }

            m_components[k].setMean(mean);
            m_components[k].setCovariance(Matrixd::Identity() * weightDist(gen));

            m_weights[k] = weightDist(gen);
            weight_sum += m_weights[k];
        }

        for (auto& weight : m_weights) {
            weight /= weight_sum;
        }
    }

    template<typename TInIterator>
    Scalar calculateResponsibilitesForSample(TInIterator& sample, size_t i, Responsibilities& responsibilities) {
        Scalar partialSumResponsibility{0.0f};

        for (size_t k = 0; k < m_numComponents; ++k) {
            Scalar pdfValue = m_components[k].pdf(sample);
            responsibilities(k, i) = m_weights[k] * pdfValue;
            partialSumResponsibility += responsibilities(k, i);
        }

        // float mixturePDF = sum(partialSumResponsibility); -- uncomment for SIMD (to be implemented)
        float mixturePDF = partialSumResponsibility;
        const float invMixturePDF = 1.0f/mixturePDF;

        for(size_t k = 0; k < m_numComponents; ++k)
            responsibilities(k, i) *= invMixturePDF;

#ifdef MITSUBA_LOGGING
        SLog(mitsuba::EDebug, "Finished calculating responsibilities for %i sample.", i);
#endif
        return partialSumResponsibility;
    }

    // todo:
    // integrate it further to the system

    Scalar fit(SampleVector& newSamples) {
#ifdef MITSUBA_LOGGING
        SLog(mitsuba::EInfo, "Begining to fit samples.");
#endif
        addSamples(newSamples);
#ifdef MITSUBA_LOGGING
        SLog(mitsuba::EInfo, "Added new samples.");
#endif
        size_t numSamples = m_samples.cols();

        Scalar logLikelihood = 0;
        Responsibilities responsibilities(t_components, numSamples);
        responsibilities.setZero();

#ifdef MITSUBA_LOGGING
        SLog(mitsuba::EInfo, "Performing E-step of EM algorithm.");
#endif
        for (size_t col = 0; col < numSamples; ++col) {
            const Vectord sample = m_samples.col(col);

            Scalar partialLogLikelihood = calculateResponsibilitesForSample(sample, col, responsibilities); // todo: refactoring idea: move to a class, similarly to vmm

            logLikelihood += std::log(partialLogLikelihood);
        }

#ifdef MITSUBA_LOGGING
        SLog(mitsuba::EInfo, "Performing M-step of EM algorithm.");
#endif
        SLog(mitsuba::EInfo, "Responsibilities size: row: %i, col: %i.\nNumber of samples: %i.\nm_samples: rows: %i, cols: %i.\n",
            responsibilities.rows(), responsibilities.cols(),
            numSamples,
            m_samples.rows(), m_samples.cols()
        );

        for (size_t k = 0; k < t_components; ++k) { // M-step -- todo: modularize
            Scalar responsibilitySum = responsibilities.row(k).sum();
            m_weights[k] = responsibilitySum / numSamples;

            Vectord newMean = Vectord::Zero();
            for (size_t col = 0; col < numSamples; ++col) {
                const Vectord sample = m_samples.col(col);
                newMean += responsibilities(k, col) * sample;
            }
            newMean /= responsibilitySum;
            m_components[k].setMean(newMean);

            Matrixd newCovariance = Matrixd::Zero();
            for (size_t col = 0; col < numSamples; ++col) {
                const Vectord sample = m_samples.col(col);
                Vectord diff = sample - newMean;
                newCovariance += responsibilities(k, col) * (diff * diff.transpose());
            }
            newCovariance /= responsibilitySum;
            m_components[k].setCovariance(newCovariance);
        }

#ifdef MITSUBA_LOGGING
        SLog(mitsuba::EInfo, "Finished fitting samples.");
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

    void addSamples(SampleVector &newSamples) {
        SLog(mitsuba::EInfo, "m_samples cols: %i, rows: %i", m_samples.cols(), m_samples.rows());
        SLog(mitsuba::EInfo, "newSamples cols: %i, rows: %i", newSamples.cols(), newSamples.rows());
        if (m_samples.cols() == 0) { // no previous samples at this point
            m_samples = newSamples;
        } else {
            assert(m_samples.rows() == newSamples.rows() && "Row dimensions must match to add as new columns");
            m_samples.conservativeResize(Eigen::NoChange, m_samples.cols() + newSamples.cols());
            m_samples.rightCols(newSamples.cols()) = newSamples;
        }
    }

    pmm_focal::alignedVector<Component> m_components;
    pmm_focal::alignedVector<Scalar> m_weights;
    pmm_focal::alignedVector<Scalar> m_cdf;
    size_t m_numComponents = t_components;
    SampleVector m_samples;

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