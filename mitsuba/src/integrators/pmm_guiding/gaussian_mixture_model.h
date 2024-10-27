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

#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#ifdef MITSUBA_LOGGING
#include <mitsuba/core/logger.h>
#endif

#include "distribution.h"

#define FAIL_ON_ZERO_CDF 0
#define USE_MAX_KEEP 0

namespace pmm {

template<typename T>
using alignedVector = std::vector<T, Eigen::aligned_allocator<T>>;

template<
    size_t t_dims,
    size_t t_components,
    typename Scalar_t,
    template<size_t, typename> class Component_t
>
class GaussianMixtureModel : public Distribution<t_dims, Scalar_t> {
public:
    using Scalar = Scalar_t;

    using Component = Component_t<t_dims, Scalar>;

    using Vectord = Eigen::Matrix<Scalar, t_dims, 1>; // todo: same in gaussian_compoennt - can be moved outside the class, into namespace
    using Matrixd = Eigen::Matrix<Scalar, t_dims, t_dims>;

    using Responsibilities = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, t_components, Eigen::Dynamic>;

    GaussianMixtureModel() :
        m_components(t_components),
        m_weights(t_components),
        m_cdf(t_components)
    {
#ifdef MITSUBA_LOGGING
        Log(EInfo, "Created GMM with %i components", t_components);
#endif
    }

    Vectord sample(const std::function<Scalar()>& rng) const {
        size_t component_i = sampleDiscreteCdf(std::begin(m_cdf), std::begin(m_cdf) + m_lastIdx, rng());
        return m_components[component_i].sample(rng);
    }

    Scalar surfacePdf(const Vectord& sample) const {
        Scalar pdfAccum = 0.f;
        for(size_t component_i = 0; component_i < m_lastIdx; ++component_i) {
            if(m_weights[component_i] != 0) {
                pdfAccum += m_weights[component_i] * m_components[component_i].pdf(sample);
            }
        }
        return m_normalization * pdfAccum / m_surfacesize_tegral;
    }

    Scalar surfacePdf(const Vectord& sample, Scalar heuristicPdf) {
        return (1 - m_heuristicWeight) * surfacePdf(sample) + m_heuristicWeight * heuristicPdf;
    }

    Scalar pdf(const Vectord& sample) const {
        Scalar pdfAccum = 0.f;
        for(size_t component_i = 0; component_i < m_lastIdx; ++component_i) {
            if(m_weights[component_i] != 0) {
                pdfAccum += m_weights[component_i] * m_components[component_i].pdf(sample);
            }
        }
        return pdfAccum;
    }

    pmm::alignedVector<Component>& components() { return m_components; }
    pmm::alignedVector<Scalar>& weights() { return m_weights; }

    void setMixtureThreshold(Scalar mixtureThreshold) { m_mixtureThreshold = mixtureThreshold; }

    size_t nComponents() const { return m_lastIdx; }
    void setNComponents(size_t lastIdx) {m_lastIdx = lastIdx;}

    Scalar heuristicWeight() const { return m_heuristicWeight; }
    void setHeuristicWeight(Scalar weight) { m_heuristicWeight = weight; }

    Scalar normalization() const { return m_normalization; }
    void setNormalization(Scalar normalization) { m_normalization = normalization; }

    Scalar surfacesize_tegral() const { return m_surfacesize_tegral; }
    void setSurfacesize_tegral(Scalar surfacesize_tegral) { m_surfacesize_tegral = surfacesize_tegral; }

    Scalar surfaceArea() const { return m_surfaceArea; }
    void setSurfaceArea(Scalar surfaceArea) { m_surfaceArea = surfaceArea; }

    Scalar modelError() const { return m_modelError; }
    void setModelError(Scalar modelError) { m_modelError = modelError; }

    void posteriorAndLog(
        const Vectord& sample,
        bool useHeuristic,
        Scalar heuristicPdf,
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& pdf,
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& posterior,
        Eigen::Matrix<Scalar, Component::t_josize_tTangentDims, Eigen::Dynamic>& tangentVectors,
        Scalar& heuristicPosterior
    ) const {
        heuristicPosterior = 0.f;
        typename Component::Josize_tTangentVectord tangent;
        for(size_t component_i = 0; component_i < m_lastIdx; ++component_i) {
            pdf(component_i) = m_components[component_i].pdfAndLog(sample, tangent);
            if(!std::isfinite(pdf(component_i))) {
                std::cerr << "Infinite pdf: " << pdf(component_i) << ".\n";
            }
            posterior(component_i) = m_weights[component_i] * pdf(component_i);
            tangentVectors.col(component_i) = tangent;
        }

        Scalar sum = posterior.sum();
        if(useHeuristic) {
            sum = (1 - m_heuristicWeight) * sum + m_heuristicWeight * heuristicPdf;
        }

        const Scalar invSum = 1 / sum;
        if(std::isfinite(invSum)) {
            posterior *= invSum;
            if(useHeuristic) {
                posterior *= (1.f - m_heuristicWeight);
                heuristicPosterior = m_heuristicWeight * heuristicPdf * invSum;
                pdf = m_heuristicWeight * heuristicPdf + (1.f - m_heuristicWeight) * pdf.array();
            }
        } else {
            std::cerr << "Infinite or nan posterior sum = 1.f / " << sum << ", " << pdf.sum() << "\n";
            Scalar weightSum = 0.f;
            for(size_t component_i = 0; component_i < m_lastIdx; ++component_i) {
                weightSum += m_weights[component_i];
            }
            posterior.setZero();
            pdf.setZero();
            heuristicPosterior = 0.f;
        }
    }

    void posterior(
        const Vectord& sample,
        bool useHeuristic,
        Scalar heuristicPdf,
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& posterior,
        Scalar& heuristicPosterior
    ) const {
        heuristicPosterior = 0.f;
        for(size_t component_i = 0; component_i < m_lastIdx; ++component_i) {
            Scalar pdf = m_components[component_i].pdf(sample);
            posterior(component_i) = m_weights[component_i] * pdf;
        }

        Scalar sum = posterior.sum();
        if(useHeuristic) {
            sum = (1 - m_heuristicWeight) * sum + m_heuristicWeight * heuristicPdf;
        }

        const Scalar invSum = 1 / sum;
        if(std::isfinite(invSum)) {
            posterior *= invSum;
            if(useHeuristic) {
                posterior *= (1.f - m_heuristicWeight);
                heuristicPosterior = m_heuristicWeight * heuristicPdf * invSum;
            }
        } else {
            std::cerr << "Infinite or nan posterior sum = 1.f / " << sum << "\n";
            posterior.setZero();
            heuristicPosterior = 0.f;
        }
    }

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

    template<typename TInIterator>
    Scalar calculateResponsibilitesForSample(TInIterator& sample, size_t i, Responsibilities& responsibilities) {
        Scalar partialSumResponsibility{0.0f};

        for (size_t k = 0; k < m_lastIdx; ++k) {
            Scalar pdfValue = m_components[k].pdf(sample);
            responsibilities(k, i) = m_weights[k] * pdfValue;
            partialSumResponsibility += responsibilities[k];
        }

        float mixturePDF = sum(partialSumResponsibility);
        const float invMixturePDF = 1.0f/mixturePDF;

        for(size_t k = 0; k < m_lastIdx; ++k)
            responsibilities(k, i) *= invMixturePDF;

#ifdef MITSUBA_LOGGING
        Log(EInfo, "Finished calculating responsibilities for %i sample.", i);
#endif
        return partialSumResponsibility;
    }

    // todo:
    // integrate it further to the system
    // start sample collection -- octree

    using SampleVector = Eigen::Matrix<Scalar, t_dims, Eigen::Dynamic>;

    Scalar fit(SampleVector& samples) {
#ifdef MITSUBA_LOGGING
        Log(EInfo, "Begining to fit samples.");
#endif
        size_t numSamples = samples.size();
        Scalar logLikelihood = 0;
        Responsibilities responsibilities(t_components, numSamples);

#ifdef MITSUBA_LOGGING
        Log(EInfo, "Performing E-step of EM algorithm.");
#endif
        for (size_t col = 0; col < samples.cols(); ++col) {
            const Vectord sample = samples.col(col);

            Scalar partialLogLikelihood = responsibilities.setZero();
            calculateResponsibilitesForSample(sample, col, responsibilities); // todo: refactoring idea: move to a class, similarly to vmm

            logLikelihood += std::log(partialLogLikelihood);
        }

#ifdef MITSUBA_LOGGING
        Log(EInfo, "Performing M-step of EM algorithm.");
#endif
        for (size_t k = 0; k < m_components; ++k) { // M-step -- todo: modularize
            Scalar responsibilitySum = responsibilities.row(k).sum();
            m_weights[k] = responsibilitySum / numSamples;

            Vectord newMean = Vectord::Zero();
            for (size_t col = 0; col < samples.cols(); ++col) {
                const Vectord sample = samples.col(col);
                newMean += responsibilities(k, col) * sample;
            }
            newMean /= responsibilitySum;
            m_components[k].setMean(newMean);

            Matrixd newCovariance = Matrixd::Zero();
            for (size_t col = 0; col < samples.cols(); ++col) {
                const Vectord sample = samples.col(col);
                Vectord diff = sample - newMean;
                newCovariance += responsibilities(k, col) * (diff * diff.transpose());
            }
            newCovariance /= responsibilitySum;
            m_components[k].setCovariance(newCovariance);
        }

#ifdef MITSUBA_LOGGING
        Log(EInfo, "Finished fitting samples.");
#endif
        // todo: check for convergence outside of function (?)
        return logLikelihood;
    }


private:
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const size_t version) {
        ar  & m_components;
        ar  & m_weights;
        ar  & m_lastIdx;
        ar  & m_heuristicWeight;
        if(version > 0) {
            ar  & m_normalization;
        }
    }

    pmm::alignedVector<Component> m_components;
    pmm::alignedVector<Scalar> m_weights;
    pmm::alignedVector<Scalar> m_cdf;
    size_t m_lastIdx = 0;

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