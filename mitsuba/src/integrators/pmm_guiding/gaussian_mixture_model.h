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

#include <eigen3/Eigen/Dense>

#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include "gaussian_component.h"

// #define MITSUBA_LOGGING = 1

#ifdef MITSUBA_LOGGING
#include <mitsuba/core/logger.h>
#endif

#include "distribution.h"
#include "util.h"

#define FAIL_ON_ZERO_CDF 0
#define USE_MAX_KEEP 0

namespace pmm_focal {

template<
    typename Scalar_t,
    typename Env
>
class GaussianMixtureModel : public Distribution<Scalar_t> {
public:
    using Scalar = Scalar_t;
    using AABB = typename Env::AABB;

    std::vector<GaussianComponent> components;
    double learningRate;
    double chiSquaredThreshold;

    GaussianMixtureModel() {
        initialVariance_ = 0;
        learningRate = 0;
        chiSquaredThreshold = 0;
    }

    GaussianMixtureModel(double initialVariance, double learningRate, double chiSquaredThreshold)
        : learningRate(learningRate), chiSquaredThreshold(chiSquaredThreshold) {
        initialVariance_ = initialVariance;
#ifdef MITSUBA_LOGGING
        SLog(mitsuba::EInfo, "Created GMM with %i components", t_components);
#endif
    }

    void setInitialVariance( double variance ) { initialVariance_ = variance; }

    void addComponent(const Eigen::VectorXd& x) {
        int dimension = x.size();
        GaussianComponent newComponent(dimension, initialVariance_);
        newComponent.setMean(x);
        components.push_back(newComponent);
    }

    void updateComponent(GaussianComponent& component, const Eigen::VectorXd& x) {
        Eigen::VectorXd diff = x - component.getMean();
        double responsibility = this->computeResponsibility(component, x);

        // Update weight
        auto newWeight = (1 - learningRate) * component.getWeight() + learningRate * responsibility;
        component.setWeight(newWeight);

        // Update mean
        Eigen::VectorXd deltaMean = learningRate * responsibility * diff;
        auto newMean = component.getMean() + deltaMean;
        component.setMean(newMean);

        // Update precision matrix using Sherman-Morrison formula
        Eigen::MatrixXd outerProd = diff * diff.transpose();
        Eigen::MatrixXd adjustment = (component.getPrecisionMatrix() * outerProd * component.getPrecisionMatrix()) /
                              (1.0 + (diff.transpose() * component.getPrecisionMatrix() * diff)(0));
        auto newPrecisionMatrix = component.getPrecisionMatrix() - adjustment;
        component.setPrecisionMatrix(newPrecisionMatrix);

        // Update determinant using rank-one update formula
        double adjustmentFactor = 1.0 - (diff.transpose() * component.getPrecisionMatrix() * diff)(0);
        double newDeterminant = component.getDeterminant() * adjustmentFactor;
        component.setDeterminant(newDeterminant);
    }

    void process(const Eigen::VectorXd& x) {
        double minMahalanobisDistance = std::numeric_limits<double>::max();
        GaussianComponent* bestComponent = nullptr;

        for (auto& component : components) {
            double distance = computeMahalanobisDistance(component, x);
            if (distance < minMahalanobisDistance) {
                minMahalanobisDistance = distance;
                bestComponent = &component;
            }
        }

        if (bestComponent && minMahalanobisDistance < chiSquaredThreshold) {
            updateComponent(*bestComponent, x);
        } else {
            addComponent(x);
        }
    }

    // TODO: fix
    std::string toString() const
    {
        std::ostringstream oss;
        oss << "GMM";
        return oss.str();
    }

    double pdf(const Eigen::VectorXd& x) {
        double totalPdf = 0.0;
        for (const auto& component : components) {
            double density = computeGaussianDensity(component, x);
            totalPdf += component.getWeight() * density;
        }
        return totalPdf;
    }

private:
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const size_t version) {
        ar  & components;
    }

    double initialVariance_;
    double computeMahalanobisDistance(const GaussianComponent& component, const Eigen::VectorXd& x) {
        Eigen::VectorXd diff = x - component.getMean();
        return diff.transpose() * component.getPrecisionMatrix() * diff;
    }

    double computeGaussianDensity(const GaussianComponent& component, const Eigen::VectorXd& x) {
        Eigen::VectorXd diff = x - component.getMean();
        double exponent = -0.5 * diff.transpose() * component.getPrecisionMatrix() * diff;
        double normalization = pow(2 * M_PI, -x.size() / 2.0) * sqrt(component.getDeterminant());
        return normalization * exp(exponent);
    }

    double computeResponsibility(const GaussianComponent& component, const Eigen::VectorXd& x) {
        double exponent = -0.5 * computeMahalanobisDistance(component, x);
        double normalization = pow(2 * M_PI, -x.size() / 2.0) * sqrt(component.getDeterminant());
        return exp(exponent) / normalization;
    }

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}

#endif /* __MIXTURE_MODEL_H */