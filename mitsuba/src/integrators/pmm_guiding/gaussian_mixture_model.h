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
private:
    using Scalar = Scalar_t;
    using AABB = typename Env::AABB;

    std::vector<GaussianComponent> components;
    double alpha;
    int splittingThreshold;
    int mergingThreshold;
    std::mutex mtx;

    double pdf(const Eigen::VectorXd& x, const GaussianComponent& component) {
        int d = x.size();
        double det = component.getCovariance().determinant();

        assert(det != 0);
        // double normConst = 1.0 / std::sqrt(std::pow(2 * M_PI, d) * det);
        double logNormConst = -0.5 * (d * std::log(2 * M_PI) +std::log(det));

        Eigen::VectorXd diff = x - component.getMean();
        double exponent = -0.5 * diff.transpose() * component.getCovariance().inverse() * diff;
        double logPdf = logNormConst + exponent;
        // std::cout << "Condition number: " << condition_number << std::endl;

        // std::cout << "exponent = " << exponent << " logNormConst = " << logNormConst << " logPdf = " << logPdf << std::endl;
        return std::exp(logPdf);
    }

    void updateSufficientStatistics(const std::vector<Eigen::VectorXd>& batch, const Eigen::MatrixXd& responsibilities) {
        std::lock_guard<std::mutex> lock(mtx);

        size_t N = 0;
        size_t NNew = batch.size();

        for (const auto& comp : components) {
            N += comp.getPriorSampleCount();
        }

        double weightSum = 0;
        for (size_t j = 0; j < components.size(); ++j) {
            auto& comp = components[j];

            double responsibilitySum = responsibilities.col(j).sum();
            if (responsibilitySum == 0) {
                SLog(mitsuba::EInfo, components[j].toString().c_str());
                // std::cout << components[j].toString().c_str();
            }
            assert(responsibilitySum != 0);
            double newWeight = responsibilitySum / NNew;
            assert(!std::isinf(newWeight));
            weightSum += newWeight;

            auto meanSize = comp.getMean().size();

            Eigen::VectorXd newMean = Eigen::VectorXd::Zero(meanSize);
            for (size_t i = 0; i < NNew; i++) {
                newMean += responsibilities(i, j) * batch[i];
            }
            newMean /= responsibilitySum;
            assert(!std::isinf(newMean[0]));

            Eigen::MatrixXd newCov = Eigen::MatrixXd::Zero(meanSize, meanSize);
            for (size_t i = 0; i < NNew; ++i) {
                Eigen::VectorXd diff = batch[i] - comp.getMean();
                newCov += responsibilities(i, j) * (diff * diff.transpose());
            }
            newCov /= responsibilitySum;
            assert(!std::isinf(newCov(0,0)));

            comp.updateComponent(N, NNew, newMean, newCov, newWeight, alpha);
        }

        if (weightSum != 1.0) {
            for (size_t j = 0; j < components.size(); ++j) {
                components[j].setWeight(components[j].getWeight() / weightSum);
            }
        }

        components.erase(
            std::remove_if(components.begin(), components.end(),
                [](const GaussianComponent& comp) {
                    return comp.getWeight() < 1e-3;
                }),
            components.end()
        );

        if (components.size() > 100) {
            std::cout << "Boom";
        }
        // splitting
        for (size_t j = 0; j < components.size(); ++j) {
            double conditionNumber = components[j].getCovariance().norm() * components[j].getCovariance().inverse().norm();
            if (conditionNumber > splittingThreshold)
                splitComponent(j);
        }

        // merging
        for (size_t i = 0; i < components.size(); ++i) {
            for (size_t j = i + 1; j < components.size(); ++j) {
                double distance = (components[i].getMean() - components[j].getMean()).norm();
                if (distance < mergingThreshold) {
                    mergeComponents(i, j);
                    break;
                }
            }
        }
    }

    void splitComponent(size_t index) {
        if (index >= components.size()) return;

        GaussianComponent& comp = components[index];

        // Split into two components
        Eigen::VectorXd offset = Eigen::VectorXd::Random(comp.getMean().size()) * 0.1; // Small random offset
        GaussianComponent newComp;

        Eigen::VectorXd mean = comp.getMean();

        comp.setWeight(comp.getWeight() / 2.0);
        comp.setMean(mean + offset);
        comp.setPriorSampleCount(comp.getPriorSampleCount() / 2);

        Eigen::MatrixXd perturbation = Eigen::MatrixXd::Identity(comp.getCovariance().rows(), comp.getCovariance().cols()) * 0.05;
        comp.setCovariance(comp.getCovariance() * 0.5 + perturbation);

        newComp.setMean(mean - 2 * offset);
        newComp.setCovariance(comp.getCovariance());
        newComp.setPriorSampleCount(comp.getPriorSampleCount());
        newComp.setCovariance(comp.getCovariance());

        components.push_back(newComp);
    }

    void mergeComponents(size_t index1, size_t index2) {
        if (index1 >= components.size() || index2 >= components.size() || index1 == index2) return;

        GaussianComponent& comp1 = components[index1];
        GaussianComponent& comp2 = components[index2];

        double totalWeight = comp1.getWeight() + comp2.getWeight();
        Eigen::VectorXd mergedMean = (comp1.getWeight() * comp1.getMean() + comp2.getWeight() * comp2.getMean()) / totalWeight;

        Eigen::MatrixXd mergedConvs =
            (comp1.getWeight() * (comp1.getCovariance() + (comp1.getMean() - mergedMean) * (comp1.getMean() - mergedMean).transpose()) +
             comp2.getWeight() * (comp2.getCovariance() + (comp2.getMean() - mergedMean) * (comp2.getMean() - mergedMean).transpose())) /
            totalWeight;

        comp1.setWeight(totalWeight);
        comp1.setMean(mergedMean);
        comp1.setCovariance(mergedConvs);
        comp1.setPriorSampleCount(comp1.getPriorSampleCount() + comp2.getPriorSampleCount());

        components.erase(components.begin() + index2);
    }


public:
    GaussianMixtureModel() {}

    void init(size_t numComponents, size_t dimension, double _alpha, int _splittingThreshold, int _mergingThreshold) {
        alpha = _alpha;
        splittingThreshold = _splittingThreshold;
        mergingThreshold = _mergingThreshold;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);

        components.resize(numComponents);
        for (auto& comp : components) {
            comp.setWeight(1.0 / numComponents);
            comp.setMean(Eigen::VectorXd::Zero(dimension));
            comp.setCovariance(Eigen::MatrixXd::Identity(dimension, dimension));

            Eigen::VectorXd compMean(dimension);
            for (size_t d = 0; d < dimension; ++d) {
                compMean[d] = dis(gen);
            }
            comp.setMean(compMean);
        }
    }

     void processBatch(const std::vector<Eigen::VectorXd>& batch) {
        Eigen::MatrixXd responsibilities = Eigen::MatrixXd::Zero(batch.size(), components.size());

        // E-step
        for (size_t j = 0; j < components.size(); ++j) {
            for (size_t i = 0; i < batch.size(); ++i) {
                double resp = components[j].getWeight() * pdf(batch[i], components[j]);
                // std::cout << resp << std::endl;
                responsibilities(i, j) = resp;
            }
        }

        // Normalize responsibilities
        for (int i = 0; i < responsibilities.rows(); ++i) {
            responsibilities.row(i) /= responsibilities.row(i).sum();
        }

        // M-step
        updateSufficientStatistics(batch, responsibilities);
    }

    Eigen::VectorXd sample(std::mt19937& gen) const {
        std::vector<double> weights;
        for (const auto& comp : components) {
            weights.push_back(comp.getWeight());
        }
        std::discrete_distribution<> component_dist(weights.begin(), weights.end());

        const GaussianComponent& selected_component = components[component_dist(gen)];

        return selected_component.sample(gen);
    }

    std::string toString() const
    {
        std::ostringstream oss;
        oss << "GMM[";
        for (const auto& component : components) {
            oss << component.toString() << "\n";
        }
        oss << "]";
        return oss.str();
    }

private:
    friend class boost::serialization::access;
    template<class Archive>
    void serialize(Archive & ar, const size_t version) {
        ar  & components;
    }

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}

#endif /* __MIXTURE_MODEL_H */