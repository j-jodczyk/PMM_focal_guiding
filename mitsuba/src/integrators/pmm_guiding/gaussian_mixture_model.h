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
#include <boost/thread/locks.hpp>
#include <boost/thread/shared_mutex.hpp>

#include <eigen3/Eigen/Dense>

#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include "gaussian_component.h"
#include "weighted_sample.h"

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
    mutable boost::shared_mutex mtx;

    std::vector<GaussianComponent> components;
    double alpha;
    double splittingThreshold;
    double mergingThreshold;
    u_int16_t minNumComp;
    u_int16_t maxNumComp;
    AABB m_aabb;
    size_t m_dimension;

    double pdf(const Eigen::VectorXd& x, GaussianComponent& component) {
        int d = x.size();
        double det = component.getCovariance().determinant();

        if (det <= 1e-6) {
            // SLog(mitsuba::EInfo, component.toString().c_str());
            // SLog(mitsuba::EWarn, "Unexpected non positive value of determinant, reseting component");
            resetComponent(component);
            det = component.getCovariance().determinant();
        }

        double logNormConst = -0.5 * (d * std::log(2 * M_PI) +std::log(det));
        // SLog(mitsuba::EInfo, "det: %f", det);

        Eigen::VectorXd diff = x - component.getMean();
        // SLog(mitsuba::EInfo, "diff: %f, %f, %f", diff(0), diff(1), diff(2));
        double exponent = -0.5 * diff.transpose() * component.getCovariance().inverse() * diff;
        // SLog(mitsuba::EInfo, "exponent: %f", exponent);
        double logPdf = logNormConst + exponent;
        // SLog(mitsuba::EInfo, "logPdf: %f", logPdf);
        // SLog(mitsuba::EInfo, "exp(logPdf): %f", std::exp(logPdf));

        return std::exp(logPdf);
    }

    void updateSufficientStatistics(const std::vector<WeightedSample>& batch, const Eigen::MatrixXd& responsibilities) {
        size_t N = 0;
        size_t NNew = batch.size();

        for (const auto& comp : components) {
            N += comp.getPriorSampleCount();
        }

        // double weightSum = 0;
        std::vector<size_t> componentIdxToDelete;
        for (size_t j = 0; j < components.size(); ++j) {
            auto& comp = components[j];
            if (j >= responsibilities.cols()) {
                SLog(mitsuba::EInfo, "j = %d, responsibilities.cols() = %d", j, responsibilities.cols());
                break; // TODO: probably do something better about this - maybe need to resize responsibilities (how do rebust vmm do it?)
            }

            double responsibilitySum = responsibilities.col(j).sum();
            if (responsibilitySum < 1e-6) {
                // SLog(mitsuba::EWarn, "Responsibility sum is close to 0 - deleting the component");
                componentIdxToDelete.push_back(j);
                continue;
            }
            if (responsibilitySum != responsibilitySum) {
                SLog(mitsuba::EInfo, "Crashing program - responsibility is -nan");
                SLog(mitsuba::EInfo, this->toString().c_str());
            }
            assert(responsibilitySum == responsibilitySum);
            double newWeight = responsibilitySum / NNew;
            assert(!std::isinf(newWeight));
            // weightSum += newWeight;

            Eigen::VectorXd newMean(comp.getMean().size());
            newMean.setZero();
            for (size_t i = 0; i < NNew; i++) {
                newMean += responsibilities(i, j) * batch[i].point;
                // SLog(mitsuba::EInfo, "Responsibility = %f", responsibilities(i, j));
            }

            newMean /= responsibilitySum;
            if(!newMean.allFinite()) {
                SLog(mitsuba::EInfo, this->toString().c_str());
                SLog(mitsuba::EInfo, "ResponsibilitySum = %f", responsibilitySum);
                SLog(mitsuba::EInfo, "Mean: %f, %f, %f", newMean[0], newMean[1], newMean[2]);
                assert(newMean.allFinite());
            }

            Eigen::MatrixXd newCov = Eigen::MatrixXd::Zero(comp.getMean().size(), comp.getMean().size());
            for (size_t i = 0; i < NNew; ++i) {
                Eigen::VectorXd diff = batch[i].point - comp.getMean();
                newCov += responsibilities(i, j) * (diff * diff.transpose());
            }
            newCov /= responsibilitySum;
            assert(newCov.allFinite());

            comp.updateComponent(N, NNew, newMean, newCov, newWeight, alpha);
        }

        if (componentIdxToDelete.size() == components.size()) {
            SLog(mitsuba::EWarn, "All components are supposed to be deleted... not sure what to do about this");
        }

        int componentCountAfterDeleting = components.size() - componentIdxToDelete.size();
        std::sort(componentIdxToDelete.rbegin(), componentIdxToDelete.rend()); // reverse sort so we don't invalidate idxs
        for (int idx : componentIdxToDelete)
            components.erase(components.begin() + idx);

        // if (weightSum != 1.0) {
        //     for (size_t j = 0; j < components.size(); ++j) {
        //         components[j].setWeight(components[j].getWeight() / weightSum);
        //     }
        // }

        // remove if weight too small
        components.erase(
            std::remove_if(components.begin(), components.end(),
                [](const GaussianComponent& comp) {
                    return comp.getWeight() < 1e-3;
                }),
            components.end()
        );

        // splitting
        for (size_t j = 0; j < components.size(); ++j) {
            if (components.size() >= maxNumComp)
                break;
            // high condition number indicates that matrix is close to being singular (det near 0), or its eigenbalues vary widely in magnitude.
            double conditionNumber = components[j].getCovariance().norm() * components[j].getCovariance().inverse().norm();
            if (conditionNumber > splittingThreshold) {
                // SLog(mitsuba::EInfo, "Splitting");
                splitComponent(j);
            }
        }

        // merging - when components too similar
        for (size_t i = 0; i < components.size(); ++i) {
            for (size_t j = i + 1; j < components.size(); ++j) {
                if (components.size() <= minNumComp)
                    break;
                if (components[i].getMean().cols() != components[j].getMean().cols()) {
                    SLog(mitsuba::EInfo, this->toString().c_str());
                }
                double distance = (components[i].getMean() - components[j].getMean()).norm();
                if (distance < mergingThreshold) {
                    // SLog(mitsuba::EInfo, "Merging");
                    mergeComponents(i, j);
                    break;
                }
            }
        }


        // todo: if this is needed, then weights need to be thought through
        // if (componentCountAfterDeleting < minNumComp) {
        //     int componentsToCreateCount = minNumComp - componentCountAfterDeleting;
        //     // SLog(mitsuba::EInfo, "Too many components to delete, will initialize %d new components", componentsToCreateCount);
        //     components.resize(components.size() + componentsToCreateCount);
        //     for (size_t i = componentsToCreateCount; i < components.size(); i++) {
        //         GaussianComponent& comp = components[i];
        //         initComponentMean(comp);
        //         initComponentCovaraince(comp);
        //     }
        // }

        // final weight recount
        double weightSum = 0;
        for (auto& component : components) {
            weightSum += component.getWeight();
        }

        for (size_t j = 0; j < components.size(); ++j) {
            components[j].setWeight(components[j].getWeight() / weightSum);
        }

    }

    void splitComponent(size_t index) {
        // assert(mtx.try_lock() == false);
        // boost::unique_lock<boost::shared_mutex> lock(mtx);
        if (components.size() > maxNumComp)
            SLog(mitsuba::EInfo, "splitting despite too large count of components");
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
        // assert(mtx.try_lock() == false);
        // boost::unique_lock<boost::shared_mutex> lock(mtx);
        if (components.size() < minNumComp)
            SLog(mitsuba::EInfo, "merging despite too small count of components");
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

    void setAlpha(double newAlpha) { alpha = newAlpha; };
    double getSplittingThreshold() { return splittingThreshold; };
    void setSplittingThreshold(double newThreshold) { splittingThreshold = newThreshold; };
    double getMergingThreshold() { return mergingThreshold; };
    void setMergingThreshold(double newThreshold) { mergingThreshold = newThreshold; };

    void setMinNumComp(u_int16_t newMinNumComp) { minNumComp = newMinNumComp; };
    void setMaxNumComp(u_int16_t newMaxNumComp) { maxNumComp = newMaxNumComp; };

    void initComponentMean(GaussianComponent& comp) {
        std::random_device rd;
        std::mt19937 gen(rd());

        comp.setMean(Eigen::VectorXd::Zero(m_dimension));

        Eigen::VectorXd compMean(m_dimension);
        for (size_t d = 0; d < m_dimension; ++d) {
            std::uniform_real_distribution<> dist(m_aabb.min[d], m_aabb.max[d]);
            compMean[d] = dist(gen);
        }
        comp.setMean(compMean);
    }

    void initComponentCovaraince(GaussianComponent& comp) {
        comp.setCovariance(Eigen::MatrixXd::Identity(m_dimension, m_dimension));
    }

    void init(size_t numComponents, size_t dimension, const AABB& aabb) {
        m_aabb = aabb;
        m_dimension = dimension;

        components.resize(numComponents);
        for (auto& comp : components) {
            comp.setWeight(1.0 / numComponents);
            initComponentMean(comp);
            initComponentCovaraince(comp);
        }
    }

    void resetComponent(GaussianComponent& component) {
        std::random_device rd;
        std::mt19937 gen(rd());

        component.setWeight(1.0 / components.size());

        Eigen::VectorXd compMean(component.getMean().size());
        for (int d = 0; d < component.getMean().size(); ++d) {
            std::uniform_real_distribution<> dist(m_aabb.min[d], m_aabb.max[d]);
            compMean[d] = dist(gen);
        }
        component.setMean(compMean);
        component.setCovariance(Eigen::MatrixXd::Identity(component.getCovariance().rows(), component.getCovariance().cols()));
    }

     void processBatch(const std::vector<WeightedSample>& batch) {
        // TODO: there must be a better way to do this, but for now:
        boost::unique_lock<boost::shared_mutex> lock(mtx);

        Eigen::MatrixXd responsibilities = Eigen::MatrixXd::Zero(batch.size(), components.size());

        // E-step
        {
            // boost::shared_lock<boost::shared_mutex> lock(mtx);
            for (size_t j = 0; j < components.size(); ++j) {
                for (size_t i = 0; i < batch.size(); ++i) {
                    double resp = components[j].getWeight() * pdf(batch[i].point, components[j]);
                    responsibilities(i, j) = resp * batch[i].weight;
                }
            }
            // SLog(mitsuba::EInfo, "E-step mutex release");
        }

        // Normalize responsibilities
        for (int i = 0; i < responsibilities.rows(); ++i) {
            responsibilities.row(i) /= responsibilities.row(i).sum();
        }

        // M-step
        {
            // boost::unique_lock<boost::shared_mutex> lock(mtx);
            updateSufficientStatistics(batch, responsibilities);
            // SLog(mitsuba::EInfo, "Sufficient Statistics mutex release");
        }
    }

    Eigen::VectorXd sample(std::mt19937& gen) const {
        boost::shared_lock<boost::shared_mutex> lock(mtx);
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