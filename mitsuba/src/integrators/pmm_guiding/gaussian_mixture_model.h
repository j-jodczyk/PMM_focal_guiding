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
    size_t minNumComp;
    size_t maxNumComp;
    std::atomic<size_t> numActiveComponents;
                                             // todo granual mutexes - only on modifications of the vector, right?
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

        for (size_t i = 0; i < numActiveComponents.load(); ++i) {
            N += components[i].getPriorSampleCount();
        }

        std::vector<size_t> componentIdxToDelete;
        for (size_t j = 0; j < numActiveComponents.load(); ++j) {
            auto& comp = components[j];
            if (j >= (size_t)responsibilities.cols()) {
                // SLog(mitsuba::EInfo, "j = %d, responsibilities.cols() = %d", j, responsibilities.cols());
                break; // TODO: probably do something better about this - maybe need to resize responsibilities (how do rebust vmm do it?)
            }

            double responsibilitySum = responsibilities.col(j).sum();
            if (responsibilitySum < 1e-6) {
                componentIdxToDelete.push_back(j);
                continue;
            }
            if (responsibilitySum != responsibilitySum) {
                // SLog(mitsuba::EInfo, "Crashing program - responsibility is -nan");
                // SLog(mitsuba::EInfo, this->toString().c_str());
                throw std::runtime_error("Responsibility cannot be nan");
            }
            double newWeight = responsibilitySum / NNew;
            assert(!std::isinf(newWeight));

            Eigen::VectorXd newMean(comp.getMean().size());
            newMean.setZero();
            for (size_t i = 0; i < NNew; i++) {
                newMean += responsibilities(i, j) * batch[i].point;
            }

            newMean /= responsibilitySum;
            if(!newMean.allFinite()) {
                // SLog(mitsuba::EInfo, this->toString().c_str());
                // SLog(mitsuba::EInfo, "ResponsibilitySum = %f", responsibilitySum);
                // SLog(mitsuba::EInfo, "Mean: %f, %f, %f", newMean[0], newMean[1], newMean[2]);
                throw std::runtime_error("new mean must be finite number");
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
        SLog(mitsuba::EInfo, "Prior to deletion analysis");
        SLog(mitsuba::EInfo, this->toString().c_str());

        if (componentIdxToDelete.size() == components.size()) {
            SLog(mitsuba::EWarn, "All components are supposed to be deleted... not sure what to do about this");
        }

        numActiveComponents.store(components.size() - componentIdxToDelete.size());
        for (int idx : componentIdxToDelete) {
            this->deactivateComponent(idx);
        }

        // remove if weight too small
        for (size_t j = 0; j < numActiveComponents.load(); ++j) {
            if (components[j].getWeight() < 1e-3)
                this->deactivateComponent(j);
        }
        SLog(mitsuba::EInfo, "Prior to splitting analysis");
        SLog(mitsuba::EInfo, this->toString().c_str());

        // splitting
        for (size_t j = 0; j < numActiveComponents.load(); ++j) {
            if (numActiveComponents.load() >= maxNumComp)
                break;
            // high condition number indicates that matrix is close to being singular (det near 0), or its eigenbalues vary widely in magnitude.
            double conditionNumber = components[j].getCovariance().norm() * components[j].getCovariance().inverse().norm();
            if (conditionNumber > splittingThreshold) {
                // SLog(mitsuba::EInfo, "Splitting");
                splitComponent(j);
            }
        }

        SLog(mitsuba::EInfo, "Prior to merging");
        SLog(mitsuba::EInfo, this->toString().c_str());

        // merging - when components too similar
        for (size_t i = 0; i < numActiveComponents.load(); ++i) {
            for (size_t j = i + 1; j < numActiveComponents.load(); ++j) {
                if (numActiveComponents.load() <= minNumComp)
                    break;
                double distance = (components[i].getMean() - components[j].getMean()).norm();
                if (distance < mergingThreshold) {
                    mergeComponents(i, j);
                    break;
                }
            }
        }

        // final weight recount
        double weightSum = 0;
        for (size_t i = 0; i < numActiveComponents.load(); ++i) {
            weightSum += components[i].getWeight();
        }

        for (size_t j = 0; j < numActiveComponents.load(); ++j) {
            components[j].setWeight(components[j].getWeight() / weightSum);
        }

        SLog(mitsuba::EInfo, "Final");
        SLog(mitsuba::EInfo, this->toString().c_str());
    }

public:
    void deactivateComponent(int idx) {
        components[idx].deactivate(m_dimension);
        numActiveComponents.fetch_sub(1);
        std::swap(components[idx], components[numActiveComponents]);
    }

    void splitComponent(size_t index) {
        // assert(mtx.try_lock() == false);
        // boost::unique_lock<boost::shared_mutex> lock(mtx);
        // todo: components.size() can be replaced with maxNumComp
        if (components.size() == numActiveComponents.load()) {
            // SLog(mitsuba::EInfo, "Called splitting function at components capacity - returning without split");
            return;
        }

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

        newComp.setWeight(comp.getWeight());
        newComp.setMean(mean - 2 * offset);
        newComp.setCovariance(comp.getCovariance());
        newComp.setPriorSampleCount(comp.getPriorSampleCount());
        newComp.setCovariance(comp.getCovariance());

        components[numActiveComponents.load()] = newComp; // todo: might need some synchronization around this
                                                   // or check if components[numActiveComponents] has weight == 0
        numActiveComponents.fetch_add(1);
        // SLog(mitsuba::EInfo, this->toString().c_str());
    }

    void mergeComponents(size_t index1, size_t index2) {
        // assert(mtx.try_lock() == false);
        // boost::unique_lock<boost::shared_mutex> lock(mtx);
        if (numActiveComponents.load() == minNumComp) {
            // SLog(mitsuba::EInfo, "Called merge function when minumum number of components already reached. Returing without merge");
            return;
        }
        // sanity check:
        if (index1 >= components.size() || index2 >= components.size() || index1 == index2) return;

        size_t earlierIdx = index1 < index2 ? index1 : index2;
        size_t laterIdx = index1 < index2 ? index2 : index1;

        GaussianComponent& earlierComponent = components[earlierIdx];
        GaussianComponent& laterComponent = components[laterIdx];

        if (earlierComponent.getWeight() == 0 || laterComponent.getWeight() == 0) {
            // SLog(mitsuba::EInfo, "Trying to merge components, when one of them is inactive - returing");
            return;
        }

        double totalWeight = earlierComponent.getWeight() + laterComponent.getWeight();

        Eigen::VectorXd mergedMean = (
            earlierComponent.getWeight() * earlierComponent.getMean()
            + laterComponent.getWeight() * laterComponent.getMean()) / totalWeight;

        Eigen::MatrixXd mergedConvs =
            (earlierComponent.getWeight() * (earlierComponent.getCovariance() + (earlierComponent.getMean() - mergedMean) * (earlierComponent.getMean() - mergedMean).transpose()) +
             laterComponent.getWeight() * (laterComponent.getCovariance() + (laterComponent.getMean() - mergedMean) * (laterComponent.getMean() - mergedMean).transpose())) /
            totalWeight;

        // todo: probably safer to first create a new component and then atomically try to replace it with a new one
        earlierComponent.setWeight(totalWeight);
        earlierComponent.setMean(mergedMean);
        earlierComponent.setCovariance(mergedConvs);
        earlierComponent.setPriorSampleCount(earlierComponent.getPriorSampleCount() + laterComponent.getPriorSampleCount());

        deactivateComponent(laterIdx);

        // no need to swap or decrease the number of active components because deactiveComponents already does it
        // SLog(mitsuba::EInfo, this->toString().c_str());
    }

    GaussianMixtureModel() {}

    void setAlpha(double newAlpha) { alpha = newAlpha; };
    double getSplittingThreshold() { return splittingThreshold; };
    void setSplittingThreshold(double newThreshold) { splittingThreshold = newThreshold; };
    double getMergingThreshold() { return mergingThreshold; };
    void setMergingThreshold(double newThreshold) { mergingThreshold = newThreshold; };

    size_t getMinNumComp() { return minNumComp; };
    size_t getMaxNumComp() { return maxNumComp; };
    size_t getNumActiveComponents() { return numActiveComponents.load(); };
    void setMinNumComp(size_t newMinNumComp) { minNumComp = newMinNumComp; };
    void setMaxNumComp(size_t newMaxNumComp) { maxNumComp = newMaxNumComp; };

    std::vector<GaussianComponent> getComponents() { return components; };

    void initComponentMean(GaussianComponent& comp) {
        std::random_device rd;
        std::mt19937 gen(rd());

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
        if (numComponents > maxNumComp)
            throw std::runtime_error("Too many components on initialization");
        m_aabb = aabb;
        m_dimension = dimension;
        if (components.size() != maxNumComp)
            components.resize(maxNumComp);

        for (size_t i = 0; i < numComponents; ++i) {
            auto& component = components[i];
            component.setWeight(1.0 / numComponents);
            initComponentMean(component);
            initComponentCovaraince(component);
        }
        numActiveComponents.store(numComponents);

        // the rest of the components are initated with zero values
        for (size_t i = numComponents; i < maxNumComp; ++i) {
            components[i].deactivate(m_dimension);
        }
    }

    void resetComponent(GaussianComponent& component) {
        std::random_device rd;
        std::mt19937 gen(rd());
        component.setWeight(1.0 / numActiveComponents.load());

        Eigen::VectorXd compMean(component.getMean().size());
        for (int d = 0; d < component.getMean().size(); ++d) {
            std::uniform_real_distribution<> dist(m_aabb.min[d], m_aabb.max[d]);
            compMean[d] = dist(gen);
        }
        component.setMean(compMean);
        component.setCovariance(Eigen::MatrixXd::Identity(component.getCovariance().rows(), component.getCovariance().cols()));
    }

     void processBatch(const std::vector<WeightedSample>& batch) {
        // just a note:
        //  unique_lock is used for EXCLUSIVE WRITE access during modifications.
        //  shared_lock is used for READ access in functions like sample and during the E-step of processBatch.
        // we do it outside of render - no need for locking - possibly need to parallelize later

        Eigen::MatrixXd responsibilities = Eigen::MatrixXd::Zero(batch.size(), components.size());

        // E-step
        {
            for (size_t j = 0; j < numActiveComponents.load(); ++j) {
                auto component = components[j];
                for (size_t i = 0; i < batch.size(); ++i) {
                    double resp = component.getWeight() * pdf(batch[i].point, component);
                    responsibilities(i, j) = resp * batch[i].weight;
                }
            }
        }

        // Normalize responsibilities
        for (int i = 0; i < responsibilities.rows(); ++i) { // rows is batch.size()
            responsibilities.row(i) /= responsibilities.row(i).sum();
        }

        // M-step
        {
            updateSufficientStatistics(batch, responsibilities);
        }
    }

    Eigen::VectorXd sample(std::mt19937& gen) const {
        // boost::shared_lock<boost::shared_mutex> lock(mtx);
        std::vector<double> weights;
        for (size_t i; i < numActiveComponents.load(); ++i) {
            if (components.size() < numActiveComponents.load()) {
                SLog(mitsuba::EInfo, "List of components is smaller than it claims to be: expected %d, but got %d", numActiveComponents.load(), components.size());
                throw std::runtime_error("List of components is smaller than it claims to be");
            }
            auto component = components[i];
            weights.push_back(component.getWeight());
        }

        std::discrete_distribution<> componentDist(weights.begin(), weights.end());
        const GaussianComponent& selectedComponent = components[componentDist(gen)];

        if (selectedComponent.getWeight() == 0) {
            // todo: this cannot happen
            SLog(mitsuba::EInfo, "Sampled component with 0 weight (currently not active)");
        }

        return selectedComponent.sample(gen);
    }

    std::string toString() const
    {
        std::ostringstream oss;
        oss << "GMM[\n";
        for (size_t i = 0; i < components.size(); ++i) {
            oss << components[i].toString() << "\n";
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