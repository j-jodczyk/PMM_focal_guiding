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
#include <queue>
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

class PRNG {
    public:
        PRNG() {
            std::random_device rd;
            generator.seed(rd());
        }

        int getRandomNumber(int min, int max) {
            std::uniform_int_distribution<int> distribution(min, max);
            return distribution(generator);
        }

    private:
        std::default_random_engine generator;
    };

template<
    typename Scalar_t,
    typename Env
>
class GaussianMixtureModel : public Distribution<Scalar_t> {
private:
    PRNG prng;
    using Scalar = Scalar_t;
    using AABB = typename Env::AABB;

    std::vector<GaussianComponent> components;
    double alpha;
    double splittingThreshold;
    double mergingThreshold;
    size_t minNumComp;
    size_t maxNumComp;
    AABB m_aabb;
    size_t m_dimension;
    size_t sampleCount = 0;
    double divergeProbability = 0.5;
    bool initialized = false;

    double pdf(const Eigen::VectorXd& x, GaussianComponent& component) {
        int d = x.size();
        double det = component.getCovariance().determinant();

        if (det <= 1e-6) {
            resetComponent(component);
            det = component.getCovariance().determinant();
        }

        double logNormConst = -0.5 * (d * std::log(2 * M_PI) +std::log(det));

        Eigen::VectorXd diff = x - component.getMean();
        double exponent = -0.5 * diff.transpose() * component.getCovariance().inverse() * diff;
        double logPdf = logNormConst + exponent;

        return std::exp(logPdf);
    }

    double totalSampleWeight(const std::vector<WeightedSample>& batch) {
        double sum = 0;
        for (const auto& sample : batch) {
            sum += sample.weight;
        }
        return sum;
    }

    void updateSufficientStatistics(const std::vector<WeightedSample>& batch, const Eigen::MatrixXd& responsibilities) {
        size_t NNew = batch.size();
        double totalWeight = totalSampleWeight(batch);

        double weights = 0;
        for (size_t j = 0; j < components.size(); ++j) {
            if (components[j].getWeight() == 0)
                continue;
            auto& comp = components[j];

            // double responsibilitySum = responsibilities.col(j).sum();
            // Compute weighted responsibility sum - Hanebeck, Frisch
            double responsibilitySum = 0;
            for (size_t i = 0; i < NNew; ++i) {
                responsibilitySum += responsibilities(i, j) * batch[i].weight;
            }
            if (responsibilitySum < 1e-6) {
                SLog(mitsuba::EInfo, "Responsibility sum is below threshold");
                // componentIdxToDelete.push_back(j);
                continue;
            }
            if (responsibilitySum != responsibilitySum) {
                throw std::runtime_error("Responsibility cannot be nan");
            }
            double newWeight = responsibilitySum / totalWeight;
            assert(!std::isinf(newWeight));

            Eigen::VectorXd newMean(comp.getMean().size());
            newMean.setZero();
            for (size_t i = 0; i < NNew; i++) {
                newMean += responsibilities(i, j) * batch[i].point * batch[i].weight;
            }

            newMean /= responsibilitySum;
            if(!newMean.allFinite()) {
                throw std::runtime_error("new mean must be finite number");
            }

            Eigen::MatrixXd newCov = Eigen::MatrixXd::Zero(comp.getMean().size(), comp.getMean().size());
            for (size_t i = 0; i < NNew; ++i) {
                Eigen::VectorXd diff = batch[i].point - comp.getMean();
                newCov += responsibilities(i, j) * batch[i].weight * (diff * diff.transpose());
            }
            newCov /= responsibilitySum;
            assert(newCov.allFinite());

            comp.updateComponent(sampleCount, NNew, newMean, newCov, newWeight, alpha);
            weights += newWeight;
        }

        SLog(mitsuba::EInfo, "Components accumulated weight is: %f", weights);

        sampleCount += NNew; // update sample count

        mergeAllComponents();
        splitAllComponents();
    }

    void deactivateComponent(int idx) {
        components[idx].deactivate(m_dimension);
    }

    double bhattacharyyaDistance(int idx1, int idx2) {
        const auto& comp1 = components[idx1];
        const auto& comp2 = components[idx2];

        Eigen::VectorXd meanDiff = comp1.getMean() - comp2.getMean();

        Eigen::MatrixXd covMean = 0.5 * (comp1.getCovariance() + comp2.getCovariance());
        Eigen::MatrixXd covMeanInv = covMean.inverse();

        double cov1Det = comp1.getCovariance().determinant();
        double cov2Det = comp2.getCovariance().determinant();

        double multiply = meanDiff.transpose() * covMeanInv * meanDiff;
        double first =  multiply / 8;

        double second = 0.5 * log(covMean.determinant() / sqrt(cov1Det * cov2Det));

        return first + second;
    }

    void mergeAllComponents() {
        int maxMergeCount = maxNumComp; // todo: think through / parametrize
        // table of bhattacharyya coefficients between all components
        // the closer BC is to 1 the more similar the distributions are
        Eigen::MatrixXd bhattacharyyaCoefficients(components.size(), components.size());
        for (size_t i = 0; i < components.size(); ++i) {
            for (size_t j = i + 1; j < components.size(); ++j) {
                if (components[j].getWeight() == 0 || components[i].getWeight() == 0 || j == i) {
                    bhattacharyyaCoefficients(i, j) = 0;
                    bhattacharyyaCoefficients(j, i) = 0;
                    continue;
                }
                auto dist = exp(-bhattacharyyaDistance(i, j)); // BC = 1/exp(BD)
                bhattacharyyaCoefficients(i, j) = dist;
                bhattacharyyaCoefficients(j, i) = dist;
                SLog(mitsuba::EInfo, "exp(-dist) between %d and %d is %f", i, j, dist);
            }
        }

        size_t i, j;
        bhattacharyyaCoefficients.maxCoeff(&i, &j);
        double maxBC = bhattacharyyaCoefficients(i, j);
        SLog(mitsuba::EInfo, "components before the merge: %d", getNumActiveComponents());
        int mergeCount = 0;
        do {
            SLog(mitsuba::EInfo, "Merging %d and %d, because their BC is %f", i, j, maxBC);
            mergeComponents(i, j);
            mergeCount++;
            bhattacharyyaCoefficients.col(j).setZero();
            bhattacharyyaCoefficients.row(j).setZero();

            for (size_t k = 0; k < components.size(); ++k) {
                if (k != i && components[k].getWeight() > 0) {
                    auto dist = exp(-bhattacharyyaDistance(i, k));
                    bhattacharyyaCoefficients(i, k) = dist;
                    bhattacharyyaCoefficients(k, i) = dist;
                }
            }

            bhattacharyyaCoefficients.maxCoeff(&i, &j);
            maxBC = bhattacharyyaCoefficients(i, j);
        } while(maxBC > mergingThreshold && mergeCount < maxMergeCount);
        SLog(mitsuba::EInfo, "components after the merge: %d, merged %d component", getNumActiveComponents(), mergeCount);
    }

    void splitAllComponents() {
        int maxSplitCount = maxNumComp; // todo: think through / parametrize
        std::queue<size_t> splitCandidates;
        for (size_t i = 0; i < components.size(); ++i) {
            if (components[i].getWeight() > 0)
                splitCandidates.push(i);
        }
        int splitCount = 0;

        while(!splitCandidates.empty() && splitCount < maxSplitCount) {
            int i = splitCandidates.front();
            splitCandidates.pop();

            splitComponent(i, splitCandidates, splitCount);
        }
        SLog(mitsuba::EInfo, "components after the split: %d, split %d components", getNumActiveComponents(), splitCount);
    }

    void splitComponent(size_t index, std::queue<size_t> &splitCandidates, int &splitCount) {
        if (getNumActiveComponents() >= maxNumComp) return;

        // using PCA
        auto cov = components[index].getCovariance();
        auto mean = components[index].getMean();

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(cov);
        Eigen::VectorXd eigenvalues = solver.eigenvalues();
        Eigen::MatrixXd eigenvectors = solver.eigenvectors();

        // check if to split
        double minEigenvalue = eigenvalues.minCoeff();
        double maxEigenvalue = eigenvalues.maxCoeff();

        double ratio = maxEigenvalue / minEigenvalue;
        SLog(mitsuba::EInfo, "Ratio is %f", ratio);
        if (ratio < splittingThreshold)
            return; // below threshold - do nothing

        splitCount++;

        // Identify the principal eigenvector (largest eigenvalue)
        int maxIndex;
        eigenvalues.maxCoeff(&maxIndex);
        Eigen::VectorXd principalAxis = eigenvectors.col(maxIndex);

        double scalingFactor = 0.5 * std::sqrt(maxEigenvalue);
        Eigen::VectorXd deltaMean = scalingFactor * principalAxis;

        auto newMean1 = mean + deltaMean;
        auto newMean2 = mean - deltaMean;

        //  Adjust the covariance matrix
        Eigen::MatrixXd principalCov = maxEigenvalue * (principalAxis * principalAxis.transpose());
        auto newCov = cov - (0.5 * principalCov); // keeping symmetrical

        auto newWeight = components[index].getWeight() / 2.0;

        components[index].setWeight(newWeight);
        components[index].setMean(newMean1);
        components[index].setCovariance(newCov);
        splitCandidates.push(index);

        GaussianComponent newComp;
        newComp.setWeight(newWeight);
        newComp.setMean(newMean2);
        newComp.setCovariance(newCov);

        for (size_t i=0; i < components.size(); ++i) {
            if (components[i].getWeight() == 0) {
                // found first deactivated component - replace with the new one
                components[i] = newComp;
                splitCandidates.push(i);
                break;
            }
        }
    }

    void mergeComponents(size_t index1, size_t index2) {
        if (getNumActiveComponents() == minNumComp) return;

        if (index1 >= components.size() || index2 >= components.size() || index1 == index2) return;

        GaussianComponent& component1 = components[index1];
        GaussianComponent& component2 = components[index2];

        // new values based on:
        // Z. Zhang, C. Chen, J. Sun, and K. L. Chan, “EM algorithms for Gaussian mixtures with split-and-merge operation,” Pattern Recognition, vol. 36, no. 9, pp. 1973 – 1983, 2003.
        double totalWeight = component1.getWeight() + component2.getWeight(); // we're just adding weights together

        Eigen::VectorXd mergedMean = (
            component1.getWeight() * component1.getMean() +
            component2.getWeight() * component2.getMean()) / totalWeight;

        Eigen::MatrixXd correction = mergedMean * mergedMean.transpose();
        Eigen::MatrixXd weightedCovaraince1 = component1.getWeight() * component1.getCovariance();
        Eigen::MatrixXd weightedCovaraince2 = component2.getWeight() * component2.getCovariance();

        Eigen::MatrixXd weightedMean1TimesTransposed = component1.getWeight() * component1.getMean() * component1.getMean().transpose();
        Eigen::MatrixXd weightedMean2TimesTransposed = component2.getWeight() * component2.getMean() * component2.getMean().transpose();

        Eigen::MatrixXd mergedConvs = (weightedCovaraince1 + weightedCovaraince2 + weightedMean1TimesTransposed + weightedMean2TimesTransposed) / totalWeight - correction;

        component1.setWeight(totalWeight);
        component1.setMean(mergedMean);
        component1.setCovariance(mergedConvs);

        deactivateComponent(index2);
    }

public:
    GaussianMixtureModel(): prng() {}
    GaussianMixtureModel(std::istream& in) {
        deserialize(in);
    }

    double getAlpha() { return alpha; }
    void setAlpha(double newAlpha) { alpha = newAlpha; };
    double getSplittingThreshold() { return splittingThreshold; };
    void setSplittingThreshold(double newThreshold) { splittingThreshold = newThreshold; };
    double getMergingThreshold() { return mergingThreshold; };
    void setMergingThreshold(double newThreshold) { mergingThreshold = newThreshold; };

    void setAABB(AABB newAabb) { m_aabb = newAabb; };

    size_t getMinNumComp() { return minNumComp; };
    size_t getMaxNumComp() { return maxNumComp; };
    size_t getNumActiveComponents() {
        return std::accumulate(components.begin(), components.end(), 0, [](int count, const GaussianComponent& component) {
            return count + (component.getWeight() > 0 ? 1 : 0);
        });
     };
    void setMinNumComp(size_t newMinNumComp) { minNumComp = newMinNumComp; };
    void setMaxNumComp(size_t newMaxNumComp) { maxNumComp = newMaxNumComp; };

    double getDivergeProbability() { return divergeProbability; }
    void setDivergeProbability(double newDivergeProbability) { divergeProbability = newDivergeProbability; }

    std::vector<GaussianComponent> getComponents() { return components; };

    void initComponentCovariance(GaussianComponent& comp) {
        comp.setCovariance(Eigen::MatrixXd::Identity(m_dimension, m_dimension));
    }

    void init(
        const std::vector<WeightedSample>& batch
    ) {
        // Method number 1.: Initialize with random values from the first batch - initializeing maximum number of components
        SLog(mitsuba::EInfo, "Starting initialization using random value from the first batch");
        m_dimension = batch[0].point.size();

        if (components.size() != maxNumComp)
            components.resize(maxNumComp);

        for (size_t i = 0; i < maxNumComp; ++i) {
            auto& component = components[i];
            component.setWeight(1.0 / maxNumComp);
            int randomIndex = prng.getRandomNumber(0, batch.size() - 1);
            component.setMean(batch[randomIndex].point); // initialize mean as random value from batch
            initComponentCovariance(component);
        }
        SLog(mitsuba::EInfo, "Initialized using random value from the first batch");

        initialized = true;
    }

    void initKMeans(const std::vector<WeightedSample>& batch) {
        // Method number 2: Using KMeans
        SLog(mitsuba::EInfo, "Starting initialization using KMeans");
        m_dimension = batch[0].point.size();

        if (components.size() != maxNumComp)
            components.resize(maxNumComp);

        std::vector<Eigen::VectorXd> centroids(maxNumComp, Eigen::VectorXd::Zero(m_dimension));
        std::vector<int> clusterAssignments(batch.size(), 0);

        for (size_t i = 0; i < maxNumComp; ++i) {
            int randomIndex = prng.getRandomNumber(0, batch.size() - 1);
            centroids[i] = batch[randomIndex].point;
        }

        bool converged = false;
        int maxIterations = 100;

        std::vector<int> clusterSizes(maxNumComp, 0);

        for (int iter = 0; iter < maxIterations && !converged; ++iter) {
            converged = true;

            // assign each sample to the nearest centroid
            for (size_t i = 0; i < batch.size(); ++i) {
                double minDist = std::numeric_limits<double>::max();
                int bestCluster = 0;

                for (size_t j = 0; j < maxNumComp; ++j) {
                    double dist = (batch[i].point - centroids[j]).squaredNorm();
                    if (dist < minDist) {
                        minDist = dist;
                        bestCluster = j;
                    }
                }

                if (clusterAssignments[i] != bestCluster) {
                    clusterAssignments[i] = bestCluster;
                    converged = false;
                }
            }

            // update centroids
            std::vector<Eigen::VectorXd> newCentroids(maxNumComp, Eigen::VectorXd::Zero(m_dimension));

            for (size_t i = 0; i < batch.size(); ++i) {
                newCentroids[clusterAssignments[i]] += batch[i].point;
                clusterSizes[clusterAssignments[i]]++;
            }

            for (size_t j = 0; j < maxNumComp; ++j) {
                if (clusterSizes[j] > 0) {
                    centroids[j] = newCentroids[j] / clusterSizes[j];
                }
            }
        }

        // assign K-means results to GMM components
        for (size_t j = 0; j < maxNumComp; ++j) {
            auto& component = components[j];
            component.setWeight(static_cast<double>(clusterSizes[j]) / batch.size());
            component.setMean(centroids[j]);

            // Compute covariance for each cluster
            Eigen::MatrixXd covariance = Eigen::MatrixXd::Zero(m_dimension, m_dimension);
            if (clusterSizes[j] > 1) {
                for (size_t i = 0; i < batch.size(); ++i) {
                    if (clusterAssignments[i] == j) {
                        Eigen::VectorXd diff = batch[i].point - centroids[j];
                        covariance += diff * diff.transpose();
                    }
                }
                covariance /= (clusterSizes[j] - 1);
            } else {
                covariance = Eigen::MatrixXd::Identity(m_dimension, m_dimension);
            }

            component.setCovariance(covariance);
        }
        SLog(mitsuba::EInfo, "Initialized using KMeans");

        initialized = true;
    }

    void resetComponent(GaussianComponent& component) {
        SLog(mitsuba::EInfo, "Resetting component");
        std::random_device rd;
        std::mt19937 gen(rd());

        component.setWeight(1.0 / getNumActiveComponents());

        Eigen::VectorXd compMean(component.getMean().size());
        for (int d = 0; d < component.getMean().size(); ++d) {
            std::uniform_real_distribution<> dist(m_aabb.min[d], m_aabb.max[d]);
            compMean[d] = dist(gen);
        }
        component.setMean(compMean);
        component.setCovariance(Eigen::MatrixXd::Identity(component.getCovariance().rows(), component.getCovariance().cols()));
    }

     void processBatch(const std::vector<WeightedSample>& batch) {
        if (!initialized)
            initKMeans(batch);


        Eigen::MatrixXd responsibilities = Eigen::MatrixXd::Zero(batch.size(), components.size());

        // E-step
        {
            for (size_t j = 0; j < components.size(); ++j) {
                auto component = components[j];
                if (component.getWeight() == 0)
                    continue;
                for (size_t i = 0; i < batch.size(); ++i) {
                    double resp = component.getWeight() * pdf(batch[i].point, component);
                    responsibilities(i, j) = resp; // without * batch[i].weight; - Hanebeck, Frisch - consider weights only in M step - this is an absolute game changer!
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

    // using Inverse Transform Sampling
    Eigen::VectorXd sample(mitsuba::RadianceQueryRecord &rRec) const {
        // Compute total weight
        double totalWeight = 0;
        for (const auto& component : components) {
            totalWeight += component.getWeight();
        }

        // Sample a component directly using a cumulative sum
        double randWeight = rRec.nextSample1D() * totalWeight;
        double cumulativeWeight = 0;

        for (const auto& component : components) {
            cumulativeWeight += component.getWeight();
            if (randWeight <= cumulativeWeight) {
                return component.sample(rRec);
            }
        }

        // fallback
        SLog(mitsuba::EInfo, "Sampling from the first component (should never happen)");
        return components[0].sample(rRec);
    }

    std::string toString() const
    {
        std::ostringstream oss;
        oss << "GMM[\n";
        for (size_t i = 0; i < components.size(); ++i) {
            if (components[i].getWeight() == 0)
                continue; // no need to print deactivated components
            oss << components[i].toString() << "\n";
        }
        oss << "]";
        return oss.str();
    }

    void serialize(mitsuba::FileStream* out)const
    {
        out->write(reinterpret_cast<const char*>(&alpha), sizeof(alpha));
        out->write(reinterpret_cast<const char*>(&splittingThreshold), sizeof(splittingThreshold));
        out->write(reinterpret_cast<const char*>(&mergingThreshold), sizeof(mergingThreshold));
        out->write(reinterpret_cast<const char*>(&minNumComp), sizeof(minNumComp));
        out->write(reinterpret_cast<const char*>(&maxNumComp), sizeof(maxNumComp));
        out->write(reinterpret_cast<const char*>(&m_dimension), sizeof(m_dimension));
        out->write(reinterpret_cast<const char*>(&sampleCount), sizeof(sampleCount));
        // out->write(reinterpret_cast<const char*>(&divergeProbability), sizeof(divergeProbability));

        // Save components
        size_t numComponents = components.size();
        out->write(reinterpret_cast<const char*>(&numComponents), sizeof(numComponents));
        for (const auto& comp : components) {
            comp.serialize(out);  // Call GaussianComponent's save method
        }
    }

    void deserialize(mitsuba::FileStream* in)
    {
        in->read(reinterpret_cast<char*>(&alpha), sizeof(alpha));
        in->read(reinterpret_cast<char*>(&splittingThreshold), sizeof(splittingThreshold));
        in->read(reinterpret_cast<char*>(&mergingThreshold), sizeof(mergingThreshold));
        in->read(reinterpret_cast<char*>(&minNumComp), sizeof(minNumComp));
        in->read(reinterpret_cast<char*>(&maxNumComp), sizeof(maxNumComp));
        in->read(reinterpret_cast<char*>(&m_dimension), sizeof(m_dimension));
        in->read(reinterpret_cast<char*>(&sampleCount), sizeof(sampleCount));
        // in->read(reinterpret_cast<char*>(&divergeProbability), sizeof(divergeProbability));

        // Read components
        size_t numComponents;
        in->read(reinterpret_cast<char*>(&numComponents), sizeof(numComponents));
        components.resize(numComponents);
        for (auto& comp : components) {
            comp.deserialize(in);
        }
    }

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}

#endif /* __MIXTURE_MODEL_H */