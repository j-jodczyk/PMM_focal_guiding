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
#include <thread>

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

struct ComponentCache {
    Eigen::MatrixXd invCov;
    Eigen::VectorXd mean;
    float logNormConst;
    float weight;
};

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
    float alpha;
    float splittingThreshold;
    float mergingThreshold;
    size_t minNumComp;
    size_t maxNumComp;
    AABB m_aabb;
    size_t m_dimension;
    size_t N_prev = 0;
    std::string initMethod = "Unifrom";
    float logLikelihood = 0.0f;

    float divergeProbability = 0.5;
    bool initialized = false;

    float totalSampleWeight(const std::vector<WeightedSample>& batch) {
        float sum = 0;
        for (const auto& sample : batch) {
            sum += sample.weight;
        }
        return sum;
    }

    void updateSufficientStatistics(const std::vector<WeightedSample>& batch, const Eigen::MatrixXd& responsibilities) {
        size_t NNew = batch.size();
        float totalWeight = totalSampleWeight(batch);
        size_t numComponents = components.size();

        std::vector<Eigen::VectorXd> newMeans(numComponents);
        std::vector<Eigen::Matrix3d> newCovariances(numComponents);
        std::vector<float> newWeights(numComponents);
        float totalSoftCount = 0.0;

        std::vector<std::thread> threads;

        for (size_t j = 0; j < numComponents; ++j) {
            threads.emplace_back([&, j]() {
                // SLog(mitsuba::EInfo, "Starting batch processing M-step");
                auto& comp = components[j];
                if (comp.getWeight() == 0)
                    return;

                size_t dim = comp.getMean().size();
                Eigen::VectorXd mean = Eigen::VectorXd::Zero(dim);
                Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(dim, dim);
                float responsibilitySum = 0.0;

                // Compute weighted responsibility sum - Hanebeck, Frisch
                for (size_t i = 0; i < NNew; ++i) {
                    float r = responsibilities(i, j);
                    if (r != r) {
                        SLog(mitsuba::EInfo, "R is none for %d, %d", i, j);
                        throw std::runtime_error("Responsibility cannot be nan");
                    }
                    responsibilitySum += r * batch[i].weight;
                    if (responsibilitySum != responsibilitySum) {
                        SLog(mitsuba::EInfo, "batch weight is: %f", batch[i].weight);
                        throw std::runtime_error("Responsibility cannot be nan");
                    }
                    mean += r * batch[i].point * batch[i].weight;
                }

                if (responsibilitySum < 1e-6f) {
                    SLog(mitsuba::EInfo, "Responsibility sum is below threshold");
                    return;
                }
                if (responsibilitySum != responsibilitySum) {
                    throw std::runtime_error("Responsibility cannot be nan");
                }

                mean /= responsibilitySum;

                for (size_t i = 0; i < NNew; ++i) {
                    Eigen::VectorXd diff = batch[i].point - comp.getMean();
                    cov += responsibilities(i, j) * batch[i].weight * (diff * diff.transpose());
                }

                cov /= responsibilitySum;

                float newWeight = responsibilitySum / totalWeight;
                assert(!std::isinf(newWeight));

                if (!mean.allFinite() || !cov.allFinite())
                    throw std::runtime_error("Mean or covariance is not finite during M-step");

                newMeans[j] = mean;
                newCovariances[j] = cov;
                newWeights[j] = newWeight;
            });
        }

        for (auto& t : threads) t.join();

        for (size_t j = 0;  j < numComponents; ++j) {
            if (components[j].getWeight() == 0)
                continue;
            components[j].updateComponent(N_prev, NNew, newMeans[j], newCovariances[j], newWeights[j], alpha);
            totalSoftCount += components[j].getSoftCount();
        }

        for (auto& c : components) {
            if (c.getWeight() != 0)
                c.setWeight(c.getSoftCount() / totalSoftCount);
        }

        N_prev = NNew; // update sample count
        SLog(mitsuba::EInfo, "sample count: %d", N_prev);

        mergeAllComponents();
        splitAllComponents();

        // prune too small weights
        float componentsWeight = 0;
        for (size_t i = 0; i < components.size(); ++i) {
            if (components[i].getWeight() < 1e-6f)
                deactivateComponent(i);
            else
                componentsWeight += components[i].getWeight();
        }
        for (auto& component : components)
            component.setWeight(component.getWeight() / componentsWeight);
    }

    void deactivateComponent(int idx) {
        components[idx].deactivate(m_dimension);
    }

    float bhattacharyyaDistance(int idx1, int idx2) {
        const auto& comp1 = components[idx1];
        const auto& comp2 = components[idx2];

        Eigen::VectorXd meanDiff = comp1.getMean() - comp2.getMean();

        Eigen::MatrixXd covMean = 0.5 * (comp1.getCovariance() + comp2.getCovariance());
        Eigen::MatrixXd covMeanInv = covMean.inverse();

        float cov1Det = comp1.getCovariance().determinant();
        float cov2Det = comp2.getCovariance().determinant();

        float multiply = meanDiff.transpose() * covMeanInv * meanDiff;
        float first =  multiply / 8;

        float second = 0.5 * log(covMean.determinant() / sqrt(cov1Det * cov2Det));

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
            }
        }

        size_t i, j;
        bhattacharyyaCoefficients.maxCoeff(&i, &j);
        float maxBC = bhattacharyyaCoefficients(i, j);
        SLog(mitsuba::EInfo, "components before the merge: %d", getNumActiveComponents());
        int mergeCount = 0;
        do {
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
        float minEigenvalue = eigenvalues.minCoeff();
        float maxEigenvalue = eigenvalues.maxCoeff();

        float ratio = maxEigenvalue / minEigenvalue;
        if (ratio < splittingThreshold)
            return; // below threshold - do nothing

        splitCount++;

        // Identify the principal eigenvector (largest eigenvalue)
        int maxIndex;
        eigenvalues.maxCoeff(&maxIndex);
        Eigen::VectorXd principalAxis = eigenvectors.col(maxIndex);

        float scalingFactor = 0.5 * std::sqrt(maxEigenvalue);
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
        float totalWeight = component1.getWeight() + component2.getWeight(); // we're just adding weights together

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

    float computeLogLikelihood(const std::vector<pmm_focal::WeightedSample>& batch) {
        size_t N = batch.size();

        float logLikelihoodNew = 0.0;

        for (size_t i = 0; i < N; ++i) {
            float sum = 0.0;
            for (size_t k = 0; k < components.size(); ++k) {
                if (components[k].getWeight() == 0)
                    continue;
                sum += components[k].getWeight() * componentPdf(components[k], batch[i].point);
            }
            if (sum > 0)
                logLikelihoodNew += std::log(sum);
        }

        return logLikelihoodNew;
    }

public:
    GaussianMixtureModel(): prng() {}
    GaussianMixtureModel(std::istream& in) {
        deserialize(in);
    }

    float getAlpha() { return alpha; }
    void setAlpha(float newAlpha) { alpha = newAlpha; };
    float getSplittingThreshold() { return splittingThreshold; };
    void setSplittingThreshold(float newThreshold) { splittingThreshold = newThreshold; };
    float getMergingThreshold() { return mergingThreshold; };
    void setMergingThreshold(float newThreshold) { mergingThreshold = newThreshold; };
    std::string getInitMethod() { return initMethod; }
    void setInitMethod(std::string newMethod) { initMethod = newMethod; };

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

    float getDivergeProbability() { return divergeProbability; }
    void setDivergeProbability(float newDivergeProbability) { divergeProbability = newDivergeProbability; }

    std::vector<GaussianComponent> getComponents() { return components; };

    void initComponentCovariance(GaussianComponent& comp) {
        comp.setCovariance(Eigen::MatrixXd::Identity(m_dimension, m_dimension));
    }

    float pdf(const mitsuba::Point3f x) {
        Eigen::VectorXd xEigen(3);
        xEigen << x.x, x.y, x.z;
        return pdf(xEigen);
    }

    float componentPdf(const GaussianComponent& component, const Eigen::VectorXd& x) {
        int d = x.size();
        float logNormConst = -0.5 * (d * std::log(2 * M_PI) + component.getLogDetCov());

        Eigen::VectorXd diff = x - component.getMean();
        float exponent = -0.5 * diff.transpose() * component.getInverseCovariance() * diff;
        float logPdf = logNormConst + exponent;

        return component.getWeight() * std::exp(logPdf);
    }

    float pdf(const Eigen::VectorXd& x) {
        float totalPdf = 0;
        float totalWeight = 0;

        for (const auto& component: components) {
            if (component.getWeight() < 1e-6f)
                continue;
            totalPdf += componentPdf(component, x);
            totalWeight += component.getWeight();
        }

        return totalPdf / totalWeight;
    }

    void init(const std::vector<WeightedSample>& batch) {
        SLog(mitsuba::EInfo, initMethod.c_str());
        if (initMethod == "KMeans") {
            initKMeans(batch);
            return;
        }

        if (initMethod == "Uniform") {
            initUniform(batch);
            return;
        }

        initRandom(batch);
    }

    void initUniform(const std::vector<WeightedSample>& batch) {
        SLog(mitsuba::EInfo, "Starting uniform initialization");

        m_dimension = batch[0].point.size();

        if (components.size() != maxNumComp)
            components.resize(maxNumComp);

        auto N = maxNumComp;
        auto length = std::abs(m_aabb.max.x - m_aabb.min.x);
        auto width = std::abs(m_aabb.max.y - m_aabb.min.y);
        auto height = std::abs(m_aabb.max.z - m_aabb.min.z);

         // Determine approximate grid size
        size_t nx = std::round(std::cbrt(N)); // Number of points along X
        size_t ny = std::round(std::sqrt(N / nx)); // Number of points along Y
        size_t nz = std::ceil((float)N / (nx * ny)); // Number of points along Z

        size_t newMaxNumComp = nx * ny * nz;
        if (newMaxNumComp> maxNumComp) {
            SLog(mitsuba::EInfo, "Increasing the number of max components to %d to allow uniform distribution", newMaxNumComp);
            maxNumComp = newMaxNumComp;
            components.resize(maxNumComp);
        }

        float dx = length / nx;
        float dy = width / ny;
        float dz = height / nz;

        size_t componentCounter = 0;

        for (size_t i = 0; i < nx; ++i) {
            for (size_t j = 0; j < ny; ++j) {
                for (size_t k = 0; k < nz; ++k) {
                    auto& component = components[componentCounter];
                    component.setWeight(1.0 / maxNumComp);
                    Eigen::VectorXd mean(3);
                    mean << m_aabb.min.x + (i + 0.5) * dx, m_aabb.min.y + (j + 0.5) * dy, m_aabb.min.z + (k + 0.5) * dz;
                    component.setMean(mean);
                    initComponentCovariance(component);
                    componentCounter++;
                }
            }
        }

        SLog(mitsuba::EInfo, this->toString().c_str());

        initialized = true;
    }

    void initRandom(const std::vector<WeightedSample>& batch) {
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
        SLog(mitsuba::EInfo, "Starting initialization using KMeans++");
        m_dimension = batch[0].point.size();

        if (components.size() != maxNumComp)
            components.resize(maxNumComp);

        std::vector<Eigen::VectorXd> centroids(maxNumComp, Eigen::VectorXd::Zero(m_dimension));
        std::vector<int> clusterAssignments(batch.size(), 0);
        std::vector<int> clusterSizes(maxNumComp, 0);

        // K-Means++ Initialization
        centroids[0] = batch[prng.getRandomNumber(0, batch.size() - 1)].point;
        for (size_t i = 1; i < maxNumComp; ++i) {
            std::vector<float> distances(batch.size(), std::numeric_limits<float>::max());

            for (size_t j = 0; j < batch.size(); ++j) {
                for (size_t c = 0; c < i; ++c) {
                    float dist = (batch[j].point - centroids[c]).squaredNorm();
                    distances[j] = std::min(distances[j], dist);
                }
            }

            std::discrete_distribution<int> distribution(distances.begin(), distances.end());
            int randomIndex = prng.getRandomNumber(0, batch.size() - 1);
            centroids[i] = batch[randomIndex].point;
        }

        bool converged = false;
        int maxIterations = 100;

        for (int iter = 0; iter < maxIterations && !converged; ++iter) {
            converged = true;
            std::fill(clusterSizes.begin(), clusterSizes.end(), 0);
            std::vector<Eigen::VectorXd> newCentroids(maxNumComp, Eigen::VectorXd::Zero(m_dimension));

            // Assign samples to the nearest centroid
            for (size_t i = 0; i < batch.size(); ++i) {
                float minDist = std::numeric_limits<float>::max();
                int bestCluster = 0;

                for (size_t j = 0; j < maxNumComp; ++j) {
                    float dist = (batch[i].point - centroids[j]).squaredNorm();
                    if (dist < minDist) {
                        minDist = dist;
                        bestCluster = j;
                    }
                }

                if (clusterAssignments[i] != bestCluster) {
                    clusterAssignments[i] = bestCluster;
                    converged = false;
                }

                newCentroids[bestCluster] += batch[i].point;
                clusterSizes[bestCluster]++;
            }

            // Update centroids
            for (size_t j = 0; j < maxNumComp; ++j) {
                if (clusterSizes[j] > 0) {
                    centroids[j] = newCentroids[j] / clusterSizes[j];
                } else {
                    // Reinitialize empty cluster centroid using a random sample
                    centroids[j] = batch[prng.getRandomNumber(0, batch.size() - 1)].point;
                }
            }
        }

        // Assign K-means results to GMM components
        for (size_t j = 0; j < maxNumComp; ++j) {
            auto& component = components[j];
            component.setWeight(static_cast<float>(clusterSizes[j]) / batch.size());
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
                covariance = Eigen::MatrixXd::Identity(m_dimension, m_dimension) * 1e-6f; // Small regularization
            }

            component.setCovariance(covariance);
        }

        SLog(mitsuba::EInfo, "Initialized using KMeans++");
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

    bool processInChunks(const std::vector<WeightedSample>& batch) {
        // we do not need to calculate full responsibilities matrix at once - such approach might cause bad_alloc for big batches
        // instead, we can loop over the batch in chunks while accumulating the sufficient statistics
        // for each component we need:
        //      - a sum of responsibilities: N_k = sum(r_ik)
        //      - mean: mean_k = 1/N_k * sum(r_ik * x_i)
        //      - covariance: covaraince_k = 1/N_k * sum(r_ik * (x_i - mean_k)(x_i - mean_k)^T)
        SLog(mitsuba::EInfo, "Starting processing in chunks");

        if (!initialized)
            init(batch);

        const size_t d = batch[0].point.size();
        const size_t chunkSize = 5000000;
        const size_t k = components.size();
        const size_t totalSamples = batch.size();

        // create component cache so we don't recompute this a bunch of times
        std::vector<ComponentCache> componentCache(k);

        std::vector<std::thread> cacheThreads;
        for (size_t j = 0; j < k; ++j) {
            cacheThreads.emplace_back([&, j]() {
                const auto& c = components[j];
                if (c.getWeight() == 0) return;
                componentCache[j] = {
                    c.getInverseCovariance(),
                    c.getMean(),
                    -0.5f * (batch[0].point.size() * std::log(2 * M_PI) + c.getLogDetCov()),
                    c.getWeight()
                };
            });
        }

        for (auto& t : cacheThreads) t.join();
        SLog(mitsuba::EInfo, "Finished component cache");

        // Allocate sufficient stats accumulators
        std::vector<double> N_k(k, 0.0);
        std::vector<Eigen::VectorXd> sum_x_k(k, Eigen::VectorXd::Zero(d));
        std::vector<Eigen::MatrixXd> sum_cov_k(k, Eigen::MatrixXd::Zero(d, d));

        for (size_t offset = 0; offset < totalSamples; offset += chunkSize) {
            SLog(mitsuba::EInfo, "Next chunk");
            size_t end = std::min(offset + chunkSize, totalSamples);
            size_t currentSize = end - offset;

            Eigen::MatrixXf R_chunk = Eigen::MatrixXf::Zero(currentSize, k);

            // E-step on chunk
            #pragma omp parallel for
            for (size_t i = 0; i < currentSize; ++i) {
                size_t globalIdx = offset + i;
                const auto& sample = batch[globalIdx];

                for (size_t j = 0; j < k; ++j) {
                    const auto& c = componentCache[j];
                    if (c.weight == 0) continue;
                    Eigen::VectorXd diff = sample.point - c.mean;
                    float exponent = -0.5 * diff.transpose() * c.invCov * diff;
                    float resp = c.weight * std::exp(c.logNormConst + exponent);
                    if (resp != resp)
                        SLog(mitsuba::EInfo, ("responsibility nan: " + sample.toString()).c_str());

                    R_chunk(i, j) = resp;
                }

                // Normalize row
                float rowSum = R_chunk.row(i).sum();
                if (rowSum > 0)
                    R_chunk.row(i) /= rowSum;
            }

            // Accumulate stats for chunk
            for (size_t i = 0; i < currentSize; ++i) {
                size_t globalIdx = offset + i;
                const auto& sample = batch[globalIdx];
                for (size_t j = 0; j < k; ++j) {
                    float r = R_chunk(i, j);
                    float w = sample.weight;
                    N_k[j] += r * w;
                    sum_x_k[j] += r * w * sample.point;
                }
            }

            for (size_t i = 0; i < currentSize; ++i) {
                size_t globalIdx = offset + i;
                const auto& sample = batch[globalIdx];
                for (size_t j = 0; j < k; ++j) {
                    float r = R_chunk(i, j);
                    float w = sample.weight;
                    Eigen::VectorXd mean_est = sum_x_k[j] / std::max(N_k[j], 1e-6);
                    Eigen::VectorXd diff = sample.point - mean_est;
                    sum_cov_k[j] += r * w * (diff * diff.transpose());
                }
            }
        }

        SLog(mitsuba::EInfo, "processed all chunks");

        // Final M-step: update components
        float totalWeight = totalSampleWeight(batch);
        float totalSoftCount = 0.0;
        for (size_t j = 0; j < k; ++j) {
            if (components[j].getWeight() == 0 || N_k[j] < 1e-6)
                continue;

            Eigen::VectorXd newMean = sum_x_k[j] / N_k[j];
            Eigen::MatrixXd newCov = sum_cov_k[j] / N_k[j];
            float newWeight = N_k[j] / totalWeight;

            components[j].updateComponent(N_prev, totalSamples, newMean, newCov, newWeight, alpha);
            totalSoftCount += components[j].getSoftCount();
        }

        for (auto& c : components) {
            if (c.getWeight() != 0)
                c.setWeight(c.getSoftCount() / totalSoftCount);
        }

        N_prev = totalSamples;

        splitAllComponents();
        mergeAllComponents();

        // Prune
        float componentsWeight = 0;
        for (size_t i = 0; i < components.size(); ++i) {
            if (components[i].getWeight() < 1e-6f)
                deactivateComponent(i);
            else
                componentsWeight += components[i].getWeight();
        }
        for (auto& component : components)
            component.setWeight(component.getWeight() / componentsWeight);

        auto logLikelihoodNew = computeLogLikelihood(batch);
        auto diff = std::abs(logLikelihood - logLikelihoodNew);
        SLog(mitsuba::EInfo, "logLikelihoodOld: %f, logLIkelihoodNew: %f, diff: %f", logLikelihood, logLikelihoodNew, diff);
        logLikelihood = logLikelihoodNew;

        return diff < 1e-6f;
    }

    void processBatchParallel(const std::vector<WeightedSample>& batch) {
        SLog(mitsuba::EInfo, "Starting batch processing in parallel");

        if (!initialized)
            init(batch);

        // create component cache so we don't recompute this a bunch of times
        std::vector<ComponentCache> componentCache(components.size());

        std::vector<std::thread> cacheThreads;
        for (size_t j = 0; j < components.size(); ++j) {
            cacheThreads.emplace_back([&, j]() {
                const auto& c = components[j];
                if (c.getWeight() == 0) return;
                componentCache[j] = {
                    c.getInverseCovariance(),
                    c.getMean(),
                    -0.5f * (batch[0].point.size() * std::log(2 * M_PI) + c.getLogDetCov()),
                    c.getWeight()
                };
            });
        }

        for (auto& t : cacheThreads) t.join();
        SLog(mitsuba::EInfo, "Finished component cache");

        // E-step
        Eigen::MatrixXd responsibilities(batch.size(), components.size());


        SLog(mitsuba::EInfo, "Allocated");

        size_t numThreads = std::thread::hardware_concurrency();
        size_t batchSize = batch.size();
        size_t chunkSize = (batchSize + numThreads - 1) / numThreads;

        std::vector<std::thread> threads;

        for (size_t threadIdx = 0; threadIdx < numThreads; ++threadIdx) {
            size_t startIdx = threadIdx * chunkSize;
            size_t endIdx = std::min(startIdx + chunkSize, batchSize);

            threads.emplace_back([&, startIdx, endIdx]() {
                // SLog(mitsuba::EInfo, "Starting batch processing E-step");
                for (size_t i = startIdx; i < endIdx; ++i) {
                    for (size_t j = 0; j < components.size(); ++j) {
                        const auto& c = componentCache[j];
                        if (c.weight == 0) continue;

                        Eigen::VectorXd diff = batch[i].point - c.mean;
                        float exponent = -0.5 * diff.transpose() * c.invCov * diff;
                        float resp = c.weight * std::exp(c.logNormConst + exponent);

                        if (std::isnan(resp)) {
                            SLog(mitsuba::EInfo, "point (%f, %f, %f), weight %f",
                                batch[i].point[0], batch[i].point[1], batch[i].point[2], batch[i].weight);
                        }

                        responsibilities(i, j) = resp;
                    }
                }
            });
        }

        for (auto& t : threads) t.join();
        SLog(mitsuba::EInfo, "Finish E step");

        // normalize responsibilities
        std::vector<std::thread> normThreads;
        for (size_t threadIdx = 0; threadIdx < numThreads; ++threadIdx) {
            size_t startIdx = threadIdx * chunkSize;
            size_t endIdx = std::min(startIdx + chunkSize, batchSize);

            normThreads.emplace_back([&, startIdx, endIdx]() {
                for (size_t i = startIdx; i < endIdx; ++i) {
                    float rowSum = responsibilities.row(i).sum();
                    if (rowSum > 0)
                        responsibilities.row(i) /= rowSum;
                }
            });
        }

        for (auto& t : normThreads) t.join();

        // M-step
        updateSufficientStatistics(batch, responsibilities);

        SLog(mitsuba::EInfo, "Finish M step");
    }

    void processBatch(const std::vector<WeightedSample>& batch) {
        SLog(mitsuba::EInfo, "Start processing batch");
        if (!initialized)
            init(batch);


        Eigen::MatrixXd responsibilities = Eigen::MatrixXd(batch.size(), components.size());

        // E-step
        {
            for (size_t j = 0; j < components.size(); ++j) {
                auto& component = components[j];
                if (component.getWeight() == 0)
                    continue;

                // instead of calling pdf, we use eigen vector operations to make this quicker
                Eigen::MatrixXd invCov = component.getInverseCovariance();
                Eigen::VectorXd mean = component.getMean();
                float logNormConst = -0.5 * (batch[0].point.size() * std::log(2 * M_PI) + component.getLogDetCov());
                float weight = component.getWeight();
                for (size_t i = 0; i < batch.size(); ++i) {
                    Eigen::VectorXd diff = batch[i].point - mean;
                    float exponent = -0.5 * diff.transpose() * invCov * diff;
                    auto resp = weight * std::exp(logNormConst + exponent);
                    if (resp != resp) {
                        SLog(mitsuba::EInfo, "point (%f, %f, %f), weight %f", batch[i].point[0], batch[i].point[1], batch[i].point[2], batch[i].weight);
                    }
                    responsibilities(i, j) = resp; // without * batch[i].weight; - Hanebeck, Frisch - consider weights only in M step - this is an absolute game changer!
                }
            }
        }

        SLog(mitsuba::EInfo, "Finish E step");

        // Normalize responsibilities
        for (int i = 0; i < responsibilities.rows(); ++i) { // rows is batch.size()
            responsibilities.row(i) /= responsibilities.row(i).sum();
        }
        SLog(mitsuba::EInfo, "Normalized responsibilities");

        // M-step
        {
            updateSufficientStatistics(batch, responsibilities);
        }

        SLog(mitsuba::EInfo, "Finish M step");
    }

    // using Inverse Transform Sampling
    Eigen::VectorXd sample(mitsuba::RadianceQueryRecord &rRec) const {
        // Compute total weight
        float totalWeight = 0;
        for (const auto& component : components) {
            totalWeight += component.getWeight();
        }

        // Sample a component directly using a cumulative sum
        float randWeight = rRec.nextSample1D() * totalWeight;
        float cumulativeWeight = 0;

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
        // out->write(reinterpret_cast<const char*>(&N_prev), sizeof(N_prev));
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
        // in->read(reinterpret_cast<char*>(&N_prev), sizeof(N_prev));
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