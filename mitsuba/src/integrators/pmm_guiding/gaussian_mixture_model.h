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
#include <mitsuba/core/logger.h>

#include "distribution.h"
#include "util.h"

#define FAIL_ON_ZERO_CDF 0
#define USE_MAX_KEEP 0

namespace std {
    template<>
    struct hash<mitsuba::Point3f> {
        std::size_t operator()(const mitsuba::Point3f &p) const {
            size_t h1 = std::hash<float>()(p.x);
            size_t h2 = std::hash<float>()(p.y);
            size_t h3 = std::hash<float>()(p.z);
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };
}

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
    float alpha = 0.25;
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
    size_t n = 0;
    // thread_local static std::unordered_map<Eigen::Vector3f, float, VectorHash> gmmPdfCache;

    float totalSampleWeight(const std::vector<WeightedSample>& batch) {
        float sum = 0;
        for (const auto& sample : batch) {
            sum += sample.weight;
        }
        return sum;
    }

    void computeResponsibilities(const std::vector<WeightedSample>& batch, Eigen::MatrixXd& responsibilities) {
        const size_t N = batch.size();
        const size_t K = components.size();
        const size_t d = batch[0].point.size();

        responsibilities.setZero(N, K);

        std::vector<ComponentCache> cache(K);
        #pragma omp parallel for
        for (int j = 0; j < (int)K; ++j) {
            const auto& c = components[j];
            if (c.getWeight() == 0) continue;
                cache[j] = {
                c.getInverseCovariance(),
                c.getMean(),
                -0.5f * (d * std::log(2 * M_PI) + c.getLogDetCov()),
                c.getWeight()
            };
        }

        #pragma omp parallel for
        for (int i = 0; i < (int)N; ++i) {
            const auto& x = batch[i].point;
            float sum = 0.0f;
            for (size_t j = 0; j < K; ++j) {
                if (cache[j].weight == 0) continue;
                Eigen::VectorXd diff = x - cache[j].mean;
                float logp = cache[j].logNormConst - 0.5f * (diff.transpose() * cache[j].invCov * diff)(0);
                float resp = cache[j].weight * std::exp(logp);
                responsibilities(i, j) = resp;
                sum += resp;
            }
            if (sum > 0)
                responsibilities.row(i) /= sum;
        }
        SLog(mitsuba::EInfo, "computed responsibilities");
    }


    void updateSufficientStatistics(const std::vector<WeightedSample>& batch, const Eigen::MatrixXd& responsibilities) {
        const size_t NNew = batch.size();
        const float totalWeight = totalSampleWeight(batch);
        const size_t numComponents = components.size();
        const float decay = 0.25;

        std::vector<std::thread> threads;

        for (size_t j = 0; j < numComponents; ++j) {
            threads.emplace_back([&, j]() {
                // SLog(mitsuba::EInfo, "Starting batch processing M-step");
                auto& comp = components[j];
                if (comp.getWeight() == 0)
                    return;

                size_t dim = comp.getMean().size();
                Eigen::VectorXd sum_x = Eigen::VectorXd::Zero(dim);
                Eigen::MatrixXd sum_xxT = Eigen::MatrixXd::Zero(dim, dim);
                float responsibilitySum = 0.0f;

                // Compute weighted responsibility sum - Hanebeck, Frisch
                for (size_t i = 0; i < NNew; ++i) {
                    float r = responsibilities(i, j);
                    if (r != r) {
                        SLog(mitsuba::EInfo, "R is none for %d, %d", i, j);
                        throw std::runtime_error("Responsibility cannot be nan");
                    }
                    float w = batch[i].weight;
                    const Eigen::VectorXd& x = batch[i].point;

                    responsibilitySum += r * w;
                    sum_x += r * w * x;
                    sum_xxT += r * w * (x * x.transpose());
                }

                if (responsibilitySum != responsibilitySum) {
                    throw std::runtime_error("Responsibility cannot be nan");
                }

                if (responsibilitySum < 1e-6f)
                    return; // skip component update if soft count is too low

                comp.updateComponentWithSufficientStatistics(sum_x, sum_xxT, responsibilitySum, totalWeight, decay);
            });
        }

        for (auto& t: threads)
            t.join();

        float totalSoft = 0.f;
        for (auto& comp : components)
            totalSoft += comp.r_k;

        for (auto& comp : components) {
            if (comp.r_k > 0)
                comp.setWeight(comp.r_k / totalSoft);
        }

        N_prev = NNew;

        SLog(mitsuba::EInfo, this->toString().c_str());

        SLog(mitsuba::EInfo, "updating responsibiliites");
        Eigen::MatrixXd updatedResponsibilities(batch.size(), components.size());
        computeResponsibilities(batch, updatedResponsibilities);

        SLog(mitsuba::EInfo, "Splitting and merging");

        splitAllComponents(batch, updatedResponsibilities);
        mergeAllComponents();

        // prune too small weights
        float componentsWeight = 0;
        for (size_t i = 0; i < components.size(); ++i) {
            if (components[i].getWeight() < 1e-6f)
                deactivateComponent(i);
            else
                componentsWeight += components[i].getWeight();
        }

        for (auto& component : components) {
            component.setWeight(component.getWeight() / componentsWeight);
            component.isNew = false;
        }
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

        if (!std::isfinite(first+second) || (first + second) == 0) {
            SLog(mitsuba::EInfo, comp1.toString().c_str());
            SLog(mitsuba::EInfo, comp2.toString().c_str());
            SLog(mitsuba::EInfo, "multiply: %f, cov1Det: %f, cov2Det: %f", multiply, cov1Det, cov2Det);
        }

        return first + second;
    }

    void mergeAllComponents() {
        int maxMergeCount = maxNumComp; // todo: think through / parametrize
        // table of bhattacharyya coefficients between all components
        // the closer BC is to 1 the more similar the distributions are
        Eigen::MatrixXd bhattacharyyaCoefficients(components.size(), components.size());
        for (size_t i = 0; i < components.size(); ++i) {
            for (size_t j = i + 1; j < components.size(); ++j) {
                if (components[j].getWeight() == 0 || components[i].getWeight() == 0 || j == i || components[i].isNew || components[j].isNew) {
                    bhattacharyyaCoefficients(i, j) = 0;
                    bhattacharyyaCoefficients(j, i) = 0;
                    continue;
                }
                auto dist = exp(-bhattacharyyaDistance(i, j)); // BC = 1/exp(BD)
                bhattacharyyaCoefficients(i, j) = dist;
                bhattacharyyaCoefficients(j, i) = dist;
                if (!std::isfinite(dist)) {
                    SLog(mitsuba::EInfo, "%d, %d", i, j);
                }
            }
        }
        bhattacharyyaCoefficients.diagonal().setZero();

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
            SLog(mitsuba::EInfo, "merge maxBC = %f", maxBC);
        } while(maxBC > mergingThreshold && mergeCount < maxMergeCount);
        SLog(mitsuba::EInfo, "components after the merge: %d, merged %d component", getNumActiveComponents(), mergeCount);
    }

    float computeChiSquareDivergence(
        size_t k,
        const std::vector<WeightedSample>& batch,
        const Eigen::MatrixXd& responsibilities,
        size_t topN = 100
    ) {
        const auto& comp = components[k];
        if (comp.getWeight() == 0) return 0.0f;

        const Eigen::VectorXd& mu = comp.getMean();
        const Eigen::MatrixXd& cov = comp.getCovariance();
        const Eigen::MatrixXd& invCov = comp.getInverseCovariance();
        float logDet = comp.getLogDetCov();
        const size_t d = mu.size();

        std::vector<std::pair<size_t, float>> weightedIndices;
        for (size_t i = 0; i < batch.size(); ++i) {
            float resp = responsibilities(i, k);
            if (resp > 1e-10f)
                weightedIndices.emplace_back(i, resp);
        }

        if (topN > 0 && weightedIndices.size() > topN) {
            std::partial_sort(weightedIndices.begin(), weightedIndices.begin() + topN, weightedIndices.end(),
                              [](const std::pair<size_t, float>& a, const std::pair<size_t, float>& b) { return a.second > b.second; });
            weightedIndices.resize(topN);
        }

        float totalWeight = 0.0f;
        for (const auto& weightedIdx : weightedIndices)
            totalWeight += weightedIdx.second;

        if (totalWeight < 1e-6f) return 0.0f;

        float divergence = 0.0f;
        for (const auto& weightedIdx : weightedIndices) {
            const auto& idx = weightedIdx.first;
            const auto& resp = weightedIdx.second;
            const auto& x = batch[idx].point;
            Eigen::VectorXd diff = x - mu;

            // Normalized Gaussian density
            float logG = -0.5f * (diff.transpose() * invCov * diff)(0)
                         - 0.5f * (d * std::log(2 * M_PI) + logDet);
            float g = std::exp(logG);

            float p = resp / totalWeight;

            if (g > 1e-10f) {
                divergence += (p - g) * (p - g) / g;
            }
        }

        return divergence;
    }


    void splitAllComponents(const std::vector<WeightedSample>& batch, const Eigen::MatrixXd& responsibilities) {
        SLog(mitsuba::EInfo, "components before the split: %d", getNumActiveComponents());
        if (getNumActiveComponents() >= maxNumComp) return;

        for (size_t i = 0; i < components.size(); ++i) {
            if (getNumActiveComponents() >= maxNumComp) break;
            float jsplit = computeChiSquareDivergence(i, batch, responsibilities, 0);
            SLog(mitsuba::EInfo, "split score: %f", jsplit);
            if (jsplit < splittingThreshold) continue;
            splitComponent(i);
        }

        SLog(mitsuba::EInfo, "components after the split: %d", getNumActiveComponents());
    }

    void splitComponent(size_t index) {
        if (getNumActiveComponents() >= maxNumComp) return;

        // using PCA
        auto cov = components[index].getCovariance();
        auto mean = components[index].getMean();

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(cov);
        Eigen::VectorXd eigenvalues = solver.eigenvalues();
        Eigen::MatrixXd eigenvectors = solver.eigenvectors();

        float maxEigenvalue = eigenvalues.maxCoeff();

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
        components[index].isNew = true;

        GaussianComponent newComp;
        newComp.setWeight(newWeight);
        newComp.setMean(newMean2);
        newComp.setCovariance(newCov);

        for (size_t i=0; i < components.size(); ++i) {
            if (components[i].getWeight() == 0) {
                // found first deactivated component - replace with the new one
                components[i] = newComp;
                components[i].isNew = true;
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

        Eigen::MatrixXd correction1 = (component1.getMean() - mergedMean) * (component1.getMean() - mergedMean).transpose();
        Eigen::MatrixXd correction2 = (component2.getMean() - mergedMean) * (component2.getMean() - mergedMean).transpose();

        Eigen::MatrixXd mergedConvs = component1.getWeight() * (component1.getCovariance() + correction1) / totalWeight + component2.getWeight() * (component2.getCovariance() + correction2) / totalWeight;

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

    float pdf(const mitsuba::Point3f &x) {
        Eigen::VectorXd xEigen(3);
        xEigen << x.x, x.y, x.z;
        auto val = pdf(xEigen);
        return val;
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
        // auto it = gmmPdfCache.find(x);
        //     if (it != gmmPdfCache.end())
        //         return it->second;
        float totalPdf = 0;

        for (const auto& component: components) {
            if (component.getWeight() < 1e-6f)
                continue;
            totalPdf += componentPdf(component, x);
        }
        // gmmPdfCache[x] = totalPdf;

        return totalPdf;
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

        size_t initNumComp = maxNumComp;

        std::vector<Eigen::VectorXd> centroids;
        centroids.reserve(initNumComp);

        std::vector<int> clusterAssignments(batch.size(), -1);
        std::random_device rd;
        std::mt19937 gen(rd());

        // --- KMeans++ Initialization ---
        // 1. First centroid picked randomly
        std::uniform_int_distribution<> uniDist(0, batch.size() - 1);
        centroids.push_back(batch[uniDist(gen)].point);

        for (size_t i = 1; i < initNumComp; ++i) {
            std::vector<double> distances(batch.size(), std::numeric_limits<double>::max());

            for (size_t j = 0; j < batch.size(); ++j) {
                for (size_t c = 0; c < centroids.size(); ++c) {
                    double dist = (batch[j].point - centroids[c]).squaredNorm();
                    distances[j] = std::min(distances[j], dist);
                }
            }

            std::discrete_distribution<int> weightedDist(distances.begin(), distances.end());
            centroids.push_back(batch[weightedDist(gen)].point);
        }

        // --- One iteration of KMeans (assign points and compute means) ---
        std::vector<int> clusterSizes(initNumComp, 0);
        std::vector<Eigen::VectorXd> newCentroids(initNumComp, Eigen::VectorXd::Zero(m_dimension));

        for (size_t i = 0; i < batch.size(); ++i) {
            double minDist = std::numeric_limits<double>::max();
            int bestCluster = -1;

            for (size_t k = 0; k < initNumComp; ++k) {
                double dist = (batch[i].point - centroids[k]).squaredNorm();
                if (dist < minDist) {
                    minDist = dist;
                    bestCluster = static_cast<int>(k);
                }
            }

            clusterAssignments[i] = bestCluster;
            clusterSizes[bestCluster]++;
            newCentroids[bestCluster] += batch[i].point;
        }

        for (size_t k = 0; k < initNumComp; ++k) {
            if (clusterSizes[k] > 0)
                newCentroids[k] /= clusterSizes[k];
            else
                newCentroids[k] = centroids[k]; // fallback
        }

        // --- Initialize GMM components from centroids ---
        for (size_t k = 0; k < initNumComp; ++k) {
            auto& component = components[k];
            component.setWeight(static_cast<float>(clusterSizes[k]) / batch.size());
            component.setMean(newCentroids[k]);

            Eigen::MatrixXd covariance = Eigen::MatrixXd::Zero(m_dimension, m_dimension);

            if (clusterSizes[k] > 1) {
                for (size_t i = 0; i < batch.size(); ++i) {
                    if (clusterAssignments[i] == static_cast<int>(k)) {
                        Eigen::VectorXd diff = batch[i].point - newCentroids[k];
                        covariance += diff * diff.transpose();
                    }
                }
                covariance /= (clusterSizes[k] - 1);
            } else {
                covariance = Eigen::MatrixXd::Identity(m_dimension, m_dimension) * 1e-6;
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
        size_t dims = component.getCovariance().rows();
        component.setCovariance(Eigen::MatrixXd::Identity(dims, dims));
    }

    void processBatchParallel(std::vector<WeightedSample>& batch) {
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

        SLog(mitsuba::EInfo, "Normalized responibilities");

        // M-step
        updateSufficientStatistics(batch, responsibilities);

        SLog(mitsuba::EInfo, "Finish M step");

        // for (const auto& sample : batch)
        //     SLog(mitsuba::EInfo, "point (%f, %f, %f), weight %f", sample.point[0], sample.point[1], sample.point[2], sample.weight);
    }

    void processMegaBatch(std::vector<WeightedSample>& megaBatch, size_t subBatchSize = 100000) {
        SLog(mitsuba::EInfo, "Processing mega batch of size %zu", megaBatch.size());

        size_t numSubBatches = (megaBatch.size() + subBatchSize - 1) / subBatchSize;
        for (size_t i = 0; i < numSubBatches; ++i) {
            size_t startIdx = i * subBatchSize;
            size_t endIdx = std::min(startIdx + subBatchSize, megaBatch.size());
            std::vector<WeightedSample> subBatch(megaBatch.begin() + startIdx, megaBatch.begin() + endIdx);

            SLog(mitsuba::EInfo, "Processing sub-batch %zu/%zu (%zu samples)", i + 1, numSubBatches, subBatch.size());

            processBatchParallel(subBatch);
        }
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

    void serialize(mitsuba::FileStream* out) const {}

    void deserialize(mitsuba::FileStream* in) {}

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}

#endif /* __MIXTURE_MODEL_H */