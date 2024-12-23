// based on A Fast Incremental Gaussian Mixture Model - Rafael Pinto, Paulo Engel
#include <iostream>
#include <vector>
#include <cmath>
#include <eigen3/Eigen/Dense>

using namespace Eigen;
using namespace std;

class GaussianComponent {
public:
    VectorXd mean;
    MatrixXd precisionMatrix;
    double determinant;
    double weight;

    GaussianComponent(int dimension, double initialVariance) {
        mean = VectorXd::Zero(dimension);
        precisionMatrix = MatrixXd::Identity(dimension, dimension) / initialVariance;
        determinant = pow(initialVariance, -dimension);
        weight = 1.0;
    }
};

class OnlineGMM {
public:
    vector<GaussianComponent> components;
    double learningRate;
    double chiSquaredThreshold;

    OnlineGMM(double initialVariance, double learningRate, double chiSquaredThreshold)
        : learningRate(learningRate), chiSquaredThreshold(chiSquaredThreshold) {
        initialVariance_ = initialVariance;
    }

    void addComponent(const VectorXd& x) {
        int dimension = x.size();
        GaussianComponent newComponent(dimension, initialVariance_);
        newComponent.mean = x;
        components.push_back(newComponent);
    }

    void updateComponent(GaussianComponent& component, const VectorXd& x) {
        VectorXd diff = x - component.mean;
        double responsibility = this->computeResponsibility(component, x);

        // Update weight
        component.weight = (1 - learningRate) * component.weight + learningRate * responsibility;

        // Update mean
        VectorXd deltaMean = learningRate * responsibility * diff;
        component.mean += deltaMean;

        // Update precision matrix using Sherman-Morrison formula
        MatrixXd outerProd = diff * diff.transpose();
        MatrixXd adjustment = (component.precisionMatrix * outerProd * component.precisionMatrix) /
                              (1.0 + (diff.transpose() * component.precisionMatrix * diff)(0));
        component.precisionMatrix -= adjustment;

        // Update determinant using rank-one update formula
        double adjustmentFactor = 1.0 - (diff.transpose() * component.precisionMatrix * diff)(0);
        component.determinant *= adjustmentFactor;
    }

    void process(const VectorXd& x) {
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

private:
    double initialVariance_;

    double computeMahalanobisDistance(const GaussianComponent& component, const VectorXd& x) {
        VectorXd diff = x - component.mean;
        return diff.transpose() * component.precisionMatrix * diff;
    }

    double computeResponsibility(const GaussianComponent& component, const VectorXd& x) {
        double exponent = -0.5 * computeMahalanobisDistance(component, x);
        double normalization = pow(2 * M_PI, -x.size() / 2.0) * sqrt(component.determinant);
        return exp(exponent) / normalization;
    }
};

int main() {
    int dimension = 2;
    OnlineGMM gmm(1.0, 0.01, 5.991); // 95% confidence interval for chi-squared with 2 DOF

    vector<VectorXd> data = {
        VectorXd::Random(dimension),
        VectorXd::Random(dimension),
        VectorXd::Random(dimension)};

    for (const auto& x : data) {
        gmm.process(x);
    }

    cout << "Number of components: " << gmm.components.size() << endl;
    for (const auto& component : gmm.components) {
        cout << "Mean: \n" << component.mean << endl;
        cout << "Precision Matrix: \n" << component.precisionMatrix << endl;
        cout << "Weight: " << component.weight << endl;
    }

    return 0;
}
