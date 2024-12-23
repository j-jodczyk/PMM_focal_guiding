// based on Online Expectation-Maximisation∗ - Olivier Cappe
#include <iostream>
#include <vector>
#include <cmath>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

class GaussianComponent {
public:
    VectorXd mean;
    MatrixXd covariance;
    double weight;

    GaussianComponent(int dimension) {
        mean = VectorXd::Zero(dimension);
        covariance = MatrixXd::Identity(dimension, dimension);
        weight = 1.0;
    }
};

class OnlineEMGMM {
public:
    vector<GaussianComponent> components;
    double stepSizeAlpha;

    OnlineEMGMM(int numComponents, int dimension, double stepSizeAlpha)
        : stepSizeAlpha(stepSizeAlpha) {
        for (int i = 0; i < numComponents; ++i) {
            components.emplace_back(dimension);
        }
    }

    void processObservation(const VectorXd &observation) {
        int dimension = observation.size();

        // Step 1: Compute responsibilities (E-step)
        VectorXd responsibilities = computeResponsibilities(observation);

        // Step 2: Update parameters (M-step)
        for (int k = 0; k < components.size(); ++k) {
            GaussianComponent &comp = components[k];
            double gamma = responsibilities(k);

            // Update weight
            comp.weight = (1.0 - stepSizeAlpha) * comp.weight + stepSizeAlpha * gamma;

            // Update mean
            VectorXd meanDiff = observation - comp.mean;
            comp.mean += stepSizeAlpha * gamma * meanDiff;

            // Update covariance
            MatrixXd outerProd = meanDiff * meanDiff.transpose();
            comp.covariance = (1.0 - stepSizeAlpha) * comp.covariance + stepSizeAlpha * gamma * outerProd;
        }
    }

    void printComponents() const {
        for (int k = 0; k < components.size(); ++k) {
            const GaussianComponent &comp = components[k];
            cout << "Component " << k + 1 << ":" << endl;
            cout << "  Weight: " << comp.weight << endl;
            cout << "  Mean: \n" << comp.mean << endl;
            cout << "  Covariance: \n" << comp.covariance << endl;
        }
    }

private:
    VectorXd computeResponsibilities(const VectorXd &observation) {
        int numComponents = components.size();
        VectorXd responsibilities(numComponents);
        VectorXd likelihoods(numComponents);
        double totalLikelihood = 0.0;

        for (int k = 0; k < numComponents; ++k) {
            const GaussianComponent &comp = components[k];
            double likelihood = computeGaussianLikelihood(observation, comp);
            likelihoods(k) = comp.weight * likelihood;
            totalLikelihood += likelihoods(k);
        }

        responsibilities = likelihoods / totalLikelihood;
        return responsibilities;
    }

    double computeGaussianLikelihood(const VectorXd &x, const GaussianComponent &comp) {
        int d = x.size();
        MatrixXd invCov = comp.covariance.inverse();
        double detCov = comp.covariance.determinant();
        VectorXd diff = x - comp.mean;

        double exponent = -0.5 * diff.transpose() * invCov * diff;
        double normalization = pow(2 * M_PI, -d / 2.0) * pow(detCov, -0.5);

        return normalization * exp(exponent);
    }
};

int main() {
    int numComponents = 2;
    int dimension = 2;
    double stepSizeAlpha = 0.1;

    OnlineEMGMM gmm(numComponents, dimension, stepSizeAlpha);

    vector<VectorXd> data = {
        VectorXd::Random(dimension),
        VectorXd::Random(dimension),
        VectorXd::Random(dimension)};

    for (const auto &observation : data) {
        gmm.processObservation(observation);
    }

    gmm.printComponents();

    return 0;
}
