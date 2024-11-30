/* Simple program for testing how well GMM fits to data */
/* TODO: doesn't work anymore - fix in free time */
#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <string>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "../include/matplotlibcpp.h"

#include "../../../mitsuba/src/integrators/pmm_guiding/gaussian_mixture_model.h"
#include "../../../mitsuba/src/integrators/pmm_guiding/gaussian_component.h"
#include "../../../mitsuba/src/integrators/pmm_guiding/envs/3d_env.h"
#include "../../../mitsuba/src/integrators/pmm_guiding/envs/2d_env.h"

namespace plt = matplotlibcpp;
using Scalar = double;

// constexpr int dims = 3;
constexpr int dims = 2;
constexpr int components = 4;

using Vectord = Eigen::Matrix<Scalar, dims, 1>;
using Matrixd = Eigen::Matrix<Scalar, dims, Eigen::Dynamic>;

using SampleVector = Eigen::Matrix<Scalar, dims, Eigen::Dynamic>;

Matrixd generateGaussianSamples(const Vectord& mean, const Eigen::Matrix<Scalar, dims, dims>& cov, int num_samples) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0, 1);

    Matrixd samples(dims, num_samples);
    Eigen::LLT<Eigen::Matrix<Scalar, dims, dims>> llt(cov);
    Eigen::Matrix<Scalar, dims, dims> transform = llt.matrixL();

    for (int i = 0; i < num_samples; ++i) {
        Vectord sample;
        sample << dist(gen), dist(gen);
        samples.col(i) = mean + transform * sample;
    }
    return samples;
}

// copied from gaussian component for debugging
Scalar pdf(const Vectord& sample, const Vectord& mean, const Matrixd& covariance) {
    constexpr Scalar epsilon = std::numeric_limits<Scalar>::epsilon();
    const Scalar twoPi = static_cast<Scalar>(2.0 * M_PI);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, dims, dims>> eigenSolver(covariance);
    Eigen::Matrix<Scalar, dims, dims> stableCovariance = covariance;
    for (int i = 0; i < dims; ++i) {
        if (eigenSolver.eigenvalues()[i] < epsilon)
            stableCovariance(i, i) += epsilon;
    }
    Eigen::Matrix<Scalar, dims, dims> covarianceInv = stableCovariance.inverse();
    Scalar detCovariance = stableCovariance.determinant();

    std::cout << detCovariance << std::endl;

    Vectord diff = sample - mean;
    Scalar mahalanobisDist = diff.transpose() * covarianceInv * diff;
    std::cout << mahalanobisDist << std::endl;

    Scalar normalization = std::pow(twoPi, static_cast<Scalar>(-dims / 2.0)) * std::pow(detCovariance, static_cast<Scalar>(-0.5));
    std::cout << normalization << std::endl;

    Scalar pdfValue = normalization * std::exp(static_cast<Scalar>(-0.5) * mahalanobisDist);
    return pdfValue;
}

void plotGMM3D(
    const Eigen::Matrix<double, 3, Eigen::Dynamic>& samples,
    pmm_focal::GaussianMixtureModel<3, components, Scalar, pmm_focal::GaussianComponent, Env3D>& gmm,
    int figure_index) {

    // plt::figure(figure_index); // Create a new figure for each plot

    // Extract sample points
    std::vector<double> sample_x, sample_y, sample_z;
    for (int i = 0; i < samples.cols(); ++i) {
        sample_x.push_back(samples(0, i));
        sample_y.push_back(samples(1, i));
        sample_z.push_back(samples(2, i));
    }

    // Scatter plot for sample points
    plt::scatter(sample_x, sample_y, sample_z, 10.0, {{"label", "Samples"}});

    for (int k = 0; k < gmm.components().size(); ++k) {
        auto mean = gmm.getComponentMean(k);
        auto cov = gmm.getComponentCovariance(k);

        // Generate points for the ellipsoid
        std::vector<double> ellipsoid_x, ellipsoid_y, ellipsoid_z;

        // Create a sphere and transform it into an ellipsoid
        for (double phi = 0; phi < M_PI; phi += 0.1) {
            for (double theta = 0; theta < 2 * M_PI; theta += 0.1) {
                // Points on the unit sphere
                Eigen::Vector3d unit_point(
                    std::sin(phi) * std::cos(theta),
                    std::sin(phi) * std::sin(theta),
                    std::cos(phi)
                );

                // Transform unit sphere into an ellipsoid
                Eigen::Vector3d ellipsoid_point = mean + cov.llt().matrixL() * unit_point;

                // Collect transformed points
                ellipsoid_x.push_back(ellipsoid_point(0));
                ellipsoid_y.push_back(ellipsoid_point(1));
                ellipsoid_z.push_back(ellipsoid_point(2));
            }
        }

        // Use scatter3 to simulate a surface for the ellipsoid
        plt::scatter(ellipsoid_x, ellipsoid_y, ellipsoid_z, 1.0,
                      {{"label", "Cluster " + std::to_string(k)}, {"color", "red"}});
    }
    // Set plot labels and display
    plt::xlabel("X");
    plt::ylabel("Y");
    // plt::zlabel("Z");
    plt::title("Gaussian Mixture Model Fitting (3D) - Iteration " + std::to_string(figure_index));
    plt::legend();

}


int main() {
    std::cout<<"Begin program"<<std::endl;
    // pmm_focal::GaussianMixtureModel<3, 4, Scalar, pmm_focal::GaussianComponent, Env3D> gmm;
    pmm_focal::GaussianMixtureModel<2, 4, Scalar, pmm_focal::GaussianComponent, Env2D> gmm;
    std::cout<<"GMM created"<<std::endl;

    // [min=[-21.7494, -3.70393, -3.09172], max=[7.19249, 11.2965, 6.96201]
    // Env3D::AABB aabb = { {-21.7494, -3.70393, -3.09172}, {7.19249, 11.2965, 6.96201} };
    Env2D::AABB aabb = { {-21.7494, -3.70393}, {7.19249, 11.2965} };
    std::cout<<"GMM initialized"<<std::endl;
    gmm.initialize(aabb);

    std::cout<<gmm.toString()<<std::endl;

    std::cout<<"Reading samples file"<<std::endl;
    const std::string filename = "./src/samples.txt";
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << "\n";
        return 1;
    }
    int lineMax = 50;
    int figureIndex = 1;
    // Eigen::Matrix<double, 3, Eigen::Dynamic> allSamples(3, 0);
    std::vector<Eigen::Vector2d> allSamples;

    std::string line;
    while (lineMax > 0 && std::getline(file, line)) {
        // Remove whitespace and brackets from the line
        line.erase(std::remove(line.begin(), line.end(), '['), line.end());
        line.erase(std::remove(line.begin(), line.end(), ']'), line.end());

        // Split the line into individual points
        std::stringstream ss(line);
        std::string point;
        // std::vector<Eigen::Vector3d> points;
        std::vector<Eigen::Vector2d> points;

        while (std::getline(ss, point, ';')) {
            std::stringstream pointStream(point);
            double x, y, z;

            // Parse the point
            char comma;
            if (pointStream >> x >> comma >> y >> comma >> z) {
                // std::cout << x << ", " << y << std::endl;
                // points.emplace_back(x, y, z);
                points.emplace_back(x, y);
                allSamples.emplace_back(x, y);
            }
        }

        // Eigen::Matrix<double, 3, Eigen::Dynamic> sample(3, points.size());
        // Eigen::Matrix<double, 2, Eigen::Dynamic> sample(2, points.size());
        // for (size_t i = 0; i < points.size(); ++i) {
        //     sample.col(i) = points[i];
        // }
        // std::cout<<gmm.samplesToString(sample) << std::endl;

        gmm.fit(points);
        std::cout << gmm.toString() << std::endl;

        // plotGMM3D(allSamples, gmm, figureIndex++);
        // plt::show();
        lineMax -= 1;
    }

    std::vector<std::string> colors = { "red", "orange", "pink", "green" };
    // Plot component initialization before fitting
    for (int k = 0; k < components; ++k) {
        auto mean = gmm.getComponentMean(k);
        auto cov = gmm.getComponentCovariance(k);

        std::vector<double> ellipse_x, ellipse_y;
        for (double theta = 0.0; theta < 2 * M_PI; theta += 0.01) {
            Eigen::Vector2d point(std::cos(theta), std::sin(theta));
            Eigen::Vector2d ellipse_point = mean + cov.llt().matrixL() * point;

            ellipse_x.push_back(ellipse_point(0));
            ellipse_y.push_back(ellipse_point(1));
        }
        plt::plot(ellipse_x, ellipse_y, {{"color", colors[k]}, {"linestyle", "--"}});
    }

    // gmm.fit(data, 100);
    // std::cout<<"GMM fitted"<<std::endl;

    // Plot original data

    std::vector<double> x, y;
    for (int i = 0; i < allSamples.size(); ++i) {
        x.push_back(allSamples[i](0));
        y.push_back(allSamples[i](1));
    }
    plt::scatter(x, y, 5.0, {{"color", "blue"}});

    // Plot fitted GMM components as ellipses
    // for (int k = 0; k < components; ++k) {
    //     auto mean = gmm.getComponentMean(k);
    //     auto cov = gmm.getComponentCovariance(k);

    //     std::vector<double> ellipse_x, ellipse_y;
    //     for (double theta = 0.0; theta < 2 * M_PI; theta += 0.01) {
    //         Eigen::Vector2d point(std::cos(theta), std::sin(theta));
    //         Eigen::Vector2d ellipse_point = mean + cov.llt().matrixL() * point;

    //         ellipse_x.push_back(ellipse_point(0));
    //         ellipse_y.push_back(ellipse_point(1));
    //     }
    //     plt::plot(ellipse_x, ellipse_y, {{"color", "green"}, {"linestyle", "--"}});
    // }

    // Display plot
    plt::xlabel("X");
    plt::ylabel("Y");
    plt::title("Gaussian Mixture Model Fitting");
    plt::show();

    // Vectord mean = (Vectord() << -21.4804, 10.5519, 3.38738).finished();
    // Matrixd covariance = (Eigen::Matrix<Scalar, dims, dims>() << 0.18122, 0, 0, 0, 0.18122, 0, 0, 0, 0.18122).finished();
    // Vectord sample = (Vectord() << 0.86144, 2.39, -0.892465).finished();

    // Scalar p = pdf(sample, mean, covariance);
    // std::cout << p << std::endl;

    return 0;
}
