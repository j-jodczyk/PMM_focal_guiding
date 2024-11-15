/* Simple program for testing how well GMM fits to data */
/* TODO: doesn't work anymore - fix in free time */
#include <iostream>
#include <random>
#include <eigen3/Eigen/Dense>
#include "../include/matplotlibcpp.h"

#include "../../../mitsuba/src/integrators/pmm_guiding/gaussian_mixture_model.h"
#include "../../../mitsuba/src/integrators/pmm_guiding/gaussian_component.h"

namespace plt = matplotlibcpp;
using Scalar = double;

constexpr int dims = 2; // 2D mixture
constexpr int components = 3;

using Vectord = Eigen::Matrix<Scalar, dims, 1>;
using Matrixd = Eigen::Matrix<Scalar, dims, Eigen::Dynamic>;

// todo: fix - need to pass Env to gmm

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

int main() {
    // Parameters for data generation
    std::cout<<"Begin program"<<std::endl;

    Vectord mean1, mean2, mean3;
    mean1 << 2.0, 3.0;
    mean2 << -2.0, -2.5;
    mean3 << 4.0, -3.5;

    Eigen::Matrix<Scalar, dims, dims> cov1 = (Eigen::Matrix<Scalar, dims, dims>() << 1.0, 0.2, 0.2, 0.5).finished();
    Eigen::Matrix<Scalar, dims, dims> cov2 = (Eigen::Matrix<Scalar, dims, dims>() << 0.3, 0.1, 0.1, 0.3).finished();
    Eigen::Matrix<Scalar, dims, dims> cov3 = (Eigen::Matrix<Scalar, dims, dims>() << 0.5, -0.2, -0.2, 0.7).finished();

    // Generate samples for each component
    int samples_per_component = 100;
    Matrixd data(dims, 3 * samples_per_component);
    data << generateGaussianSamples(mean1, cov1, samples_per_component),
            generateGaussianSamples(mean2, cov2, samples_per_component),
            generateGaussianSamples(mean3, cov3, samples_per_component);

    std::cout<<"Data generated"<<std::endl;

    pmm_focal::GaussianMixtureModel<dims, components, Scalar, pmm_focal::GaussianComponent> gmm;
    std::cout<<"GMM created"<<std::endl;

    gmm.initialize(data);
    std::cout<<"GMM initialized"<<std::endl;

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
        plt::plot(ellipse_x, ellipse_y, {{"color", "red"}, {"linestyle", "--"}});
    }

    gmm.fit(data, 100);
    std::cout<<"GMM fitted"<<std::endl;

    // Plot original data
    std::vector<double> x, y;
    for (int i = 0; i < data.cols(); ++i) {
        x.push_back(data(0, i));
        y.push_back(data(1, i));
    }
    plt::scatter(x, y, 10.0);

    // Plot fitted GMM components as ellipses
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
        plt::plot(ellipse_x, ellipse_y, {{"color", "green"}, {"linestyle", "--"}});
    }

    // Display plot
    plt::xlabel("X");
    plt::ylabel("Y");
    plt::title("Gaussian Mixture Model Fitting");
    plt::show();

    return 0;
}
