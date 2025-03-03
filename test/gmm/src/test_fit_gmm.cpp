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
#include <opencv2/opencv.hpp>

#include "../../../mitsuba/src/integrators/pmm_guiding/gaussian_mixture_model.h"
#include "../../../mitsuba/src/integrators/pmm_guiding/envs/3d_env.h"
#include "../../../mitsuba/src/integrators/pmm_guiding/envs/2d_env.h"

namespace plt = matplotlibcpp;
using Scalar = double;

constexpr int dims = 3;
// constexpr int dims = 2;
constexpr int components = 4;

using Vectord = Eigen::Matrix<Scalar, dims, 1>;
using Matrixd = Eigen::Matrix<Scalar, dims, Eigen::Dynamic>;

using SampleVector = Eigen::Matrix<Scalar, dims, Eigen::Dynamic>;
using GMM = pmm_focal::GaussianMixtureModel<double, Env3D>;

using IterationSamples = std::vector<pmm_focal::WeightedSample>;

std::vector<IterationSamples> readCSV(const std::string& filename) {
    std::vector<IterationSamples> iterations;
    std::ifstream file(filename);
    std::string line;
    std::string header;

    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return iterations;
    }

    // Read header
    std::getline(file, header);

    int currentIteration = -1;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        int iteration;
        double x, y, z;
        float weight;

        // Parse iteration
        std::getline(ss, token, ',');
        iteration = std::stoi(token);

        // Parse x, y, z
        std::getline(ss, token, ','); x = std::stod(token);
        std::getline(ss, token, ','); y = std::stod(token);
        std::getline(ss, token, ','); z = std::stod(token);

        // Parse weight
        std::getline(ss, token, ','); weight = std::stof(token);

        Eigen::VectorXd point(3);
        point << x, y, z;
        pmm_focal::WeightedSample sample{point, weight};

        // Create a new vector for each iteration
        if (iteration != currentIteration) {
            currentIteration = iteration;
            iterations.emplace_back();
        }

        iterations.back().push_back(sample);
    }

    file.close();
    return iterations;
}

void plotGaussian2D(cv::Mat& img, const Eigen::Vector2d& mean, const Eigen::Matrix2d& cov, double intensity, double alpha = 0.5) {
    cv::Point center(static_cast<int>(mean(0)), static_cast<int>(mean(1)));
    // cv::Scalar color(intensity, intensity, intensity);
    cv::Scalar color(0, 255, 0, 128);
    int radius = 20; // Example fixed radius

    cv::circle(img, center, radius, color, -1, cv::LINE_AA);
    cv::addWeighted(img, alpha, img, 1 - alpha, 0, img);
}

void plotGMMOnImage2D( GMM& gmm, const std::string& imagePath, const Eigen::Vector2d& min, const Eigen::Vector2d& max) {
    cv::Mat img = cv::imread(imagePath);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << imagePath << std::endl;
        return;
    }

    cv::rectangle(img, cv::Point(static_cast<int>(min(0)), static_cast<int>(min(1))),
                cv::Point(static_cast<int>(max(0)), static_cast<int>(max(1))),
                cv::Scalar(255, 0, 0), 2);

    for (const auto& component : gmm.getComponents()) {
        Eigen::Vector2d mean = component.getMean().head<2>();
        double intensity = component.getMean()(2);
        Eigen::Matrix2d cov = component.getCovariance().block<2, 2>(0, 0);

        plotGaussian2D(img, mean, cov, intensity);
    }

    cv::imshow("GMM Visualization", img);
    cv::waitKey(0);
}


int main() {
    std::cout<<"Begin program"<<std::endl;
    GMM gmm;

    std::cout<<"GMM created"<<std::endl;
    Env3D::AABB aabb = { {-6.3658, -1.5274, -4.73046}, {5.27597, 8.04131, 10.0598} };
    gmm.setMaxNumComp(10);
    gmm.setMinNumComp(3);
    gmm.setSplittingThreshold(1000);
    gmm.setMergingThreshold(0.01);
    gmm.init(4, 3, aabb);
    // AABB = { "min" : [-6.3658, -1.5274, -4.73046], "max" : [5.27597, 8.04131, 10.0598] }
    std::cout<<"GMM initialized"<<std::endl;
    std::cout<<gmm.toString()<<std::endl;
    std::cout<<"Reading samples file"<<std::endl;

    std::string filename = "./src/points_in_iterations_2.csv"; // Change this to your file path
    auto iterations = readCSV(filename);

    std::cout << "Total iterations: " << iterations.size() << std::endl;
    for (size_t i = 0; i < iterations.size(); ++i) {
        gmm.processBatch(iterations[i]);
        std::cout << gmm.toString() << std::endl;
        // plotGMMOnImage2D(gmm, "../../scenes/dining-room/Reference.png", Eigen::Vector2d(-6.3658, -1.5274), Eigen::Vector2d(5.27597, 8.04131));
    }

    return 0;
}
