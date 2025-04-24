#ifndef WEIGHTED_SAMPLE_H
#define WEIGHTED_SAMPLE_H

#include <eigen3/Eigen/Dense>

namespace pmm_focal {

    struct WeightedSample {
        Eigen::VectorXd point;
        float weight;

        WeightedSample() = default;
        WeightedSample(Eigen::VectorXd& p, float w): point(p), weight(w) {}

        std::string toString() const {
            std::ostringstream oss;
            oss << "point: [";
            for (int i = 0; i < point.size(); ++i) {
                oss << point[i];
                if (i < point.size() - 1) {
                    oss << ", ";
                }
            }
            oss << "], weight: " << weight << "";
            return oss.str();
        }
    };
}

#endif // WEIGHTED_SAMPLE_H