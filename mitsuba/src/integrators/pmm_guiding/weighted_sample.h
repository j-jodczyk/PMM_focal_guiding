#ifndef WEIGHTED_SAMPLE_H
#define WEIGHTED_SAMPLE_H

#include <eigen3/Eigen/Dense>

namespace pmm_focal {

    struct WeightedSample {
        Eigen::VectorXd point;
        float weight;
    };
}

#endif // WEIGHTED_SAMPLE_H