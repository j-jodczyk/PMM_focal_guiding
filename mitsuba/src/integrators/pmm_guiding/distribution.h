/** Based on Dodik 2022 */

#include "eigen_boost_serialization.h"

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/StdVector>

namespace pmm_focal {

template<typename Scalar>
class Distribution {
public:
    using Vectord = Eigen::VectorXd;
    using Matrixd = Eigen::MatrixXd;

    virtual ~Distribution() {}
    // virtual Vectord sample(const std::function<Scalar()>& rng) const = 0;
    // virtual Scalar pdf(const Vectord& sample) const = 0;
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

};
