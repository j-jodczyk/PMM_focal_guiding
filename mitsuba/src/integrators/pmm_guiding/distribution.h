/** Based on Dodik 2022 */

#include "eigen_boost_serialization.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>

namespace pmm {

template<int t_dimentions, typename Scalar>
class Distribution {
public:
    using Vectord = Eigen::Matrix<Scalar, t_dimentions, 1>;
    using Matrixd = Eigen::Matrix<Scalar, t_dimentions, t_dimentions>;

    virtual ~Distribution() {}
    virtual Vectord sample(const std::function<Scalar()>& rng) const = 0;
    virtual Scalar pdf(const Vectord& sample) const = 0;
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

};