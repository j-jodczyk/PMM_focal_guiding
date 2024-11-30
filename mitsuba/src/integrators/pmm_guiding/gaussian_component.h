
#ifdef MITSUBA_LOGGING
#include <mitsuba/core/logger.h>
#endif

namespace pmm_focal {

template<size_t t_dims, typename Scalar_t>
class GaussianComponent {
public:
    using Scalar = Scalar_t;
    using Vectord = Eigen::Matrix<Scalar, t_dims, 1>;
    using Matrixd = Eigen::Matrix<Scalar, t_dims, t_dims>;

private:
    Matrixd m_covariance;
    Vectord m_mean;

public:
    Matrixd getCovariance() { return m_covariance; }
    Vectord getMean() { return m_mean; }

    void setCovariance(Matrixd newCovariance) { m_covariance = newCovariance; }
    void setMean(Vectord newMean) { m_mean = newMean; }

    Scalar pdf(const Vectord& sample) const {
        constexpr Scalar epsilon = std::numeric_limits<Scalar>::epsilon();
        const Scalar twoPi = static_cast<Scalar>(2.0 * M_PI);
        //  safeguard against numerical instability
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, t_dims, t_dims>> eigenSolver(m_covariance);
        Matrixd stableCovariance = m_covariance;
        for (int i = 0; i < t_dims; ++i) {
            if (eigenSolver.eigenvalues()[i] < epsilon) {
                stableCovariance += epsilon * Matrixd::Identity();
            }
        }
        Eigen::Matrix<Scalar, t_dims, t_dims> covarianceInv = stableCovariance.inverse();
        Scalar detCovariance = stableCovariance.determinant() + epsilon;

        Vectord diff = sample - m_mean;
        Scalar mahalanobisDist = diff.transpose() * covarianceInv * diff;
        int minusTDims = static_cast<int>(-t_dims);

        Scalar normalization =
            std::pow(twoPi, static_cast<Scalar>(minusTDims / 2.0))
            * std::pow(detCovariance, static_cast<Scalar>(-0.5));
        Scalar pdfValue = normalization
            * std::exp(static_cast<Scalar>(-0.5) * mahalanobisDist);

        // std::cout << "detCovariance: " << detCovariance << "\tmahalanobisDist: " << mahalanobisDist << "\tnormalization2: " << normalization2 << "\tnormalization: " << normalization << "\tpdf: " << pdfValue << std::endl;

        return pdfValue < epsilon ? epsilon : pdfValue;
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "mean = " << getMeanStr() << "covariance = " << getCovarianceStr();
        return oss.str();
    }

    std::string getMeanStr() const {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < t_dims; ++i) {
            oss << m_mean(i);
            if (i < t_dims - 1) oss << " ";
        }
        oss << "]";
        return oss.str();
    }

    std::string getCovarianceStr() const {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < t_dims; ++i) {
            oss << "[";
            for (size_t j = 0; j < t_dims; ++j) {
                oss << m_covariance(i, j);
                if (j < t_dims - 1) oss << " ";
            }
            oss << "]";
        }
        oss << "]";
        return oss.str();
    }
};

}