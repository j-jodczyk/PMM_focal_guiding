template<
    size_t t_dims,
    typename Scalar_t
>
class GaussianComponent() {
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

    Scalar pdf(Vectord& sample) {
        Vectord diff = sample - m_mean;
        Scalar mahalanobisDist = diff.transpose() * m_covariance.inverse() * diff;
        Scalar detCovariance = m_covariance.determinant();
        Scalar pdfValue = (1.0 / std::sqrt(std::pow(2 * M_PI, d) * detSigma)) * std::exp(-0.5 * mahalanobisDist);
        return pdfValue;
    }
}