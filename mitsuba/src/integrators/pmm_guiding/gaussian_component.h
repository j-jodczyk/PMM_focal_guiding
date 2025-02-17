
#ifdef MITSUBA_LOGGING
#include <mitsuba/core/logger.h>
#endif

namespace pmm_focal
{
    class GaussianComponent
    {
    public:

    private:
        Eigen::VectorXd mean;
        Eigen::MatrixXd covariance;
        double weight;
        size_t priorSampleCount;

    public:
        GaussianComponent() {}

        Eigen::MatrixXd getCovariance() const { return covariance; }
        Eigen::VectorXd getMean() const { return mean; }
        double getWeight() const { return weight; }
        size_t getPriorSampleCount() const { return priorSampleCount; }

        void setCovariance(Eigen::MatrixXd newCovaraince) { covariance = newCovaraince; }
        void setMean(Eigen::VectorXd newMean) { mean = newMean; }
        void setWeight(double newWeight) { weight = newWeight; }
        void setPriorSampleCount(size_t newPSC) { priorSampleCount = newPSC; }

        void updateComponent(double N, double N_new, const Eigen::VectorXd& new_mean, const Eigen::MatrixXd& new_cov, double new_weight, double alpha) {
            N *= alpha;
            priorSampleCount *= alpha;

            weight = (N * weight + N_new * new_weight) / (N + N_new);
            Eigen::VectorXd priorMean = mean;
            mean = (N * priorMean + N_new * new_mean) / (N + N_new);

            Eigen::MatrixXd correction = (N * N_new) / pow(N + N_new, 2) * (new_mean - priorMean) * (new_mean - priorMean).transpose();
            covariance = (N * covariance + N_new * new_cov) / (N + N_new) + correction;

            covariance += 1e-6 * Eigen::MatrixXd::Identity(covariance.rows(), covariance.cols()); // Regularization

            priorSampleCount += N_new;
        }

        Eigen::VectorXd sample(std::mt19937& gen) const {
            std::normal_distribution<> dist(0.0, 1.0);
            Eigen::VectorXd z(mean.size());
            for (int i = 0; i < mean.size(); ++i) {
                z[i] = dist(gen);
            }
            return mean + covariance.llt().matrixL() * z;
        }

        void deactivate(size_t dims) {
            weight = 0;
            mean = Eigen::VectorXd::Zero(dims);
            covariance = Eigen::MatrixXd::Zero(dims, dims);
            priorSampleCount = 0;
        }

        std::string toString() const {
            std::ostringstream oss;
            // todo: apperently now this throws - wtf?
            oss << "weight = " << getWeight() << " mean = " << getMeanStr() << " covaraince = " << getCovarianceStr();
            return oss.str();
        }

        std::string getMeanStr() const {
            size_t dims =  mean.rows();
            std::ostringstream oss;
            oss << "[";
            for (size_t i = 0; i < dims; ++i)
            {
                oss << mean(i);
                if (i < dims - 1)
                    oss << " ";
            }
            oss << "]";
            return oss.str();
        }

        std::string getCovarianceStr() const {
            size_t dims = covariance.cols();
            std::ostringstream oss;
            oss << "[";
            for (size_t i = 0; i < dims; ++i)
            {
                oss << "[";
                for (size_t j = 0; j < dims; ++j)
                {
                    oss << covariance(i, j);
                    if (j < dims - 1)
                        oss << " ";
                }
                oss << "]";
            }
            oss << "]";
            return oss.str();
        }
    };
}
