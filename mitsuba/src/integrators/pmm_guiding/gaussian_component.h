
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
        Eigen::MatrixXd inverseCovariance;
        Eigen::MatrixXd L;
        double logDetCov;
        double weight;
        double softCount;

    public:
        GaussianComponent() {}
        GaussianComponent(mitsuba::FileStream* in) {
            deserialize(in);
        }

        Eigen::MatrixXd getCovariance() const { return covariance; }
        Eigen::MatrixXd getInverseCovariance() const { return inverseCovariance; }
        Eigen::VectorXd getMean() const { return mean; }
        double getLogDetCov() const { return logDetCov; }
        double getWeight() const { return weight; }
        double getSoftCount() const { return softCount; }

        void setCovariance(Eigen::MatrixXd newCovariance) {
            covariance = newCovariance;
            inverseCovariance = covariance.inverse();
            logDetCov = std::log(covariance.determinant());
            L = covariance.llt().matrixL();
        }
        void setMean(Eigen::VectorXd newMean) { mean = newMean; }
        void setWeight(double newWeight) { weight = newWeight; }

        void updateComponent(size_t N, size_t N_new, const Eigen::VectorXd& new_mean, const Eigen::MatrixXd& new_cov, double new_weight, double alpha) {
            // N *= alpha; // Ruppert 2020 --- not, misunderstood
            // weight = (N * weight + N_new * new_weight) / (N + N_new);

            SLog(mitsuba::EInfo, ("prior mean: " + getMeanStr() + " prior covaraince: " + getCovarianceStr() + " N: %d").c_str(), N);

            Eigen::VectorXd priorMean = mean;
            mean = (N * priorMean + N_new * new_mean) / (N + N_new);

            Eigen::MatrixXd correction = (N * N_new) / pow(N + N_new, 2) * (new_mean - priorMean) * (new_mean - priorMean).transpose();
            covariance = (N * covariance + N_new * new_cov) / (N + N_new) + correction;

            covariance += 1e-6 * Eigen::MatrixXd::Identity(covariance.rows(), covariance.cols()); // Regularization
            inverseCovariance = covariance.inverse();
            logDetCov = std::log(covariance.determinant());
            Eigen::MatrixXd L = covariance.llt().matrixL();

            updateSoftCount(N, N_new, new_weight, alpha);
        }

        void updateSoftCount(double N, double N_new, double new_weight, double alpha) {
            softCount = N * weight + N_new * new_weight + alpha; // using expectation, not MAP - we'll experiment what is better
        }

        // Box-Muller Transform
        Eigen::VectorXd sample(mitsuba::RadianceQueryRecord &rRec) const {
            size_t meanSize = mean.size();
            Eigen::VectorXd z(meanSize);

            for (size_t i = 0; i < meanSize; i += 2) {
                double u1 = rRec.nextSample1D();
                u1 =  std::max(1e-6, u1);
                double u2 = rRec.nextSample1D();

                double r = std::sqrt(-2.0 * std::log(u1));
                double theta = 2.0 * M_PI * u2;

                z[i] = r * std::cos(theta);
                if (i + 1 < meanSize) {
                    z[i + 1] = r * std::sin(theta);
                }
            }

            // Cholesky decomposition
            return mean + L * z;
        }

        void deactivate(size_t dims) {
            weight = 0.0;
            mean = Eigen::VectorXd::Zero(dims);
            covariance = Eigen::MatrixXd::Zero(dims, dims);
            inverseCovariance = Eigen::MatrixXd::Zero(dims, dims);
            L = Eigen::MatrixXd::Zero(dims, dims);
        }

        std::string toString() const {
            std::ostringstream oss;
            // todo: apperently now this throws - wtf?
            oss << "weight = " << getWeight() << " mean = " << getMeanStr() << " covariance = " << getCovarianceStr();
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

        void serialize(mitsuba::FileStream* out) const {
            size_t meanSize = mean.size();
            size_t covSize = covariance.rows();

            out->write(reinterpret_cast<const char*>(&weight), sizeof(weight));
            out->write(reinterpret_cast<const char*>(&meanSize), sizeof(meanSize));
            out->write(reinterpret_cast<const char*>(mean.data()), meanSize * sizeof(double));

            out->write(reinterpret_cast<const char*>(&covSize), sizeof(covSize));
            out->write(reinterpret_cast<const char*>(covariance.data()), covSize * covSize * sizeof(double));
        }

        void deserialize(mitsuba::FileStream* in) {
            size_t meanSize, covSize;

            in->read(reinterpret_cast<char*>(&weight), sizeof(weight));

            in->read(reinterpret_cast<char*>(&meanSize), sizeof(meanSize));
            mean.resize(meanSize);
            in->read(reinterpret_cast<char*>(mean.data()), meanSize * sizeof(double));

            in->read(reinterpret_cast<char*>(&covSize), sizeof(covSize));
            covariance.resize(covSize, covSize);
            in->read(reinterpret_cast<char*>(covariance.data()), covSize * covSize * sizeof(double));
        }
    };
}
