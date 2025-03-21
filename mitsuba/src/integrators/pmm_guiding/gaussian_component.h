
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

    public:
        GaussianComponent() {}
        GaussianComponent(mitsuba::FileStream* in) {
            deserialize(in);
        }

        Eigen::MatrixXd getCovariance() const { return covariance; }
        Eigen::VectorXd getMean() const { return mean; }
        double getWeight() const { return weight; }

        void setCovariance(Eigen::MatrixXd newCovariance) { covariance = newCovariance; }
        void setMean(Eigen::VectorXd newMean) { mean = newMean; }
        void setWeight(double newWeight) { weight = newWeight; }

        void updateComponent(double N, double N_new, const Eigen::VectorXd& new_mean, const Eigen::MatrixXd& new_cov, double new_weight, double alpha) {
            N *= alpha; // Ruppert 2020

            weight = (N * weight + N_new * new_weight) / (N + N_new);
            Eigen::VectorXd priorMean = mean;
            mean = (N * priorMean + N_new * new_mean) / (N + N_new);

            Eigen::MatrixXd correction = (N * N_new) / pow(N + N_new, 2) * (new_mean - priorMean) * (new_mean - priorMean).transpose();
            covariance = (N * covariance + N_new * new_cov) / (N + N_new) + correction;

            covariance += 1e-6 * Eigen::MatrixXd::Identity(covariance.rows(), covariance.cols()); // Regularization
        }

        Eigen::VectorXd sample(std::mt19937& gen) const {
            std::normal_distribution<> dist(0.0, 1.0);
            size_t meanSize = mean.size();
            Eigen::VectorXd z(meanSize);
            for (size_t i = 0; i < meanSize; ++i) {
                z[i] = dist(gen);
            }
            return mean + covariance.llt().matrixL() * z;
        }

        void deactivate(size_t dims) {
            weight = 0.0;
            mean = Eigen::VectorXd::Zero(dims);
            covariance = Eigen::MatrixXd::Zero(dims, dims);
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
