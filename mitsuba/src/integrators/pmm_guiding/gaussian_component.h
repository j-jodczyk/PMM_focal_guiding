
#ifdef MITSUBA_LOGGING
#include <mitsuba/core/logger.h>
#endif

namespace pmm_focal
{
    class GaussianComponent
    {
    private:
        Eigen::VectorXd mean;
        Eigen::MatrixXd covariance;
        Eigen::MatrixXd inverseCovariance;
        Eigen::MatrixXd L;
        float logNormConst;
        float weight;
        size_t dims = 3;

    public:
        Eigen::VectorXd sum_x;
        Eigen::MatrixXd sum_xxT;
        float r_k = 0; // total soft responsibility
        float N;
        bool isNew = false;
        GaussianComponent() {
            sum_x = Eigen::VectorXd::Zero(dims);
            sum_xxT = Eigen::MatrixXd::Zero(dims, dims);
        }

        GaussianComponent(mitsuba::FileStream* in) {
            deserialize(in);
        }

        Eigen::MatrixXd getCovariance() const { return covariance; }
        Eigen::MatrixXd getInverseCovariance() const { return inverseCovariance; }
        Eigen::VectorXd getMean() const { return mean; }
        float getLogNormConst() const { return logNormConst; }
        float getWeight() const { return weight; }

        void setCovariance(Eigen::MatrixXd newCovariance) {
            covariance = newCovariance;
            // Regularize covariance to avoid numerical issues
            covariance += 1e-5f * Eigen::MatrixXd::Identity(covariance.rows(), covariance.cols());

            // Check for positive definiteness
            Eigen::LLT<Eigen::MatrixXd> llt(covariance);
            if (llt.info() == Eigen::Success) {
                inverseCovariance = llt.solve(Eigen::MatrixXd::Identity(covariance.rows(), covariance.cols()));
                L = llt.matrixL();  // L is a cached Eigen::MatrixXd
                
            } else {
                // Fallback: eigenvalue correction
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(covariance);
                Eigen::VectorXd eigenvalues = eig.eigenvalues();
                for (int i = 0; i < eigenvalues.size(); ++i) {
                    if (eigenvalues[i] < 1e-6)
                        eigenvalues[i] = 1e-6;
                }
        
                covariance = eig.eigenvectors() * eigenvalues.asDiagonal() * eig.eigenvectors().transpose();
                // Retry LLT on corrected covariance
                Eigen::LLT<Eigen::MatrixXd> safeLLT(covariance);
                L = safeLLT.matrixL();  // Still cache L
                inverseCovariance = safeLLT.solve(Eigen::MatrixXd::Identity(covariance.rows(), covariance.cols()));
            }
            float logDetCov = 2.0 * L.diagonal().array().log().sum();
            logNormConst = -0.5 * (dims * std::log(2 * M_PI) + logDetCov);
        }

        void setMean(Eigen::VectorXd newMean) { mean = newMean; }
        void setWeight(float newWeight) { weight = newWeight; }

        void updateComponentWithSufficientStatistics(
            const Eigen::VectorXd& sum_x_new,
            const Eigen::MatrixXd& sum_xxT_new,
            float r_k_new,
            float N_new,
            float decay
        ) {
            // Decay old statistics
            sum_x *= decay;
            sum_xxT *= decay;
            r_k *= decay;
            N *= decay;

            // Accumulate new statistics
            sum_x += sum_x_new;
            sum_xxT += sum_xxT_new;
            r_k += r_k_new;
            N += N_new;

            // Compute MLE estimates
            mean = sum_x / r_k;
            covariance = (sum_xxT / r_k) - mean * mean.transpose();

            setCovariance(covariance);
        }

        // Box-Muller Transform
        Eigen::VectorXd sample(mitsuba::RadianceQueryRecord &rRec) const {
            size_t meanSize = mean.size();
            Eigen::VectorXd z(meanSize);

            for (size_t i = 0; i < meanSize; i += 2) {
                float u1 = rRec.nextSample1D();
                u1 =  std::max(1e-6f, u1);
                float u2 = rRec.nextSample1D();

                float r = std::sqrt(-2.0 * std::log(u1));
                float theta = 2.0 * M_PI * u2;

                z[i] = r * std::cos(theta);
                if (i + 1 < meanSize) {
                    z[i + 1] = r * std::sin(theta);
                }
            }

            // Cholesky decomposition
            return mean + L * z;
        }

        void deactivate(size_t dims) {
            r_k = 0.0;
            sum_x = Eigen::VectorXd::Zero(dims);
            sum_xxT = Eigen::MatrixXd::Zero(dims, dims);
            weight = 0.0;
            mean = Eigen::VectorXd::Zero(dims);
            covariance = Eigen::MatrixXd::Zero(dims, dims);
            inverseCovariance = Eigen::MatrixXd::Zero(dims, dims);
            L = Eigen::MatrixXd::Zero(dims, dims);
            logNormConst = 0.0;
            N = 0;
        }

        std::string toString() const {
            std::ostringstream oss;
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
            // size_t meanSize = mean.size();
            // size_t covSize = covariance.rows();

            // out->write(reinterpret_cast<const char*>(&weight), sizeof(weight));
            // out->write(reinterpret_cast<const char*>(&meanSize), sizeof(meanSize));
            // out->write(reinterpret_cast<const char*>(mean.data()), meanSize * sizeof(float));

            // out->write(reinterpret_cast<const char*>(&covSize), sizeof(covSize));
            // out->write(reinterpret_cast<const char*>(covariance.data()), covSize * covSize * sizeof(float));
        }

        void deserialize(mitsuba::FileStream* in) {
            // size_t meanSize, covSize;

            // in->read(reinterpret_cast<char*>(&weight), sizeof(weight));

            // in->read(reinterpret_cast<char*>(&meanSize), sizeof(meanSize));
            // mean.resize(meanSize);
            // in->read(reinterpret_cast<char*>(mean.data()), meanSize * sizeof(float));

            // in->read(reinterpret_cast<char*>(&covSize), sizeof(covSize));
            // covariance.resize(covSize, covSize);
            // in->read(reinterpret_cast<char*>(covariance.data()), covSize * covSize * sizeof(float));
        }
    };
}
