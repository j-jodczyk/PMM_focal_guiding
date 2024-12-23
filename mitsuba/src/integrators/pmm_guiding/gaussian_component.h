
#ifdef MITSUBA_LOGGING
#include <mitsuba/core/logger.h>
#endif

namespace pmm_focal
{

    // template <size_t t_dims, typename Scalar_t>
    class GaussianComponent
    {
    public:
        // using Scalar = Scalar_t;

    private:
        Eigen::VectorXd mean;
        // In this paper, we manage to reduce this complexity to O(NKD2)
        // by deriving formulas for working directly with precision
        // matrices instead of covariance matrice - precision matrix is
        // inverse of covariance
        Eigen::MatrixXd precisionMatrix;
        double determinant;
        double weight;

    public:
        GaussianComponent(int dimension, double initialVariance)
        {
            mean = Eigen::VectorXd::Zero(dimension);
            precisionMatrix = Eigen::MatrixXd::Identity(dimension, dimension) / initialVariance;
            determinant = pow(initialVariance, -dimension);
            weight = 1.0;
        }

        Eigen::MatrixXd getPrecisionMatrix() const { return precisionMatrix; }
        Eigen::VectorXd getMean() const { return mean; }
        double getDeterminant() const { return determinant; }
        double getWeight() const { return weight; }

        void setPrecisionMatrix(Eigen::MatrixXd newPrecisionMatrix) { precisionMatrix = newPrecisionMatrix; }
        void setMean(Eigen::VectorXd newMean) { mean = newMean; }
        void setDeterminant(double newDeterminant) { determinant = newDeterminant; }
        void setWeight(double newWeight) { weight = newWeight; }

        std::string toString() const
        {
            std::ostringstream oss;
            oss << "mean = " << getMeanStr() << "precision = " << getPrecisionStr();
            return oss.str();
        }

        std::string getMeanStr() const
        {
            size_t dims =  mean.cols();
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

        std::string getPrecisionStr() const
        {
            size_t dims = precisionMatrix.cols();
            std::ostringstream oss;
            oss << "[";
            for (size_t i = 0; i < dims; ++i)
            {
                oss << "[";
                for (size_t j = 0; j < dims; ++j)
                {
                    oss << precisionMatrix(i, j);
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