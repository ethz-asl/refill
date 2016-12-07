#include "refill/distributions/gaussian_distribution.h"

using std::size_t;

namespace refill {

/**
 * Creates an empty GaussianDistribution.
 *
 * Use this if you don't know the dimension beforehand or don't know the
 * parameters.
 *
 * To be able to use the distribution, first set the parameters using the
 * setDistParam() function.
 */
GaussianDistribution::GaussianDistribution()
    : GaussianDistribution(0) {}


/**
 * @param dimension Dimension of the constructed normal distribution.
 */
GaussianDistribution::GaussianDistribution(const int& dimension)
    : mean_(Eigen::VectorXd::Zero(dimension)),
      covariance_(Eigen::MatrixXd::Identity(dimension, dimension)) {}

/**
 * @param dist_mean Mean of the constructed distribution.
 * @param dist_cov Covariance matrix of the constructed distribution.
 */
GaussianDistribution::GaussianDistribution(const Eigen::VectorXd& dist_mean,
                                           const Eigen::MatrixXd& dist_cov) {
  setDistParam(dist_mean, dist_cov);
}

/**
 * Also performes checks that the mean and covariance
 * matrix have compatible size.
 *
 * @param dist_mean The new mean of the distribution.
 * @param dist_cov  The new covariance of the distribution.
 */
void GaussianDistribution::setDistParam(const Eigen::VectorXd& dist_mean,
                                        const Eigen::MatrixXd& dist_cov) {
  CHECK_EQ(dist_mean.size(), dist_cov.rows());
  CHECK_EQ(dist_cov.rows(), dist_cov.cols());

  Eigen::LLT<Eigen::MatrixXd> chol_of_cov(dist_cov);
  CHECK(chol_of_cov.info() != Eigen::NumericalIssue) << "Matrix not s.p.d.";

  mean_ = dist_mean;
  covariance_ = dist_cov;
}

/**
 * Also checks that the new mean has the same dimension as the current mean.
 *
 * @param mean The new distribution mean.
 */
void GaussianDistribution::setMean(const Eigen::VectorXd& mean) {
  CHECK_EQ(this->dimension(), mean.size());
  mean_ = mean;
}

/**
 * Also checks that the covariance matrix is a square matrix with the
 * same dimension as the mean and is symmetric positiv definite.
 *
 * @param cov The new distribution covariance.
 */
void GaussianDistribution::setCov(const Eigen::MatrixXd& cov) {
  CHECK_EQ(this->dimension(), cov.rows());
  CHECK_EQ(cov.rows(), cov.cols());

  Eigen::LLT<Eigen::MatrixXd> chol_of_cov(cov);
  CHECK_NE(chol_of_cov.info(), Eigen::NumericalIssue)<< "Matrix not s.p.d.";

  covariance_ = cov;
}

/**
 * @return the dimension of the distribution.
 */
size_t GaussianDistribution::dimension() const {
  return mean_.size();
}

/**
 * @return the current distribution mean.
 */
Eigen::VectorXd GaussianDistribution::mean() const {
  return mean_;
}

/**
 * @return the current distribution covariance.
 */
Eigen::MatrixXd GaussianDistribution::cov() const {
  return covariance_;
}

/**
 * Also checks for right dimensionality of distribution.
 *
 * @param right_side Distribution which will be added to `*this`.
 * @return new distribution which is the sum of `*this` and @e right_side.
 */
GaussianDistribution GaussianDistribution::operator+(
    const GaussianDistribution &right_side) {

  CHECK(right_side.dimension() == this->dimension())
      << "Distribution dimensions do not match.";

  GaussianDistribution result(mean_ + right_side.mean(),
                              covariance_ + right_side.cov());
  return result;
}

/**
 * @relates GaussianDistribution
 *
 * @brief Non-member overloaded operator for linear transformations of Gaussian
 *        random vectors.
 *
 * @param mat Linear transformation matrix.
 * @param gaussian Gaussian random vector.
 * @return transformed Gaussian random vector.
 */
inline GaussianDistribution operator*(const Eigen::MatrixXd &mat,
                                      const GaussianDistribution gaussian) {
  CHECK_EQ(mat.cols(), gaussian.dimension());
  return GaussianDistribution(mat * gaussian.mean(),
                              mat * gaussian.cov() * mat.transpose());
}

}  // namespace refill
