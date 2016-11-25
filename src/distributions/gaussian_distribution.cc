#include "refill/distributions/gaussian_distribution.h"

namespace refill {

/**
 * @brief Default constructor.
 *
 * Creates a univariate standard normal distribution.
 */
GaussianDistribution::GaussianDistribution()
    : GaussianDistribution(1) {
}


/**
 * @brief Constructs a normal distribution with given dimension.
 *
 * @param dimension Dimension of the constructed normal distribution.
 */
GaussianDistribution::GaussianDistribution(const int& dimension)
    : mean_(Eigen::VectorXd::Zero(dimension)),
      covariance_(Eigen::MatrixXd::Identity(dimension, dimension)) {
}

/**
 * @brief Constructs a gaussian distribution with given mean and covariance.
 *
 * @param dist_mean Mean of the constructed distribution.
 * @param dist_cov Covariance matrix of the constructed distribution.
 */
GaussianDistribution::GaussianDistribution(const Eigen::VectorXd& dist_mean,
                                           const Eigen::MatrixXd& dist_cov) {
  setDistParam(dist_mean, dist_cov);
}

/**
 * @brief Sets the distribution parameters.
 *
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
 * @brief Sets the mean to a new value.
 *
 * Also checks that the new mean has the same dimension as the current mean.
 *
 * @param mean The new distribution mean.
 */
void GaussianDistribution::setMean(const Eigen::VectorXd& mean) {
  CHECK_EQ(this->dimension(), mean.size());
  mean_ = mean;
}

/**
 * @brief Sets the covariance matrix to a new value.
 *
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
 * @brief Returns the dimension of the distribution
 *
 * @returns the dimension of the distribution.
 */
int GaussianDistribution::dimension() const {
  return mean_.size();
}

/**
 * @brief Returns the current mean of the distribution.
 *
 * @return the current distribution mean.
 */
Eigen::VectorXd GaussianDistribution::mean() const {
  return mean_;
}

/**
 * @brief Returns the current covariance of the distribution.
 *
 * @return the current distribution covariance.
 */
Eigen::MatrixXd GaussianDistribution::cov() const {
  return covariance_;
}

/**
 * @brief Implements the addition of two gaussian distributions.
 *
 * Checks for right dimensionality of distribution.
 *
 * @param right_side Distribution which will be added to `*this`.
 * @return new distribution which is the sum of `*this` and @p right_side.
 */
GaussianDistribution GaussianDistribution::operator+(
    const GaussianDistribution &right_side) {

  CHECK(right_side.dimension() == this->dimension())
      << "Distribution dimensions do not match.";

  GaussianDistribution result(mean_ + right_side.mean(),
                              covariance_ + right_side.cov());
  return result;
}

// Non-member overloaded operator for linear transforms of Gaussian random
// vectors.
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
