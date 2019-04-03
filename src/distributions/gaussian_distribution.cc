#include "refill/distributions/gaussian_distribution.h"

using std::size_t;

namespace refill {

/**
 * Creates a standard normal distribution.
 */
GaussianDistribution::GaussianDistribution() : GaussianDistribution(1) {}

/** @param dimension Dimension of the constructed normal distribution. */
GaussianDistribution::GaussianDistribution(const int& dimension)
    : mean_(Eigen::VectorXd::Zero(dimension)),
      covariance_(Eigen::MatrixXd::Identity(dimension, dimension)) {}

/**
 * @param dist_mean Mean of the constructed distribution.
 * @param dist_cov Covariance matrix of the constructed distribution.
 */
GaussianDistribution::GaussianDistribution(const Eigen::VectorXd& dist_mean,
                                           const Eigen::MatrixXd& dist_cov) {
  setDistributionParameters(dist_mean, dist_cov);
}

/**
 * Also performes checks that the mean and covariance
 * matrix have compatible size.
 *
 * @param dist_mean The new mean of the distribution.
 * @param dist_cov  The new covariance of the distribution.
 */
void GaussianDistribution::setDistributionParameters(
    const Eigen::VectorXd& dist_mean, const Eigen::MatrixXd& dist_cov) {
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
  CHECK_EQ(this->dimension(), mean.rows());
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
  CHECK_NE(chol_of_cov.info(), Eigen::NumericalIssue) << "Matrix not s.p.d.";

  covariance_ = cov;
}

/** @return the dimension of the distribution. */
size_t GaussianDistribution::dimension() const { return mean_.rows(); }

/** @return the current distribution mean. */
Eigen::VectorXd GaussianDistribution::mean() const { return mean_; }

/** @return the current distribution covariance. */
Eigen::MatrixXd GaussianDistribution::cov() const { return covariance_; }

/** @return a random vector drawn from the distribution. */
Eigen::VectorXd GaussianDistribution::drawSample() {
  CHECK_NE(mean_.rows(), 0) << "Distribution parameters have not been set.";

  // Generate normal distributed random vector
  std::normal_distribution<double> normal_dist(0.0, 1.0);

  Eigen::VectorXd uniform_random_vector(mean_.rows());
  for (int i = 0; i < mean_.rows(); ++i) {
    uniform_random_vector[i] = normal_dist(rng_);
  }

  // Calculate matrix L
  Eigen::LLT<Eigen::MatrixXd> chol_of_cov(covariance_);
  Eigen::MatrixXd L = chol_of_cov.matrixL();

  return mean_ + L * uniform_random_vector;
}

/**
 * Checks for compatible size of x.
 *
 * @param x Sample at which the pdf should be evaluated.
 * @return the relative likelihood of the sample.
 */
double GaussianDistribution::evaluatePdf(const Eigen::VectorXd& x) const {
  CHECK_EQ(this->dimension(), x.size());

  double denominator = std::sqrt(
      (std::pow(2 * M_PI, this->dimension()) * covariance_).determinant());
  Eigen::VectorXd deviation = x - mean_;
  double squared_mahalanobis_distance =
      deviation.transpose() * covariance_.inverse() * deviation;

  return std::exp(-squared_mahalanobis_distance / 2) / denominator;
}

Eigen::VectorXd GaussianDistribution::evaluatePdfVectorized(
    const Eigen::MatrixXd& x) const {
  CHECK_EQ(this->dimension(), x.rows());

  double denominator = std::sqrt(
      (std::pow(2 * M_PI, this->dimension()) * covariance_).determinant());

  Eigen::MatrixXd deviations = x.colwise() - mean_;
  Eigen::VectorXd squared_mahalanobis_distances =
      (deviations.transpose() * covariance_.inverse() * deviations).diagonal();

  return (-squared_mahalanobis_distances / 2).array().exp() / denominator;
}

/**
 * Checks for right dimensionality.
 *
 * @param right_side Distribution which will be added to `this`.
 * @return `*this`.
 */
GaussianDistribution& GaussianDistribution::operator+=(
    const GaussianDistribution& right_side) {
  CHECK_EQ(this->dimension(), right_side.dimension())
      << "Distribution dimensions do not match.";

  this->mean_ += right_side.mean_;
  this->covariance_ += right_side.covariance_;

  return *this;
}

/**
 * Checks for right dimensionality.
 *
 * @param right_side Distribution which will be subtracted from `this`.
 * @return `*this`.
 */
GaussianDistribution& GaussianDistribution::operator-=(
    const GaussianDistribution& right_side) {
  CHECK_EQ(this->dimension(), right_side.dimension())
      << "Distribution dimensions do not match.";

  this->mean_ -= right_side.mean_;
  this->covariance_ += right_side.covariance_;

  return *this;
}

/**
 * @param right_side Distribution which will be added to `this`.
 * @return new distribution which is the sum of `this` and @e right_side.
 */
GaussianDistribution GaussianDistribution::operator+(
    const GaussianDistribution& right_side) {
  GaussianDistribution result = *this;
  result += right_side;
  return result;
}

/**
 * @param right_side Distribution which will be subtracted from `this`.
 * @return new distribution which is the difference between `this` and
 *         @e right_side.
 */
GaussianDistribution GaussianDistribution::operator-(
    const GaussianDistribution& right_side) {
  GaussianDistribution result = *this;
  result -= right_side;
  return result;
}

}  // namespace refill
