#include "refill/distributions/gaussian_distribution.h"

using std::size_t;

namespace refill {

GaussianDistribution::GaussianDistribution() : GaussianDistribution(1) {}

GaussianDistribution::GaussianDistribution(const int& dimension)
    : mean_(Eigen::VectorXd::Zero(dimension)),
      covariance_(Eigen::MatrixXd::Identity(dimension, dimension)) {}

GaussianDistribution::GaussianDistribution(const Eigen::VectorXd& dist_mean,
                                           const Eigen::MatrixXd& dist_cov) {
  setDistParam(dist_mean, dist_cov);
}

void GaussianDistribution::setDistParam(const Eigen::VectorXd& dist_mean,
                                        const Eigen::MatrixXd& dist_cov) {
  CHECK_EQ(dist_mean.size(), dist_cov.rows());
  CHECK_EQ(dist_cov.rows(), dist_cov.cols());

  Eigen::LLT<Eigen::MatrixXd> chol_of_cov(dist_cov);
  CHECK(chol_of_cov.info() != Eigen::NumericalIssue) << "Matrix not s.p.d.";

  mean_ = dist_mean;
  covariance_ = dist_cov;
}

void GaussianDistribution::setMean(const Eigen::VectorXd& mean) {
  CHECK_EQ(this->dimension(), mean.size());
  mean_ = mean;
}

void GaussianDistribution::setCov(const Eigen::MatrixXd& cov) {
  CHECK_EQ(this->dimension(), cov.rows());
  CHECK_EQ(cov.rows(), cov.cols());

  Eigen::LLT<Eigen::MatrixXd> chol_of_cov(cov);
  CHECK_NE(chol_of_cov.info(), Eigen::NumericalIssue) << "Matrix not s.p.d.";

  covariance_ = cov;
}

size_t GaussianDistribution::dimension() const { return mean_.size(); }

Eigen::VectorXd GaussianDistribution::mean() const { return mean_; }

Eigen::MatrixXd GaussianDistribution::cov() const { return covariance_; }

GaussianDistribution GaussianDistribution::operator+(
    const GaussianDistribution& right_side) {
  CHECK(right_side.dimension() == this->dimension())
      << "Distribution dimensions do not match.";

  GaussianDistribution result(mean_ + right_side.mean(),
                              covariance_ + right_side.cov());
  return result;
}

// Non-member overloaded operator for linear transforms of Gaussian random
// vectors.
inline GaussianDistribution operator*(const Eigen::MatrixXd& mat,
                                      const GaussianDistribution gaussian) {
  CHECK_EQ(mat.cols(), gaussian.dimension());
  return GaussianDistribution(mat * gaussian.mean(),
                              mat * gaussian.cov() * mat.transpose());
}

}  // namespace refill
