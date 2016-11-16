#include "refill/distributions/gaussian_distribution.h"

namespace refill {

GaussianDistribution::GaussianDistribution() {
  // For the standard constructor, we use a univariate standard normal
  // distribution.
  mean_ = Eigen::VectorXd::Zero(1);
  covmat_ = Eigen::MatrixXd::Identity(1, 1);
}

GaussianDistribution::GaussianDistribution(const GaussianDistribution& dist) {
  covmat_ = dist.covmat_;
  mean_ = dist.mean_;
}

GaussianDistribution::GaussianDistribution(const int& dimension) {
  mean_ = Eigen::VectorXd::Zero(dimension);
  covmat_ = Eigen::MatrixXd::Identity(dimension, dimension);
}

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
  covmat_ = dist_cov;
}

void GaussianDistribution::setMean(const Eigen::VectorXd& mean) {
  CHECK_EQ(this->dimension(), mean.size());
  mean_ = mean;
}

void GaussianDistribution::setCov(const Eigen::MatrixXd& cov) {
  CHECK_EQ(this->dimension(), cov.rows());
  CHECK_EQ(cov.rows(), cov.cols());

  Eigen::LLT<Eigen::MatrixXd> chol_of_cov(cov);
  CHECK_NE(chol_of_cov.info(), Eigen::NumericalIssue)<< "Matrix not s.p.d.";

  covmat_ = cov;
}

int GaussianDistribution::dimension() const {
  return mean_.size();
}

Eigen::VectorXd GaussianDistribution::mean() const {
  return mean_;
}

Eigen::MatrixXd GaussianDistribution::cov() const {
  return covmat_;
}

// Operator overloading
GaussianDistribution GaussianDistribution::operator+(
    const GaussianDistribution &right_side) {

  CHECK(right_side.dimension() == this->dimension())
      << "Distribution dimensions do not match.";

  GaussianDistribution result(mean_ + right_side.mean(),
                              covmat_ + right_side.cov());
  return result;
}

// Non-member overloaded operator for linear transforms of Gaussian random
// vectors.
inline GaussianDistribution operator*(const Eigen::MatrixXd &mat,
                                      const GaussianDistribution gaussian) {
  CHECK_EQ(mat.cols(), gaussian.dimension());
  return GaussianDistribution(mat * gaussian.mean(),
                              mat * gaussian.cov() * mat.transpose());
}
// Non-member operator overloading.
GaussianDistribution operator*(const Eigen::MatrixXd& mat,
                               const GaussianDistribution gaussian);

}  // namespace refill
