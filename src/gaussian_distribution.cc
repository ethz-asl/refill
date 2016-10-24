#include "refill/gaussian_distribution.h"

#include <glog/logging.h>

namespace refill {

GaussianDistribution::GaussianDistribution()
    : mean_(Eigen::VectorXd::Zero(1)),
      covmat_(Eigen::MatrixXd::Identity(1, 1)) {}

GaussianDistribution::GaussianDistribution(Eigen::VectorXd dist_mean,
                                           Eigen::MatrixXd dist_cov) {
  SetDistParam(dist_mean, dist_cov);
}

void GaussianDistribution::SetDistParam(Eigen::VectorXd dist_mean,
                                        Eigen::MatrixXd dist_cov) {
  CHECK_EQ(dist_mean.size(), dist_cov.rows());
  CHECK_EQ(dist_cov.rows(), dist_mean.cols());

  Eigen::LLT<Eigen::MatrixXd> chol_of_cov(dist_cov);
  CHECK(chol_of_cov.info() != Eigen::NumericalIssue) << "Matrix not s.p.d.";

  mean_ = dist_mean;
  covmat_ = dist_cov;
}

GaussianDistribution GaussianDistribution::operator+(
    const GaussianDistribution& right_side) {
  CHECK(right_side.dim() == dim()) << "Distribution dimensions do not match.";
  GaussianDistribution result(mean_ + right_side.mean(),
                              covmat_ + right_side.cov());
  return result;
}

// Non-member overloaded operator for linear transforms of Gaussian random
// vectors.
GaussianDistribution operator*(const Eigen::MatrixXd& mat,
                               const GaussianDistribution gaussian) {
  CHECK_EQ(mat.cols(), gaussian.dim());
  return GaussianDistribution(mat * gaussian.mean(),
                              mat * gaussian.cov() * mat.transpose());
}

}  // namespace refill
