#include "refill/gaussian_distribution.h"

namespace refill {

GaussianDistribution::GaussianDistribution()
    : mean_(Eigen::VectorXd::Zero(1)),
      covmat_(Eigen::MatrixXd::Identity(1, 1)) {}

GaussianDistribution::GaussianDistribution(Eigen::VectorXd dist_mean,
                                           Eigen::MatrixXd dist_cov)
    : mean_(dist_mean), covmat_(dist_cov) {}

GaussianDistribution GaussianDistribution::operator*(
    const Eigen::MatrixXd& mat) {
  return GaussianDistribution(mat * mean_, mat * covmat_ * mat.transpose());
}

GaussianDistribution GaussianDistribution::operator+(
    const GaussianDistribution& right_side) {
  GaussianDistribution result(mean_ + right_side.mean(),
                              covmat_ + right_side.cov());
  return result;
}

}  // namespace refill
