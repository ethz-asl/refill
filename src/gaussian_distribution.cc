#include "refill/gaussian_distribution.h"

namespace refill {

GaussianDistribution::GaussianDistribution()
    : mean_(Eigen::VectorXd::Zero(1)),
      covmat_(Eigen::MatrixXd::Identity(1, 1)) {}

GaussianDistribution::GaussianDistribution(Eigen::VectorXd mean,
                                           Eigen::MatrixXd covmat)
    : mean_(mean), covmat_(covmat) {}

Eigen::MatrixXd GaussianDistribution::GetCovariance() const { return covmat_; }

int GaussianDistribution::GetDimension() const { return mean_.size(); }

Eigen::VectorXd GaussianDistribution::GetMean() const { return mean_; }

GaussianDistribution GaussianDistribution::operator*(
    const Eigen::MatrixXd& mat) {
  return GaussianDistribution(mat * mean_, mat * covmat_ * mat.transpose());
}

GaussianDistribution GaussianDistribution::operator+(
    const GaussianDistribution& right_side) {
  GaussianDistribution result(mean_ + right_side.GetMean(),
                              covmat_ + right_side.GetCovariance());
  return result;
}

}  // namespace refill
