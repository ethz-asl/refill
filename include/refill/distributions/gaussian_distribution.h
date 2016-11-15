#ifndef REFILL_DISTRIBUTIONS_GAUSSIAN_DISTRIBUTION_H_
#define REFILL_DISTRIBUTIONS_GAUSSIAN_DISTRIBUTION_H_

#include <glog/logging.h>

#include "refill/distributions/distribution_base.h"

namespace refill {

class GaussianDistribution : public DistributionBase<GaussianDistribution> {
 public:
  GaussianDistribution();
  GaussianDistribution(const GaussianDistribution& dist);
  explicit GaussianDistribution(const int& dimension);
  GaussianDistribution(const Eigen::VectorXd& dist_mean,
                       const Eigen::MatrixXd& dist_cov);

  void setDistParam(const Eigen::VectorXd& dist_mean,
                    const Eigen::MatrixXd& dist_cov);

  int dimension() const {
    return mean_.size();
  }
  Eigen::VectorXd mean() const {
    return mean_;
  }
  Eigen::MatrixXd cov() const {
    return covmat_;
  }
  void setMean(const Eigen::VectorXd& mean);
  void setCov(const Eigen::MatrixXd& cov) {
    covmat_ = cov;
  }

  GaussianDistribution operator+(const GaussianDistribution& right_side);

 private:
  Eigen::MatrixXd covmat_;
  Eigen::VectorXd mean_;
};

// Non-member operator overloading.
GaussianDistribution operator*(const Eigen::MatrixXd& mat,
                               const GaussianDistribution gaussian);

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
  CHECK_NE(chol_of_cov.info(), Eigen::NumericalIssue) << "Matrix not s.p.d.";

  covmat_ = cov;
}

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

}  // namespace refill

#endif  // REFILL_DISTRIBUTIONS_GAUSSIAN_DISTRIBUTION_H_
