#ifndef REFILL_DISTRIBUTIONS_GAUSSIAN_DISTRIBUTION_H_
#define REFILL_DISTRIBUTIONS_GAUSSIAN_DISTRIBUTION_H_

#include <Eigen/Dense>
#include <glog/logging.h>
#include <stdlib.h>

#include "refill/distributions/distribution_base.h"

namespace refill {

class GaussianDistribution : public DistributionBase<GaussianDistribution> {
 public:
  GaussianDistribution();
  GaussianDistribution(const GaussianDistribution& dist) = default;
  explicit GaussianDistribution(const int& dimension);
  GaussianDistribution(const Eigen::VectorXd& dist_mean,
                       const Eigen::MatrixXd& dist_cov);

  void setDistParam(const Eigen::VectorXd& dist_mean,
                    const Eigen::MatrixXd& dist_cov);

  void setMean(const Eigen::VectorXd& mean);
  void setCov(const Eigen::MatrixXd& cov);

  size_t dimension() const;
  Eigen::VectorXd mean() const;
  Eigen::MatrixXd cov() const;

  GaussianDistribution operator+(const GaussianDistribution& right_side);

 private:
  Eigen::VectorXd mean_;
  Eigen::MatrixXd covariance_;
};

}  // namespace refill

#endif  // REFILL_DISTRIBUTIONS_GAUSSIAN_DISTRIBUTION_H_
