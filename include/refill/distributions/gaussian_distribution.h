#ifndef REFILL_DISTRIBUTIONS_GAUSSIAN_DISTRIBUTION_H_
#define REFILL_DISTRIBUTIONS_GAUSSIAN_DISTRIBUTION_H_

#include <glog/logging.h>
#include <algorithm>
#include <random>

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

  int dimension() const;
  Eigen::VectorXd mean() const;
  Eigen::MatrixXd cov() const;

  Eigen::VectorXd drawSample() const;

  GaussianDistribution operator+(const GaussianDistribution& right_side);

 private:
  Eigen::MatrixXd covariance_;
  Eigen::VectorXd mean_;
};

}  // namespace refill

#endif  // REFILL_DISTRIBUTIONS_GAUSSIAN_DISTRIBUTION_H_
