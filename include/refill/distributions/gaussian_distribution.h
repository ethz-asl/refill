#ifndef REFILL_DISTRIBUTIONS_GAUSSIAN_DISTRIBUTION_H_
#define REFILL_DISTRIBUTIONS_GAUSSIAN_DISTRIBUTION_H_

#include <glog/logging.h>

#include <Eigen/Dense>

#include <cstdlib>

#include "refill/distributions/distribution_base.h"

using std::size_t;

namespace refill {

class GaussianDistribution : public DistributionBase<GaussianDistribution> {
 public:
  // Default constructor creates standard normal gaussian.
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

// Non-member overloaded operator for linear transforms of Gaussian random
// vectors.
inline GaussianDistribution operator*(const Eigen::MatrixXd& mat,
                                      const GaussianDistribution gaussian) {
  CHECK_EQ(mat.cols(), gaussian.dimension());
  return GaussianDistribution(mat * gaussian.mean(),
                              mat * gaussian.cov() * mat.transpose());
}

inline GaussianDistribution operator*(const double& scalar,
                                      const GaussianDistribution& gaussian) {
  return GaussianDistribution(scalar * gaussian.mean(),
                              scalar * gaussian.cov() * scalar);
}

}  // namespace refill

#endif  // REFILL_DISTRIBUTIONS_GAUSSIAN_DISTRIBUTION_H_
