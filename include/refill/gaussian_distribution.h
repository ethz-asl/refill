#ifndef INCLUDE_REFILL_GAUSSIAN_DISTRIBUTION_H_
#define INCLUDE_REFILL_GAUSSIAN_DISTRIBUTION_H_

#include <Eigen/Dense>

namespace refill {

class GaussianDistribution {
 public:
  GaussianDistribution();
  GaussianDistribution(Eigen::VectorXd dist_mean, Eigen::MatrixXd dist_cov);

  void SetDistParam(Eigen::VectorXd dist_mean, Eigen::MatrixXd dist_cov);

  Eigen::MatrixXd cov() const { return covmat_; }
  int dim() const { return mean_.size(); }
  Eigen::VectorXd mean() const { return mean_; }

  GaussianDistribution operator+(const GaussianDistribution& right_side);

 private:
  Eigen::MatrixXd covmat_;
  Eigen::VectorXd mean_;
};

// Non-member operator overloading.
GaussianDistribution operator*(const Eigen::MatrixXd& mat,
                               const GaussianDistribution gaussian);

}  // namespace refill

#endif  // INCLUDE_REFILL_GAUSSIAN_DISTRIBUTION_H_
