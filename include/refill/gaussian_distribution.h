#ifndef GAUSSIAN_DISTRIBUTION_H_
#define GAUSSIAN_DISTRIBUTION_H_

#include <eigen3/Eigen/Dense>

namespace refill {

class GaussianDistribution {
 public:
  GaussianDistribution();
  GaussianDistribution(Eigen::VectorXd dist_mean, Eigen::MatrixXd dist_cov);

  Eigen::MatrixXd cov() const { return covmat_; }
  int dim() const { return mean_.size(); }
  Eigen::VectorXd mean() const { return mean_; }

  GaussianDistribution operator*(const Eigen::MatrixXd& mat);
  GaussianDistribution operator+(const GaussianDistribution& right_side);

 private:
  Eigen::MatrixXd covmat_;
  Eigen::VectorXd mean_;
};

}  // namespace refill

#endif  // GAUSSIAN_DISTRIBUTION_H_
