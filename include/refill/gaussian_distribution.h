#ifndef GAUSSIAN_DISTRIBUTION_H_
#define GAUSSIAN_DISTRIBUTION_H_

#include <eigen3/Eigen/Dense>

namespace refill {

class GaussianDistribution {
 public:
  GaussianDistribution();
  GaussianDistribution(Eigen::VectorXd mean, Eigen::MatrixXd covmat);

  Eigen::MatrixXd GetCovariance() const;
  int GetDimension() const;
  Eigen::VectorXd GetMean() const;

  GaussianDistribution operator*(const Eigen::MatrixXd& mat);
  GaussianDistribution operator+(const GaussianDistribution& right_side);

 private:
  Eigen::MatrixXd covmat_;
  Eigen::VectorXd mean_;
};

}  // namespace refill

#endif  // GAUSSIAN_DISTRIBUTION_H_
