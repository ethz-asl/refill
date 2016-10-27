#ifndef INCLUDE_REFILL_GAUSSIAN_DISTRIBUTION_H_
#define INCLUDE_REFILL_GAUSSIAN_DISTRIBUTION_H_

#include <Eigen/Dense>

namespace refill {

template <int DIM = Eigen::Dynamic>
class GaussianDistribution {
 public:
  GaussianDistribution();
  GaussianDistribution(Eigen::VectorXd dist_mean, Eigen::MatrixXd dist_cov);

  void SetDistParam(Eigen::Matrix<double, DIM, 1> dist_mean,
                    Eigen::Matrix<double, DIM, DIM> dist_cov);

  Eigen::Matrix<double, DIM, DIM> cov() const { return covmat_; }
  int dim() const { return mean_.size(); }
  Eigen::Matrix<double, DIM, 1> mean() const { return mean_; }

  GaussianDistribution operator+(const GaussianDistribution& right_side);

 private:
  Eigen::Matrix<double, DIM, DIM> covmat_;
  Eigen::Matrix<double, DIM, 1> mean_;
};

// Non-member operator overloading.
GaussianDistribution<> operator*(const Eigen::MatrixXd& mat,
                                 const GaussianDistribution<> gaussian);

}  // namespace refill

#include "./gaussian_distribution-inl.h"

#endif  // INCLUDE_REFILL_GAUSSIAN_DISTRIBUTION_H_
