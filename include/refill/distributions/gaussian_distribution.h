#ifndef REFILL_DISTRIBUTIONS_GAUSSIAN_DISTRIBUTION_H_
#define REFILL_DISTRIBUTIONS_GAUSSIAN_DISTRIBUTION_H_

#include "refill/distributions/distribution_base.h"

namespace refill {

template<int DIM = Eigen::Dynamic>
class GaussianDistribution : public DistributionBase<DIM,
    GaussianDistribution<DIM>> {
 public:
  GaussianDistribution();
  GaussianDistribution(const GaussianDistribution& dist);
  GaussianDistribution(Eigen::Matrix<double, DIM, 1> dist_mean,
                       Eigen::Matrix<double, DIM, DIM> dist_cov);

  void SetDistParam(Eigen::Matrix<double, DIM, 1> dist_mean,
                    Eigen::Matrix<double, DIM, DIM> dist_cov);

  int dim() const { return mean_.size(); }
  Eigen::Matrix<double, DIM, 1> mean() const { return mean_; }
  Eigen::Matrix<double, DIM, DIM> cov() const { return covmat_; }
  void SetMean(const Eigen::Matrix<double, DIM, 1>& mean) { mean_ = mean; }
  void SetCov(const Eigen::Matrix<double, DIM, DIM>& cov) { covmat_ = cov; }

  GaussianDistribution operator+(const GaussianDistribution<DIM>& right_side);

 private:
  Eigen::Matrix<double, DIM, DIM> covmat_;
  Eigen::Matrix<double, DIM, 1> mean_;
};

// Alias for a dynamic size version of the Gaussian distribution. Notation
// compatible to Eigen.
using GaussianDistributionXd = GaussianDistribution<Eigen::Dynamic>;

// Non-member operator overloading.
template<int DIM = Eigen::Dynamic>
GaussianDistribution<DIM> operator*(const Eigen::MatrixXd& mat,
                                    const GaussianDistribution<DIM> gaussian);

}  // namespace refill

#include "./gaussian_distribution-inl.h"

#endif  // REFILL_DISTRIBUTIONS_GAUSSIAN_DISTRIBUTION_H_
