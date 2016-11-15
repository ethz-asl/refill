#ifndef REFILL_DISTRIBUTIONS_GAUSSIAN_DISTRIBUTION_H_
#define REFILL_DISTRIBUTIONS_GAUSSIAN_DISTRIBUTION_H_

#include <glog/logging.h>

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

  void setDistParam(Eigen::Matrix<double, DIM, 1> dist_mean,
                    Eigen::Matrix<double, DIM, DIM> dist_cov);

  int dim() const { return mean_.size(); }
  Eigen::Matrix<double, DIM, 1> mean() const { return mean_; }
  Eigen::Matrix<double, DIM, DIM> cov() const { return covmat_; }
  void setMean(const Eigen::Matrix<double, DIM, 1>& mean) { mean_ = mean; }
  void setCov(const Eigen::Matrix<double, DIM, DIM>& cov) { covmat_ = cov; }

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

template <int DIM>
GaussianDistribution<DIM>::GaussianDistribution() {
  // In case of dynamic size matrices, we use a univariate standard normal
  // distribution as default.
  if (DIM == Eigen::Dynamic) {
    mean_ = Eigen::Matrix<double, DIM, 1>::Zero(1);
    covmat_ = Eigen::Matrix<double, DIM, DIM>::Identity(1, 1);
  } else {
    mean_ = Eigen::Matrix<double, DIM, 1>::Zero(DIM);
    covmat_ = Eigen::Matrix<double, DIM, DIM>::Identity(DIM, DIM);
  }
}

template <int DIM>
GaussianDistribution<DIM>::GaussianDistribution(
    const GaussianDistribution &dist) {
  covmat_ = dist.covmat_;
  mean_ = dist.mean_;
}

template <int DIM>
GaussianDistribution<DIM>::GaussianDistribution(
    Eigen::Matrix<double, DIM, 1> dist_mean,
    Eigen::Matrix<double, DIM, DIM> dist_cov) {
  setDistParam(dist_mean, dist_cov);
}

template <int DIM>
void GaussianDistribution<DIM>::setDistParam(
    Eigen::Matrix<double, DIM, 1> dist_mean,
    Eigen::Matrix<double, DIM, DIM> dist_cov) {
  if (DIM == Eigen::Dynamic) {
    CHECK_EQ(dist_mean.size(), dist_cov.rows());
    CHECK_EQ(dist_cov.rows(), dist_mean.cols());
  }

  Eigen::LLT<Eigen::Matrix<double, DIM, DIM>> chol_of_cov(dist_cov);
  CHECK(chol_of_cov.info() != Eigen::NumericalIssue) << "Matrix not s.p.d.";

  mean_ = dist_mean;
  covmat_ = dist_cov;
}

template <int DIM>
GaussianDistribution<DIM> GaussianDistribution<DIM>::operator+(
    const GaussianDistribution<DIM> &right_side) {
  if (DIM == Eigen::Dynamic) {
    CHECK(right_side.dim() == dim()) << "Distribution dimensions do not match.";
  }
  GaussianDistribution<DIM> result(mean_ + right_side.mean(),
                                   covmat_ + right_side.cov());
  return result;
}

// Non-member overloaded operator for linear transforms of Gaussian random
// vectors.
template <int DIM = Eigen::Dynamic>
inline GaussianDistribution<DIM> operator*(
    const Eigen::MatrixXd &mat, const GaussianDistribution<DIM> gaussian) {
  CHECK_EQ(mat.cols(), gaussian.dim());
  return GaussianDistribution<DIM>(mat * gaussian.mean(),
                                   mat * gaussian.cov() * mat.transpose());
}

}  // namespace refill

#endif  // REFILL_DISTRIBUTIONS_GAUSSIAN_DISTRIBUTION_H_
