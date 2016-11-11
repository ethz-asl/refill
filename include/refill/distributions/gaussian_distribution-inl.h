#ifndef REFILL_DISTRIBUTIONS_GAUSSIAN_DISTRIBUTION_INL_H_
#define REFILL_DISTRIBUTIONS_GAUSSIAN_DISTRIBUTION_INL_H_

#include <glog/logging.h>

namespace refill {

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
  SetDistParam(dist_mean, dist_cov);
}

template <int DIM>
void GaussianDistribution<DIM>::SetDistParam(
    Eigen::Matrix<double, DIM, 1> dist_mean,
    Eigen::Matrix<double, DIM, DIM> dist_cov) {
  if (DIM == Eigen::Dynamic) {
    CHECK_EQ(dist_mean.size(), dist_cov.rows());
    CHECK_EQ(dist_cov.rows(), dist_cov.cols());
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

#endif  // REFILL_DISTRIBUTIONS_GAUSSIAN_DISTRIBUTION_INL_H_
