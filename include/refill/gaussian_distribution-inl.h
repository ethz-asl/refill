#ifndef INCLUDE_REFILL_GAUSSIAN_DISTRIBUTION_INL_H_
#define INCLUDE_REFILL_GAUSSIAN_DISTRIBUTION_INL_H_

#include <glog/logging.h>

namespace refill {

template <int DIM>
GaussianDistribution<DIM>::GaussianDistribution()
    : mean_(Eigen::VectorXd::Zero(DIM)),
      covmat_(Eigen::MatrixXd::Identity(DIM, DIM)) {}

template <int DIM>
GaussianDistribution<DIM>::GaussianDistribution(Eigen::VectorXd dist_mean,
                                           Eigen::MatrixXd dist_cov) {
  SetDistParam(dist_mean, dist_cov);
}

template <int DIM>
void GaussianDistribution<DIM>::SetDistParam(
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
    const GaussianDistribution<DIM>& right_side) {
  if (DIM == Eigen::Dynamic) {
    CHECK(right_side.dim() == dim()) << "Distribution dimensions do not match.";
  }
  GaussianDistribution<> result(mean_ + right_side.mean(),
                                covmat_ + right_side.cov());
  return result;
}

// Non-member overloaded operator for linear transforms of Gaussian random
// vectors.
GaussianDistribution<> operator*(const Eigen::MatrixXd& mat,
                                 const GaussianDistribution<> gaussian) {
  CHECK_EQ(mat.cols(), gaussian.dim());
  return GaussianDistribution<>(mat * gaussian.mean(),
                                mat * gaussian.cov() * mat.transpose());
}

}  // namespace refill

#endif  // INCLUDE_REFILL_GAUSSIAN_DISTRIBUTION_INL_H_

