#ifndef INCLUDE_REFILL_GAUSSIAN_DISTRIBUTIONINL_INL_H_
#define INCLUDE_REFILL_GAUSSIAN_DISTRIBUTIONINL_INL_H_

#include <glog/logging.h>

namespace refill {

template <int DIM>
GaussianDistribution<DIM>::GaussianDistribution()
    : mean_(Eigen::VectorXd::Zero(1)),
      covmat_(Eigen::MatrixXd::Identity(1, 1)) {}

template <int DIM>
GaussianDistribution<DIM>::GaussianDistribution(Eigen::VectorXd dist_mean,
                                           Eigen::MatrixXd dist_cov) {
  SetDistParam(dist_mean, dist_cov);
}

template <int DIM>
void GaussianDistribution<DIM>::SetDistParam(Eigen::VectorXd dist_mean,
                                        Eigen::MatrixXd dist_cov) {
  CHECK_EQ(dist_mean.size(), dist_cov.rows());
  CHECK_EQ(dist_cov.rows(), dist_mean.cols());

  Eigen::LLT<Eigen::MatrixXd> chol_of_cov(dist_cov);
  CHECK(chol_of_cov.info() != Eigen::NumericalIssue) << "Matrix not s.p.d.";

  mean_ = dist_mean;
  covmat_ = dist_cov;
}

template <int DIM>
GaussianDistribution<DIM> GaussianDistribution<DIM>::operator+(
    const GaussianDistribution<DIM>& right_side) {
  CHECK(right_side.dim() == dim()) << "Distribution dimensions do not match.";
  GaussianDistribution result(mean_ + right_side.mean(),
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

#endif // INCLUDE_REFILL_GAUSSIAN_DISTRIBUTIONINL_INL_H_

