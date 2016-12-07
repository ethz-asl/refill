#include "refill/distributions/gaussian_distribution.h"

using std::size_t;

namespace refill {

GaussianDistribution::GaussianDistribution()
    : GaussianDistribution(0) {}

GaussianDistribution::GaussianDistribution(const int& dimension)
    : mean_(Eigen::VectorXd::Zero(dimension)),
      covariance_(Eigen::MatrixXd::Identity(dimension, dimension)) {}

GaussianDistribution::GaussianDistribution(const Eigen::VectorXd& dist_mean,
                                           const Eigen::MatrixXd& dist_cov) {
  setDistParam(dist_mean, dist_cov);
}

void GaussianDistribution::setDistParam(const Eigen::VectorXd& dist_mean,
                                        const Eigen::MatrixXd& dist_cov) {
  CHECK_EQ(dist_mean.rows(), dist_cov.rows());
  CHECK_EQ(dist_cov.rows(), dist_cov.cols());

  Eigen::LLT<Eigen::MatrixXd> chol_of_cov(dist_cov);
  CHECK(chol_of_cov.info() != Eigen::NumericalIssue) << "Matrix not s.p.d.";

  mean_ = dist_mean;
  covariance_ = dist_cov;
}

void GaussianDistribution::setMean(const Eigen::VectorXd& mean) {
  CHECK_EQ(this->dimension(), mean.rows());
  mean_ = mean;
}

void GaussianDistribution::setCov(const Eigen::MatrixXd& cov) {
  CHECK_EQ(this->dimension(), cov.rows());
  CHECK_EQ(cov.rows(), cov.cols());

  Eigen::LLT<Eigen::MatrixXd> chol_of_cov(cov);
  CHECK_NE(chol_of_cov.info(), Eigen::NumericalIssue)<< "Matrix not s.p.d.";

  covariance_ = cov;
}


size_t GaussianDistribution::dimension() const {
  return mean_.rows();
}

Eigen::VectorXd GaussianDistribution::mean() const {
  return mean_;
}

Eigen::MatrixXd GaussianDistribution::cov() const {
  return covariance_;
}

Eigen::VectorXd GaussianDistribution::drawSample() const {
  CHECK_NE(mean_.rows(), 0)
      << "[GaussianDistribution] Distribution parameters have not been set.";

  // Generate normal distributed random vector
  std::random_device true_rng;
  std::default_random_engine generator(true_rng);
  std::normal_distribution<double> normal_dist(0.0, 1.0);

  Eigen::VectorXd uniform_random_vector(mean_.rows());
  for (int i=0; i < mean_.rows(); ++i) {
    uniform_random_vector[i] = normal_dist(generator);
  }

  // Calculate matrix L
  Eigen::LLT<Eigen::MatrixXd> chol_of_cov(covariance_);
  Eigen::MatrixXd L = chol_of_cov.matrixL();

  return mean_ + L * uniform_random_vector;
}

GaussianDistribution GaussianDistribution::operator+(
    const GaussianDistribution &right_side) {

  CHECK(right_side.dimension() == this->dimension())
      << "Distribution dimensions do not match.";

  GaussianDistribution result(mean_ + right_side.mean(),
                              covariance_ + right_side.cov());
  return result;
}

// Non-member overloaded operator for linear transforms of Gaussian random
// vectors.
inline GaussianDistribution operator*(const Eigen::MatrixXd &mat,
                                      const GaussianDistribution gaussian) {
  CHECK_EQ(mat.cols(), gaussian.dimension());
  return GaussianDistribution(mat * gaussian.mean(),
                              mat * gaussian.cov() * mat.transpose());
}

}  // namespace refill
