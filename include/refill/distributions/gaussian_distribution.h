#ifndef REFILL_DISTRIBUTIONS_GAUSSIAN_DISTRIBUTION_H_
#define REFILL_DISTRIBUTIONS_GAUSSIAN_DISTRIBUTION_H_

#include <Eigen/Dense>
#include <glog/logging.h>

#include <cstdlib>
#include <random>

#include "refill/distributions/distribution_base.h"

using std::size_t;

namespace refill {

/** @brief Class that implements a multivariate Gaussian distribution. */
class GaussianDistribution : public DistributionBase<GaussianDistribution> {
 public:
  /** @brief Default constructor creates standard normal gaussian. */
  GaussianDistribution();
  /** @brief Copy constrcutor. */
  GaussianDistribution(const GaussianDistribution& dist) = default;
  /** @brief Constructs a normal distribution with given dimension. */
  explicit GaussianDistribution(const int& dimension);
  /** @brief Constructs a gaussian distribution with given
   *         mean and covariance. */
  GaussianDistribution(const Eigen::VectorXd& dist_mean,
                       const Eigen::MatrixXd& dist_cov);
  virtual ~GaussianDistribution() = default;

  /** @brief Sets the distribution parameters. */
  void setDistributionParameters(const Eigen::VectorXd& dist_mean,
                                 const Eigen::MatrixXd& dist_cov);

  /** @brief Sets the mean to a new value. */
  void setMean(const Eigen::VectorXd& mean);
  /** @brief Sets the covariance matrix to a new value. */
  void setCov(const Eigen::MatrixXd& cov);

  /** @brief Returns the distributions dimension. */
  size_t dimension() const;
  /** @brief Returns the current mean of the distribution. */
  Eigen::VectorXd mean() const override;
  /** @brief Returns the current covariance of the distribution. */
  Eigen::MatrixXd cov() const override;

  /** @brief Returns a sample drawn from the distribution. */
  Eigen::VectorXd drawSample() override;

  /** @brief Evaluates the pdf at the given sample. */
  double evaluatePdf(const Eigen::VectorXd& x) const override;

  /** @brief Evaluates the pdf at the given samples, in a vectorized fashion. */
  Eigen::VectorXd evaluatePdfVectorized(const Eigen::MatrixXd& sampled_x) const
      override;

  /** @brief Implements the addition of a gaussian distribution to `this`. */
  GaussianDistribution& operator+=(const GaussianDistribution& right_side);

  /** @brief Implements the subtraction of a gaussian distribution of `this`. */
  GaussianDistribution& operator-=(const GaussianDistribution& right_side);

  /** @brief Implements the addition of two gaussian distributions. */
  GaussianDistribution operator+(const GaussianDistribution& right_side);

  /** @brief Implements the subtraction of two gaussian distributions. */
  GaussianDistribution operator-(const GaussianDistribution& right_side);


 private:
  Eigen::VectorXd mean_;
  Eigen::MatrixXd covariance_;
};

/**
 * @relates GaussianDistribution
 *
 * @brief Non-member overloaded operator for linear transformations of Gaussian
 *        random vectors.
 *
 * @param mat Linear transformation matrix.
 * @param gaussian Gaussian random vector.
 * @return transformed Gaussian random vector.
 */
inline GaussianDistribution operator*(const Eigen::MatrixXd &mat,
                                      const GaussianDistribution gaussian) {
  CHECK_EQ(mat.cols(), gaussian.dimension());
  return GaussianDistribution(mat * gaussian.mean(),
                              mat * gaussian.cov() * mat.transpose());
}

/**
 * @relates GaussianDistribution
 *
 * @brief Non-member overoad operator for scalar multiplication of Gaussian random vectors.
 *
 * @param scalar Scalar scaling factor.
 * @param gaussian Gaussian random vector.
 * @return transformed Gaussian random vector.
 */
inline GaussianDistribution operator*(const double& scalar,
                                      const GaussianDistribution& gaussian) {
  return GaussianDistribution(scalar * gaussian.mean(),
                              scalar * gaussian.cov() * scalar);
}

}  // namespace refill

#endif  // REFILL_DISTRIBUTIONS_GAUSSIAN_DISTRIBUTION_H_
