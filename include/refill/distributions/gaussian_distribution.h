#ifndef REFILL_DISTRIBUTIONS_GAUSSIAN_DISTRIBUTION_H_
#define REFILL_DISTRIBUTIONS_GAUSSIAN_DISTRIBUTION_H_

#include <glog/logging.h>

#include <Eigen/Dense>

#include <cstdlib>

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

  /** @brief Sets the distribution parameters. */
  void setDistParam(const Eigen::VectorXd& dist_mean,
                    const Eigen::MatrixXd& dist_cov);

  /** @brief Sets the mean to a new value. */
  void setMean(const Eigen::VectorXd& mean);
  /** @brief Sets the covariance matrix to a new value. */
  void setCov(const Eigen::MatrixXd& cov);

  /** @brief Returns the distributions dimension. */
    size_t dimension() const;
  /** @brief Returns the current mean of the distribution. */
  Eigen::VectorXd mean() const;
  /** @brief Returns the current covariance of the distribution. */
  Eigen::MatrixXd cov() const;

  /** @brief Implements the addition of two gaussian distributions. */
  GaussianDistribution operator+(const GaussianDistribution& right_side);

 private:
  Eigen::VectorXd mean_;
  Eigen::MatrixXd covariance_;
};

}  // namespace refill

#endif  // REFILL_DISTRIBUTIONS_GAUSSIAN_DISTRIBUTION_H_
