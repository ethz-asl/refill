#ifndef REFILL_DISTRIBUTIONS_DISTRIBUTION_BASE_H_
#define REFILL_DISTRIBUTIONS_DISTRIBUTION_BASE_H_

#include <Eigen/Dense>
#include <glog/logging.h>
#include <random>
#include <typeinfo>

namespace refill {

/**
 * @brief Interface for distributions
 *
 * Interface that should be used for defining new distributions.
 */
class DistributionInterface {
 public:
  /**
   * @brief Returns the mean of the distribution.
   *
   * @return mean of the distribution.
   */
  virtual Eigen::VectorXd mean() const = 0;

  /**
   * @brief Returns the covariance matrix of the distribution.
   *
   * @return the covariance matrix of the distribution.
   */
  virtual Eigen::MatrixXd cov() const = 0;

  /**
   * @brief Draws a random sample from the distribution.
   *
   * @return a sample drawn from the distribution.
   */
  virtual Eigen::VectorXd drawSample() = 0;

  /**
   * @brief Evaluate the distributions probability density function.
   *
   * @param x Point at which the pdf should be evaluated.
   * @return the pdfs value at the point x.
   */
  virtual double evaluatePdf(const Eigen::VectorXd& x) const = 0;

  /** @brief A vectorized version of the pdf evaluation. */
  virtual Eigen::VectorXd evaluatePdfVectorized(
      const Eigen::MatrixXd& sampled_x) const;

  /**
   * @brief Clones the original distribution.
   *
   * @return a pointer to the cloned distribution.
   */
  virtual DistributionInterface* clone() const = 0;

  /**
   * @brief Random number generator used to draw from distribution.
   */
  std::mt19937 rng_;
};

/**
 * @brief Class that implements the CRTP
 *
 * Class that implements the Curiously Recurring Templating Pattern
 * so the clone function doesn't have to be implemented in every
 * derived distribution.
 *
 * For new distributions, inherit from this class like this:
 *
 * `class NewDistribution : public DistributionBase<NewDistribution>`
 */
template<typename DERIVED>
class DistributionBase : public DistributionInterface {
  DistributionInterface* clone() const override {
    DERIVED casted_derived_obj;
    try {
      casted_derived_obj = dynamic_cast<DERIVED const&>(*this);
    } catch (const std::bad_cast& e) {
      LOG(FATAL)<< "Tried cloning, but encountered: " << e.what();
    }
    return new DERIVED(casted_derived_obj);
  }
};

}  // namespace refill

#endif  // REFILL_DISTRIBUTIONS_DISTRIBUTION_BASE_H_
