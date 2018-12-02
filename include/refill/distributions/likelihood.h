#ifndef REFILL_DISTRIBUTIONS_LIKELIHOOD_H_
#define REFILL_DISTRIBUTIONS_LIKELIHOOD_H_

#include <Eigen/Dense>

namespace refill {

class Likelihood {
 public:
  virtual ~Likelihood() = default;

  /**
   * @brief Computes the likelihood of a measurement given the state.
   * 
   * Computes the likelihood of measurement @f$ m @f$ given a certain state
   * @f$ x @f$. Meaning it computes:
   * 
   * @f$ P(m | x) @f$
   *
   * @param state State vector which is assumed to be given.
   * @param measurement Measurement for which the likelihood will be computed.
   * @return the likelihood of the input vector given the measurement.
   */
  virtual double getLikelihood(const Eigen::VectorXd& state,
                               const Eigen::VectorXd& measurement) const = 0;

  virtual Eigen::VectorXd getLikelihoodVectorized(
      const Eigen::MatrixXd& sampled_state,
      const Eigen::VectorXd& measurement) const;
};

}  // namespace refill

#endif  // REFILL_DISTRIBUTIONS_LIKELIHOOD_H_
