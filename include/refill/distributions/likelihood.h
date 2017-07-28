#ifndef REFILL_DISTRIBUTIONS_LIKELIHOOD_H_
#define REFILL_DISTRIBUTIONS_LIKELIHOOD_H_

#include <Eigen/Dense>

namespace refill {

class Likelihood {
 public:
  virtual ~Likelihood() = default;

  /**
   * @brief Computes the likelihood of a state vector.given a measurement
   *
   * @param state Vector for which the likelihood shall be computed.
   * @param measurement Measurement which is assumed to be given.
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
