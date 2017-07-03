#ifndef REFILL_DISTRIBUTIONS_LIKELIHOOD_H_
#define REFILL_DISTRIBUTIONS_LIKELIHOOD_H_

#include <Eigen/Dense>

namespace refill {

class Likelihood {
 public:
  virtual ~Likelihood() = default;

  /**
   * @brief Computes the likelihood of a vector.
   *
   * @param x Vector for which the likelihood shall be computed.
   * @return the likelihood of the input vector.
   */
  virtual double getLikelihood(const Eigen::VectorXd& x) const = 0;
};

}  // namespace refill

#endif  // REFILL_DISTRIBUTIONS_LIKELIHOOD_H_
