#ifndef REFILL_DISTRIBUTIONS_LIKELIHOOD_H_
#define REFILL_DISTRIBUTIONS_LIKELIHOOD_H_

#include <Eigen/Dense>

namespace refill {

class Likelihood {
 public:
  virtual ~Likelihood() = default;

  virtual double getLikelihood(const Eigen::VectorXd& x) const = 0;
};

}  // namespace refill

#endif  // REFILL_DISTRIBUTIONS_LIKELIHOOD_H_
