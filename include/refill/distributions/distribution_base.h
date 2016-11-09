#ifndef REFILL_DISTRIBUTIONS_DISTRIBUTION_BASE_H_
#define REFILL_DISTRIBUTIONS_DISTRIBUTION_BASE_H_

#include <Eigen/Dense>

namespace refill {

template <int DIM>
class DistributionBase {
 public:
  virtual Eigen::Matrix<double, DIM, 1> mean() const = 0;
  virtual Eigen::Matrix<double, DIM, DIM> cov() const = 0;
  virtual DistributionBase* Clone() const = 0;
};

}  // namespace refill

#endif  // REFILL_DISTRIBUTIONS_DISTRIBUTION_BASE_H_
