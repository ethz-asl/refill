#ifndef REFILL_DISTRIBUTIONS_DISTRIBUTION_BASE_H_
#define REFILL_DISTRIBUTIONS_DISTRIBUTION_BASE_H_

#include <Eigen/Dense>

namespace refill {

template <int DIM>
class DistributionInterface {
 public:
  virtual Eigen::Matrix<double, DIM, 1> mean() const = 0;
  virtual Eigen::Matrix<double, DIM, DIM> cov() const = 0;
  virtual DistributionInterface* Clone() const = 0;
};

template <int DIM, typename Derived>
class DistributionBase : public DistributionInterface<DIM> {
  virtual DistributionBase* Clone() const {
      return new Derived(static_cast<Derived const&>(*this));
    }
};

}  // namespace refill

#endif  // REFILL_DISTRIBUTIONS_DISTRIBUTION_BASE_H_
