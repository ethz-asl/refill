#ifndef REFILL_DISTRIBUTIONS_DISTRIBUTION_BASE_H_
#define REFILL_DISTRIBUTIONS_DISTRIBUTION_BASE_H_

#include <Eigen/Dense>

namespace refill {

// Interface class for distributions
class DistributionInterface {
 public:
  virtual Eigen::VectorXd mean() const = 0;
  virtual Eigen::MatrixXd cov() const = 0;
  virtual DistributionInterface* clone() const = 0;
};

//  Class that implements the Curiously Recurring Templating Pattern
//  so the clone function doesn't have to be implemented in every
//  derived distribution.
//  For new distributions, inherit from this class like this:
//  class NewDistribution : public DistributionBase<DIM, NewDistribution>

template<typename DERIVED>
class DistributionBase : public DistributionInterface {
  virtual DistributionInterface* clone() const {
    return new DERIVED(dynamic_cast<DERIVED const&>(*this));
  }
};

}  // namespace refill

#endif  // REFILL_DISTRIBUTIONS_DISTRIBUTION_BASE_H_