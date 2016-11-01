#ifndef INCLUDE_REFILL_FILTER_BASE_H_
#define INCLUDE_REFILL_FILTER_BASE_H_

#include <Eigen/Dense>

namespace refill {

class FilterBase {
 public:
  virtual void Predict() = 0;
  virtual void Update(Eigen::VectorXd measurement) = 0;
};

}  // namespace refill

#endif  // INCLUDE_REFILL_FILTER_BASE_H_
