#ifndef REFILL_FILTERS_FILTER_BASE_H_
#define REFILL_FILTERS_FILTER_BASE_H_

#include <Eigen/Dense>

#include "refill/measurement_models/measurement_model_base.h"
#include "refill/system_models/system_model_base.h"

namespace refill {

class FilterBase {
 public:
  virtual void predict(const Eigen::VectorXd& input) = 0;
  virtual void update(const Eigen::VectorXd& measurement) = 0;
};

}  // namespace refill

#endif  // REFILL_FILTERS_FILTER_BASE_H_
