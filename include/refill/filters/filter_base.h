#ifndef REFILL_FILTERS_FILTER_BASE_H_
#define REFILL_FILTERS_FILTER_BASE_H_

#include <Eigen/Dense>

#include "refill/measurement_models/measurement_model_base.h"
#include "refill/system_models/system_model_base.h"

namespace refill {

template<typename SYSTEM_MODEL_TYPE, typename MEASUREMENT_MODEL_TYPE>
class FilterBase {
 public:
  virtual void predict(const SYSTEM_MODEL_TYPE& system_model,
                       const Eigen::VectorXd& input) = 0;
  virtual void update(const MEASUREMENT_MODEL_TYPE& measurement_model,
                      const Eigen::VectorXd& measurement) = 0;
};

}  // namespace refill

#endif  // REFILL_FILTERS_FILTER_BASE_H_
