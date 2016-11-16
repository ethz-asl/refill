#ifndef REFILL_FILTERS_FILTER_BASE_H_
#define REFILL_FILTERS_FILTER_BASE_H_

#include <Eigen/Dense>

#include "refill/system_models/system_model_base.h"
#include "refill/measurement_models/measurement_model_base.h"

namespace refill {

class FilterBase {
 public:
//  virtual ~FilterBase();
  virtual void predict(const SystemModelBase& system_model,
                       const Eigen::VectorXd& input) = 0;
  virtual void update(const MeasurementModelBase& measurement_model,
                      const Eigen::VectorXd& measurement) = 0;
};

}  // namespace refill

#endif  // REFILL_FILTERS_FILTER_BASE_H_
