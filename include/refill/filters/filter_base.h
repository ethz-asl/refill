#ifndef REFILL_FILTERS_FILTER_BASE_H_
#define REFILL_FILTERS_FILTER_BASE_H_

#include <Eigen/Dense>

#include "refill/system_models/system_model_base.h"
#include "refill/measurement_models/measurement_model_base.h"

namespace refill {

template<int STATEDIM = Eigen::Dynamic, int MEASDIM = Eigen::Dynamic,
    int INPUTDIM = 0>
class FilterBase {
 public:
  virtual void Predict(const SystemModelBase<STATEDIM, INPUTDIM>& system_model,
                       const Eigen::Matrix<double, INPUTDIM, 1>& input) = 0;
  virtual void Update(
      const MeasurementModelBase<STATEDIM, MEASDIM>& measurement_model,
      const Eigen::Matrix<double, MEASDIM, 1>& measurement) = 0;
};

}  // namespace refill

#endif  // REFILL_FILTERS_FILTER_BASE_H_
