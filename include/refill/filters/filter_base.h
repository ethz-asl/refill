#ifndef REFILL_FILTERS_FILTER_BASE_H_
#define REFILL_FILTERS_FILTER_BASE_H_

#include <Eigen/Dense>

#include "refill/system_models/system_model_base.h"
#include "refill/measurement_models/measurement_model_base.h"

namespace refill {

template<int STATEDIM, class FILTER>
class FilterBase {
 private:
  template<int INPUTDIM>
  void PredictBase(const SystemModelBase<STATEDIM, INPUTDIM>& system_model,
                   const Eigen::Matrix<double, INPUTDIM, 1>& input) {
    static_cast<FILTER*>(this)->Predict(system_model, input);
  }

  template<int MEASDIM>
  void UpdateBase(
      const MeasurementModelBase<STATEDIM, MEASDIM>& measurement_model,
      const Eigen::Matrix<double, MEASDIM, 1>& measurement) {
    static_cast<FILTER*>(this)->Update(measurement_model, measurement);
  }
};

}  // namespace refill

#endif  // REFILL_FILTERS_FILTER_BASE_H_
