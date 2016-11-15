#ifndef REFILL_FILTERS_FILTER_BASE_H_
#define REFILL_FILTERS_FILTER_BASE_H_

#include <Eigen/Dense>

#include "refill/system_models/system_model_base.h"
#include "refill/measurement_models/measurement_model_base.h"

namespace refill {

template<int STATE_DIM, class FILTER>
class FilterBase {
 private:
  template<int INPUT_DIM>
  void PredictBase(const SystemModelBase<STATE_DIM, INPUT_DIM>& system_model,
                   const Eigen::Matrix<double, INPUT_DIM, 1>& input) {
    static_cast<FILTER*>(this)->Predict(system_model, input);
  }

  template<int MEAS_DIM>
  void UpdateBase(
      const MeasurementModelBase<STATE_DIM, MEAS_DIM>& measurement_model,
      const Eigen::Matrix<double, MEAS_DIM, 1>& measurement) {
    static_cast<FILTER*>(this)->Update(measurement_model, measurement);
  }
};

}  // namespace refill

#endif  // REFILL_FILTERS_FILTER_BASE_H_
