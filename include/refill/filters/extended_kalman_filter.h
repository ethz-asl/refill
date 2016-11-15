#ifndef REFILL_FILTERS_EXTENDED_KALMAN_FILTER_H_
#define REFILL_FILTERS_EXTENDED_KALMAN_FILTER_H_

#include <memory>

#include "refill/distributions/gaussian_distribution.h"
#include "refill/system_models/linear_system_model.h"
#include "refill/measurement_models/linear_measurement_model.h"
#include "refill/filters/filter_base.h"

namespace refill {

template<int STATE_DIM = Eigen::Dynamic>
class ExtendedKalmanFilter : public FilterBase<STATE_DIM,
    ExtendedKalmanFilter<STATE_DIM>> {
 public:
  ExtendedKalmanFilter();
  explicit ExtendedKalmanFilter(
      const GaussianDistribution<STATE_DIM>& initial_state);

  template<int INPUT_DIM = Eigen::Dynamic>
  void Predict(
      const SystemModelBase<STATE_DIM, INPUT_DIM>& system_model,
      const Eigen::Matrix<double, INPUT_DIM, 1>& input = Eigen::Matrix<double,
          INPUT_DIM, 1>::Zero(INPUT_DIM, 1));

  template<int MEAS_DIM = Eigen::Dynamic>
  void Update(
      const MeasurementModelBase<STATE_DIM, MEAS_DIM>& measurement_model,
      const Eigen::Matrix<double, MEAS_DIM, 1>& measurement);

  GaussianDistribution<STATE_DIM> state() const {
    return state_;
  }

 private:
  GaussianDistribution<STATE_DIM> state_;
};

// Alias for a dynamic size version of the Kalman Filter. Notation
// comptatible to Eigen.
using ExtendedKalmanFilterXd = ExtendedKalmanFilter<>;

}  // namespace refill

#include "./extended_kalman_filter-inl.h"

#endif  // REFILL_FILTERS_EXTENDED_KALMAN_FILTER_H_
