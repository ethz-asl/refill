#ifndef REFILL_FILTERS_KALMAN_FILTER_H_
#define REFILL_FILTERS_KALMAN_FILTER_H_

#include <memory>

#include "refill/distributions/gaussian_distribution.h"
#include "refill/system_models/linear_system_model.h"
#include "refill/measurement_models/linear_measurement_model.h"
#include "refill/filters/filter_base.h"

namespace refill {

template<int STATEDIM = Eigen::Dynamic, int MEASDIM = Eigen::Dynamic,
    int INPUTDIM = 0>
class KalmanFilter : public FilterBase<STATEDIM, MEASDIM> {
 public:
  KalmanFilter();
  explicit KalmanFilter(const GaussianDistribution<STATEDIM>& initial_state);

  // TODO(jwidauer): Figure out what to do if this is called (shouldn't happen)
  void Predict(
      const SystemModelBase<STATEDIM, INPUTDIM>& system_model,
      const Eigen::Matrix<double, INPUTDIM, 1>& input = Eigen::Matrix<double,
          INPUTDIM, 1>::Zero(INPUTDIM, 1)) {
  }
  void Predict(
      const LinearSystemModel<STATEDIM, INPUTDIM>& system_model,
      const Eigen::Matrix<double, INPUTDIM, 1>& input = Eigen::Matrix<double,
          INPUTDIM, 1>::Zero(INPUTDIM, 1));

  // TODO(jwidauer): Figure out what to do when this is called.
  void Update(const MeasurementModelBase<STATEDIM, MEASDIM>& measurement_model,
              const Eigen::Matrix<double, MEASDIM, 1>& measurement) {
  }
  void Update(
      const LinearMeasurementModel<STATEDIM, MEASDIM>& measurement_model,
      const Eigen::Matrix<double, MEASDIM, 1>& measurement);

  GaussianDistribution<STATEDIM> state() const {
    return state_;
  }

 private:
  GaussianDistribution<STATEDIM> state_;
};

// Alias for a dynamic size version of the Kalman Filter. Notation
// comptatible to Eigen.
using KalmanFilterXd = KalmanFilter<Eigen::Dynamic, Eigen::Dynamic>;

}  // namespace refill

#include "./kalman_filter-inl.h"

#endif  // REFILL_FILTERS_KALMAN_FILTER_H_
