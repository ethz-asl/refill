#ifndef REFILL_FILTERS_EXTENDED_KALMAN_FILTER_H_
#define REFILL_FILTERS_EXTENDED_KALMAN_FILTER_H_

#include <memory>

#include "refill/distributions/gaussian_distribution.h"
#include "refill/system_models/linear_system_model.h"
#include "refill/measurement_models/linear_measurement_model.h"
#include "refill/filters/filter_base.h"

namespace refill {

template<int STATEDIM = Eigen::Dynamic>
class ExtendedKalmanFilter : public FilterBase<STATEDIM,
    ExtendedKalmanFilter<STATEDIM>> {
 public:
  ExtendedKalmanFilter();
  explicit ExtendedKalmanFilter(
      const GaussianDistribution<STATEDIM>& initial_state);

  template<int INPUTDIM = Eigen::Dynamic>
  void Predict(
      const SystemModelBase<STATEDIM, INPUTDIM>& system_model,
      const Eigen::Matrix<double, INPUTDIM, 1>& input = Eigen::Matrix<double,
          INPUTDIM, 1>::Zero(INPUTDIM, 1));

  template<int MEASDIM = Eigen::Dynamic>
  void Update(const MeasurementModelBase<STATEDIM, MEASDIM>& measurement_model,
              const Eigen::Matrix<double, MEASDIM, 1>& measurement);

  GaussianDistribution<STATEDIM> state() const {
    return state_;
  }

 private:
  GaussianDistribution<STATEDIM> state_;
};

// Alias for a dynamic size version of the Kalman Filter. Notation
// comptatible to Eigen.
using ExtendedKalmanFilterXd = ExtendedKalmanFilter<>;

}  // namespace refill

#include "./extended_kalman_filter-inl.h"

#endif  // REFILL_FILTERS_EXTENDED_KALMAN_FILTER_H_
