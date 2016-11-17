#ifndef REFILL_FILTERS_EXTENDED_KALMAN_FILTER_H_
#define REFILL_FILTERS_EXTENDED_KALMAN_FILTER_H_

#include <memory>

#include "refill/measurement_models/linearized_measurement_model.h"
#include "refill/system_models/linearized_system_model.h"
#include "refill/distributions/gaussian_distribution.h"
#include "refill/filters/filter_base.h"

namespace refill {

class ExtendedKalmanFilter : public FilterBase<LinearizedSystemModel,
    LinearizedMeasurementModel> {
 public:
  ExtendedKalmanFilter();
  explicit ExtendedKalmanFilter(const GaussianDistribution& initial_state);

  void setState(const GaussianDistribution& state);

  void predict(const LinearizedSystemModel& system_model);
  void predict(const LinearizedSystemModel& system_model,
               const Eigen::VectorXd& input);

  void update(const LinearizedMeasurementModel& measurement_model,
              const Eigen::VectorXd& measurement);

  GaussianDistribution state() const {
    return state_;
  }

 private:
  GaussianDistribution state_;
};

}  // namespace refill

#endif  // REFILL_FILTERS_EXTENDED_KALMAN_FILTER_H_
