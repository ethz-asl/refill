#ifndef REFILL_FILTERS_EXTENDED_KALMAN_FILTER_H_
#define REFILL_FILTERS_EXTENDED_KALMAN_FILTER_H_

#include <memory>

#include "refill/distributions/gaussian_distribution.h"
#include "refill/system_models/linear_system_model.h"
#include "refill/measurement_models/linear_measurement_model.h"
#include "refill/filters/filter_base.h"

namespace refill {

class ExtendedKalmanFilter : public FilterBase {
 public:
  ExtendedKalmanFilter();
  explicit ExtendedKalmanFilter(const GaussianDistribution& initial_state);

  void setState(const GaussianDistribution& state);

  void predict(const SystemModelBase& system_model);
  void predict(const SystemModelBase& system_model,
               const Eigen::VectorXd& input);

  void update(const MeasurementModelBase& measurement_model,
              const Eigen::VectorXd& measurement);

  GaussianDistribution state() const {
    return state_;
  }

 private:
  GaussianDistribution state_;
};

}  // namespace refill

#endif  // REFILL_FILTERS_EXTENDED_KALMAN_FILTER_H_
