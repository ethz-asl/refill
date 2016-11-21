#ifndef REFILL_FILTERS_EXTENDED_KALMAN_FILTER_H_
#define REFILL_FILTERS_EXTENDED_KALMAN_FILTER_H_

#include <memory>

#include "refill/distributions/gaussian_distribution.h"
#include "refill/filters/filter_base.h"
#include "refill/measurement_models/linear_measurement_model.h"
#include "refill/measurement_models/linearized_measurement_model.h"
#include "refill/system_models/linear_system_model.h"
#include "refill/system_models/linearized_system_model.h"

namespace refill {

class ExtendedKalmanFilter : public FilterBase {
 public:
  // Default constructor creates a 1d Kalman Filter with identity system and
  // measurement models.
  ExtendedKalmanFilter();
  // Initializes the Kalman filter in a way that expects system models to be
  // given upon prediction / update.
  explicit ExtendedKalmanFilter(const GaussianDistribution& initial_state);
  // The ExtendedKalmanFilter class takes ownership of both models.
  explicit ExtendedKalmanFilter(
      const GaussianDistribution& initial_state,
      std::unique_ptr<LinearizedSystemModel> system_model,
      std::unique_ptr<LinearizedMeasurementModel> measurement_model);

  void setState(const GaussianDistribution& state);

  void predict();
  void predict(const Eigen::VectorXd& input);
  void predict(const LinearizedSystemModel& system_model);
  void predict(const LinearizedSystemModel& system_model,
               const Eigen::VectorXd& input);

  void update(const Eigen::VectorXd& measurement);
  void update(const LinearizedMeasurementModel& measurement_model,
              const Eigen::VectorXd& measurement);

  GaussianDistribution state() const { return state_; }

 private:
  GaussianDistribution state_;
  std::unique_ptr<LinearizedSystemModel> system_model_;
  std::unique_ptr<LinearizedMeasurementModel> measurement_model_;
};

}  // namespace refill

#endif  // REFILL_FILTERS_EXTENDED_KALMAN_FILTER_H_
