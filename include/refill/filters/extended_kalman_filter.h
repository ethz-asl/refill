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

/**
 * @brief Implementation of an extended Kalman filter.
 *
 * This class is an implementation of an extended Kalman filter.
 *
 * It can also be used as a standard Kalman filter, if used with a
 * LinearSystemModel, LinearMeasurementModel and GaussianDistribution noise.
 */
class ExtendedKalmanFilter : public FilterBase {
 public:
  /**
   * @brief Initializes the Kalman filter in a way that expects system models
   *        to be given upon prediction / update.
   */
  explicit ExtendedKalmanFilter(const GaussianDistribution& initial_state);

  /**
   * @brief Initialized the Kalman filter to use the standard models, if
   *        not stated otherwise.
   */
  explicit ExtendedKalmanFilter(
      const GaussianDistribution& initial_state,
      std::unique_ptr<LinearizedSystemModel> system_model,
      std::unique_ptr<LinearizedMeasurementModel> measurement_model);

  /** @brief Sets the state of the Kalman filter. */
  void setState(const GaussianDistribution& state);

  /**
   * @brief Performs a prediction step with the standard system model and
   *        no input.
   */
  void predict() override;
  /**
   * @brief Performs a prediction step with the standard system model and
   *        an input.
   */
  void predict(const Eigen::VectorXd& input);
  /**
   * @brief Performs a prediction step using the provided system model and
   *        no input.
   */
  void predict(const LinearizedSystemModel& system_model);
  /**
   * @brief Performs a prediction step using the provided system model and
   *        and input.
   */
  void predict(const LinearizedSystemModel& system_model,
               const Eigen::VectorXd& input);

  /** @brief Performs an update using the standard measurement model. */
  void update(const Eigen::VectorXd& measurement) override;
  /** @brief Performs an update using the provided measurement model. */
  void update(const LinearizedMeasurementModel& measurement_model,
              const Eigen::VectorXd& measurement);

  /**
   * @brief Function to get the current filter state.
   *
   * @return the current filter state.
   */
  GaussianDistribution state() const { return state_; }

 private:
  GaussianDistribution state_;
  std::unique_ptr<LinearizedSystemModel> system_model_;
  std::unique_ptr<LinearizedMeasurementModel> measurement_model_;
};

}  // namespace refill

#endif  // REFILL_FILTERS_EXTENDED_KALMAN_FILTER_H_
