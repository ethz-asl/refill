#ifndef REFILL_FILTERS_EXTENDED_KALMAN_FILTER_H_
#define REFILL_FILTERS_EXTENDED_KALMAN_FILTER_H_

#include <glog/logging.h>
#include <Eigen/Dense>
#include <Eigen/LU>

#include <memory>

#include "refill/distributions/gaussian_distribution.h"
#include "refill/filters/filter_base.h"
#include "refill/measurement_models/linear_measurement_model.h"
#include "refill/measurement_models/linearized_measurement_model.h"
#include "refill/measurement_models/measurement_model_base.h"
#include "refill/system_models/linear_system_model.h"
#include "refill/system_models/linearized_system_model.h"
#include "refill/system_models/system_model_base.h"

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
  /** @brief Initializes the Kalman filter to no value. */
  ExtendedKalmanFilter();

  /**
   * @brief Initializes the Kalman filter in a way that expects system models
   *        to be given upon prediction / update. */
  explicit ExtendedKalmanFilter(const GaussianDistribution& initial_state);

  /** @brief Initialize the Kalman filter with a default system model */
  explicit ExtendedKalmanFilter(
      std::unique_ptr<LinearizedSystemModel> system_model);

  /**
   * @brief Initialized the Kalman filter to use the standard models, if
   *        not stated otherwise. */
  explicit ExtendedKalmanFilter(
      const GaussianDistribution& initial_state,
      std::unique_ptr<LinearizedSystemModel> system_model,
      std::unique_ptr<LinearizedMeasurementModel> measurement_model);

  /** @brief Sets the state of the Kalman filter. */
  void setState(const GaussianDistribution& state) override;

  using FilterBase::predict;
  /**
   * @brief Performs a prediction step using the provided system model and
   *        and input. */
  void predict(const double dt, SystemModelBase& system_model,
              const Eigen::VectorXd& input) override;

  using FilterBase::update;
  /** @brief Performs an update using the provided measurement model. */
  void update(const MeasurementModelBase& measurement_model,
              const Eigen::VectorXd& measurement, double* likelihood) override;

  /**
   * @brief Function to get the current filter state.
   *
   * @return the current filter state.
   */
  GaussianDistribution state() const override { return state_; }

 private:
  GaussianDistribution state_;
};

}  // namespace refill

#endif  // REFILL_FILTERS_EXTENDED_KALMAN_FILTER_H_
