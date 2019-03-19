#ifndef REFILL_FILTERS_UNSCENTED_KALMAN_FILTER_H
#define REFILL_FILTERS_UNSCENTED_KALMAN_FILTER_H

#include <glog/logging.h>
#include <Eigen/Dense>
#include <Eigen/LU>

#include <memory>

#include <iostream>

#include "refill/distributions/gaussian_distribution.h"
#include "refill/filters/filter_base.h"
#include "refill/measurement_models/linear_measurement_model.h"
#include "refill/measurement_models/linearized_measurement_model.h"
#include "refill/system_models/linear_system_model.h"
#include "refill/system_models/linearized_system_model.h"

namespace refill {

/**
 * @brief Implementation of an unscented Kalman filter.
 *
 * This class is an implementation of an unscented Kalman filter.
 *
 * It can also be used as a standard Kalman filter, if used with a
 * LinearSystemModel, LinearMeasurementModel and GaussianDistribution noise.
 */
class UnscentedKalmanFilter : public FilterBase {
 public:
  /** @brief Initializes the Kalman filter to no value. */
  UnscentedKalmanFilter(double alpha);

  /**
   * @brief Initializes the Kalman filter in a way that expects system models
   *        to be given upon prediction / update. */
  explicit UnscentedKalmanFilter(const double alpha,
                                 const GaussianDistribution& initial_state);

  /**
   * @brief Initialized the Kalman filter to use the standard models, if
   *        not stated otherwise. */
  explicit UnscentedKalmanFilter(
      const double alpha, const GaussianDistribution& initial_state,
      std::unique_ptr<LinearizedSystemModel> system_model,
      std::unique_ptr<LinearizedMeasurementModel> measurement_model);

  /**
   * @brief Performs a prediction step with the standard system model and
   *        no input. */
  void predict() override;
  /**
   * @brief Performs a prediction step with the standard system model and
   *        an input. */
  void predict(const Eigen::VectorXd& input);
  /**
   * @brief Performs a prediction step using the provided system model and
   *        no input. */
  void predict(const LinearizedSystemModel& system_model) override;
  /**
   * @brief Performs a prediction step using the provided system model and
   *        and input. */
  void predict(const LinearizedSystemModel& system_model,
               const Eigen::VectorXd& input);

  /** @brief Performs an update using the standard measurement model. */
  void update(const Eigen::VectorXd& measurement) override;
  /** @brief Performs an update using the provided measurement model. */
  void update(const LinearizedMeasurementModel& measurement_model,
              const Eigen::VectorXd& measurement) override;

  /** @brief Sets the state of the Kalman filter. */
  void setState(const GaussianDistribution& state) override;

  /** @brief Function to get the current filter state. */
  GaussianDistribution state() const override { return state_; }

  /** @brief Generates a Matrix of sigma points that are sampled around the
   * state mean. */
  void generateSigmaPoints(const double alpha,
                           const GaussianDistribution& state,
                           Eigen::MatrixXd* Sx,
                           std::vector<double>& S_weights);

 private:
  GaussianDistribution state_;
  std::unique_ptr<LinearizedSystemModel> system_model_;
  std::unique_ptr<LinearizedMeasurementModel> measurement_model_;
  
  const double alpha_;
};

}  // namespace refill

#endif /*REFILL_FILTERS_UNSCENTED_KALMAN_FILTER_H*/