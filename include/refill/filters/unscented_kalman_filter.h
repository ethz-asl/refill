#ifndef REFILL_FILTERS_UNSCENTED_KALMAN_FILTER_H
#define REFILL_FILTERS_UNSCENTED_KALMAN_FILTER_H

#include <glog/logging.h>
#include <Eigen/Dense>
#include <Eigen/LU>

#include <memory>

#include <iostream>

#include "refill/distributions/gaussian_distribution.h"
#include "refill/filters/filter_base.h"
#include "refill/measurement_models/measurement_model_base.h"
#include "refill/system_models/system_model_base.h"

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
  /** @brief Initializes the Unscented Kalman filter to no value. */
  UnscentedKalmanFilter(double alpha);

  /**
   * @brief Initializes the Unscented Kalman filter in a way that expects
   *        system models to be given upon prediction / update. */
  explicit UnscentedKalmanFilter(const double alpha,
                                 const GaussianDistribution& initial_state);

  /** @brief Initialize the Unscented Kalman filter with a
  *          default system model */
  explicit UnscentedKalmanFilter(
      const double alpha, std::unique_ptr<SystemModelBase> system_model);

  /**
   * @brief Initialized the Unscented Kalman filter to use the standard models,
   *        if not stated otherwise. */
  explicit UnscentedKalmanFilter(
      const double alpha, const GaussianDistribution& initial_state,
      std::unique_ptr<SystemModelBase> system_model,
      std::unique_ptr<MeasurementModelBase> measurement_model);

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

  /** @brief Sets the state of the Kalman filter. */
  void setState(const GaussianDistribution& state) override;

  /** @brief Generates a Matrix of sigma points that are sampled around the
   * state mean. */
  void generateSigmaPoints(const double alpha,
                           const GaussianDistribution& state,
                           Eigen::MatrixXd* Sx, std::vector<double>& S_weights);

  /** @brief Function to get the current filter state. */
  GaussianDistribution state() const override { return state_; }

 private:
  GaussianDistribution state_;

  const double alpha_;
};

}  // namespace refill

#endif /*REFILL_FILTERS_UNSCENTED_KALMAN_FILTER_H*/