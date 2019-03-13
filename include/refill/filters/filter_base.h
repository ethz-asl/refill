#ifndef REFILL_FILTERS_FILTER_BASE_H_
#define REFILL_FILTERS_FILTER_BASE_H_

#include <Eigen/Dense>

#include "refill/measurement_models/measurement_model_base.h"
#include "refill/system_models/system_model_base.h"
#include "refill/system_models/linearized_system_model.h"
#include "refill/distributions/gaussian_distribution.h"
#include "refill/measurement_models/linearized_measurement_model.h"

namespace refill {

/**
 * @brief Interface for all filters
 *
 * This class is an interface for all filters in refill.
 *
 * It must be an ancestor of all filters implemented in refill.
 */
class FilterBase {
 public:
  /** @brief Perform a prediction step. */
  virtual void predict() = 0;
  /** @brief Perform an update with a measurement. */
  /** @brief Performs a prediction step using the provided system model. */
  virtual void predict(const LinearizedSystemModel& system_model) = 0;

  /** @brief Performs an update using the standard measurement model. */
  virtual void update(const Eigen::VectorXd& measurement) = 0;

  /** @brief Performs an update using the provided measurement model. */
  virtual void update(const LinearizedMeasurementModel& measurement_model,
                      const Eigen::VectorXd& measurement) = 0;

  /** @brief Sets the state of the Kalman filter. */
  virtual void setState(const GaussianDistribution& state) = 0;

  /** @brief return state mean. */
  virtual GaussianDistribution state() const = 0;
};

}  // namespace refill

#endif  // REFILL_FILTERS_FILTER_BASE_H_
