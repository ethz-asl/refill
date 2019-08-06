#ifndef REFILL_FILTERS_FILTER_BASE_H_
#define REFILL_FILTERS_FILTER_BASE_H_

#include <glog/logging.h>
#include <Eigen/Dense>

#include "refill/distributions/gaussian_distribution.h"
#include "refill/measurement_models/linearized_measurement_model.h"
#include "refill/measurement_models/measurement_model_base.h"
#include "refill/system_models/linearized_system_model.h"
#include "refill/system_models/system_model_base.h"

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
  FilterBase();

  FilterBase(std::unique_ptr<SystemModelBase> system_model,
             std::unique_ptr<MeasurementModelBase> measurement_model);

  /** @brief Perform prediction step. Use default model, zero input and dt=0 */
  void predict();

  /** @brief Perform prediction step. Use default model, zero input */
  void predict(const double dt);

  /** @brief Perform prediction step. Use provided model, zero input */
  void predict(const double dt, SystemModelBase& system_model);

  /** @brief Perform prediction step. Use provided model and input */
  virtual void predict(const double dt, SystemModelBase& system_model,
                       const Eigen::VectorXd& input) = 0;

  void update(const Eigen::VectorXd& measurement);

  void update(const MeasurementModelBase& measurement_model,
              const Eigen::VectorXd& measurement);

  /** @brief Performs an update using the provided measurement model. */
  virtual void update(const MeasurementModelBase& measurement_model,
                      const Eigen::VectorXd& measurement,
                      double* likelihood) = 0;

  /** @brief Sets the state of the Kalman filter. */
  virtual void setState(const GaussianDistribution& state) = 0;

  /** @brief return state mean. */
  virtual GaussianDistribution state() const = 0;

  void setStateStamp(const double stamp) { state_stamp_ = stamp; }

 protected:
  std::unique_ptr<SystemModelBase> system_model_;
  std::unique_ptr<MeasurementModelBase> measurement_model_;
  double state_stamp_;
};

}  // namespace refill

#endif  // REFILL_FILTERS_FILTER_BASE_H_
