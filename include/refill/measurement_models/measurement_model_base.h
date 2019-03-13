#ifndef REFILL_MEASUREMENT_MODELS_MEASUREMENT_MODEL_BASE_H_
#define REFILL_MEASUREMENT_MODELS_MEASUREMENT_MODEL_BASE_H_

#include <Eigen/Dense>
#include <glog/logging.h>

#include <cstdlib>
#include <memory>

#include "refill/distributions/distribution_base.h"

using std::size_t;

namespace refill {

/**
 * @brief Interface for measurement models.
 *
 * All measurement models must have this class as an ancestor.
 */
class MeasurementModelBase {
 public:
  virtual ~MeasurementModelBase() = default;

  /**
   * @brief Use the measurement model to receive the expected measurement.
   *
   * @param state The state vector used for the observation.
   * @param noise A measurement noise sample used for observation.
   * @return the expected measurement given the current state.
   */
  virtual Eigen::VectorXd observe(const Eigen::VectorXd& state,
                                  const Eigen::VectorXd& noise) const = 0;

  /**
   * @brief Use the measurement model to predict state based on measurement.
   *
   * @param measurement vector
   * @return the expected state
   */
  virtual Eigen::VectorXd measure(const Eigen::VectorXd& measurement) const = 0;

  /** @brief Returns the measurement models state dimension. */
  size_t getStateDim() const;
  /** @brief Returns the measurement models measurement dimension. */
  size_t getMeasurementDim() const;
  /** @brief Returns the measurement models noise dimension. */
  size_t getNoiseDim() const;
  /** @brief Returns the measurement models noise. */
  DistributionInterface* getNoise() const;

 protected:
  /** @brief Default constructor should not be used. */
  MeasurementModelBase() = delete;
  /** @brief Constructor for the measurement model base class. */
  MeasurementModelBase(const size_t& state_dim,
                       const size_t& measurement_dim,
                       const DistributionInterface& measurement_noise);

  /** @brief Function to set the measurement model base parameters. */
  void setMeasurementModelBaseParameters(
      const size_t& state_dim, const size_t& measurement_dim,
      const DistributionInterface& measurement_noise);

 private:
  size_t state_dim_;
  size_t measurement_dim_;
  std::unique_ptr<DistributionInterface> measurement_noise_;
};

}  // namespace refill

#endif  // REFILL_MEASUREMENT_MODELS_MEASUREMENT_MODEL_BASE_H_
