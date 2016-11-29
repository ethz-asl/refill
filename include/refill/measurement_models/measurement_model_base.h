#ifndef REFILL_MEASUREMENT_MODELS_MEASUREMENT_MODEL_BASE_H_
#define REFILL_MEASUREMENT_MODELS_MEASUREMENT_MODEL_BASE_H_

#include <Eigen/Dense>
#include <memory>

#include "refill/distributions/distribution_base.h"

namespace refill {

/**
 * @brief Interface for measurement models.
 *
 * All measurement models must have this class as an ancestor.
 */
class MeasurementModelBase {
 public:
  /**
   * @brief Use the measurement model to receive the expected measurement.
   *
   * @param state The state vector used for the observation.
   * @return the expected measurement given the current state.
   */
  virtual Eigen::VectorXd observe(const Eigen::VectorXd& state) const = 0;

  /** @brief Returns the measurement models state dimension. */
  std::size_t getStateDim() const;
  /** @brief Returns the measurement models measurement dimension. */
  std::size_t getMeasurementDim() const;
  /** @brief Returns the measurement models noise dimension. */
  std::size_t getMeasurementNoiseDim() const;
  /** @brief Returns the measurement models noise. */
  DistributionInterface* getMeasurementNoise() const;

 protected:
  /** @brief Default constructor should not be used. */
  MeasurementModelBase() = delete;
  /** @brief Constructor for the measurement model base class. */
  MeasurementModelBase(const std::size_t& state_dim,
                       const std::size_t& measurement_dim,
                       const DistributionInterface& measurement_noise);

  /** @brief Function to set the measurement model base parameters. */
  void setMeasurementModelBaseParameters(
      const std::size_t& state_dim, const std::size_t& measurement_dim,
      const DistributionInterface& measurement_noise);

 private:
  std::unique_ptr<DistributionInterface> measurement_noise_;
  std::size_t state_dim_;
  std::size_t measurement_dim_;
};

}  // namespace refill

#endif  // REFILL_MEASUREMENT_MODELS_MEASUREMENT_MODEL_BASE_H_
