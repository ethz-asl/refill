#ifndef REFILL_SYSTEM_MODELS_SYSTEM_MODEL_BASE_H_
#define REFILL_SYSTEM_MODELS_SYSTEM_MODEL_BASE_H_

#include <Eigen/Dense>

#include "refill/distributions/distribution_base.h"

namespace refill {

/**
 * @brief Interface for system models.
 *
 * All system models have to have this class as an ancestor.
 */
class SystemModelBase {
 public:
  /**
   * @brief Propagates a state vector through the system model.
   *
   * @param state The state vector to be propagated.
   * @param input The input vector to the system.
   * @return the propagated state vector.
   */
  virtual Eigen::VectorXd propagate(
      const Eigen::VectorXd& state,
      const Eigen::VectorXd& input) const = 0;

  /**
   * @brief Returns the systems state dimension.
   *
   * @return the state dimension.
   */
  virtual std::size_t getStateDim() const = 0;

  /**
   * @brief Returns the systems input dimension.
   *
   * @return the input dimension.
   */
  virtual std::size_t getInputDim() const = 0;

  /**
   * @brief Returns the systems noise dimension.
   *
   * @return the noise dimension.
   */
  virtual std::size_t getSystemNoiseDim() const = 0;

  /**
   * @brief Returns the system noise.
   *
   * @return a pointer to the system noise distribution.
   */
  virtual DistributionInterface* getSystemNoise() const = 0;
};

}  // namespace refill

#endif  // REFILL_SYSTEM_MODELS_SYSTEM_MODEL_BASE_H_
