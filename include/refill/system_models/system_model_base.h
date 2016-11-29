#ifndef REFILL_SYSTEM_MODELS_SYSTEM_MODEL_BASE_H_
#define REFILL_SYSTEM_MODELS_SYSTEM_MODEL_BASE_H_

#include <glog/logging.h>
#include <Eigen/Dense>
#include <memory>

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
  virtual Eigen::VectorXd propagate(const Eigen::VectorXd& state,
                                    const Eigen::VectorXd& input) const = 0;

  /**
   * @brief Returns the systems state dimension.
   *
   * @return the state dimension.
   */
  std::size_t getStateDim() const;

  /**
   * @brief Returns the systems input dimension.
   *
   * @return the input dimension.
   */
  std::size_t getInputDim() const;

  /**
   * @brief Returns the systems noise dimension.
   *
   * @return the noise dimension.
   */
  std::size_t getSystemNoiseDim() const;

  /**
   * @brief Returns the system noise.
   *
   * @return a pointer to the system noise distribution.
   */
  DistributionInterface* getSystemNoise() const;

 protected:
  SystemModelBase() = delete;
  SystemModelBase(const std::size_t& state_dim,
                  const DistributionInterface& system_noise);
  SystemModelBase(const std::size_t& state_dim,
                  const DistributionInterface& system_noise,
                  const std::size_t& input_dim);

  void setSystemModelBaseParameters(const std::size_t& state_dim,
                                    const DistributionInterface& system_noise);
  void setSystemModelBaseParameters(const std::size_t& state_dim,
                                    const DistributionInterface& system_noise,
                                    const std::size_t& input_dim);

 private:
  std::unique_ptr<DistributionInterface> system_noise_;
  std::size_t state_dim_;
  std::size_t input_dim_;
};

}  // namespace refill

#endif  // REFILL_SYSTEM_MODELS_SYSTEM_MODEL_BASE_H_
