#ifndef REFILL_SYSTEM_MODELS_SYSTEM_MODEL_BASE_H_
#define REFILL_SYSTEM_MODELS_SYSTEM_MODEL_BASE_H_

#include <glog/logging.h>

#include <Eigen/Dense>
#include <cstdlib>
#include <memory>

#include "refill/distributions/distribution_base.h"

using std::size_t;

namespace refill {

/**
 * @brief Interface for system models.
 *
 * All system models must have this class as an ancestor.
 */
class SystemModelBase {
 public:
  virtual ~SystemModelBase() = default;

  virtual void setTimeStamp(const double dt, const double stamp) = 0;

  /**
   * @brief Propagates a state and input vector through the system model.
   *
   * @param state The state vector to be propagated.
   * @param input The input vector to the system.
   * @param noise A system noise sample used for propagation.
   * @return the new state vector.
   */
  virtual Eigen::VectorXd propagate(const Eigen::VectorXd& state,
                                    const Eigen::VectorXd& input,
                                    const Eigen::VectorXd& noise) = 0;

  /** @brief A vectorized version of the propagation. */
  virtual Eigen::MatrixXd propagateVectorized(
      const Eigen::MatrixXd& sampled_state, const Eigen::VectorXd& input,
      const Eigen::MatrixXd& sampled_noise);

  /** @brief Returns the systems state dimension. */
  size_t getStateDim() const;
  /** @brief Returns the systems input dimension. */
  size_t getInputDim() const;
  /** @brief Returns the systems noise dimension. */
  size_t getNoiseDim() const;
  /** @brief Returns a pointer to the system noise. */
  DistributionInterface* getNoise() const;

  virtual Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd& state,
                                           const Eigen::VectorXd& input) {
    return Eigen::Matrix3d::Zero();
  }
  virtual Eigen::MatrixXd getNoiseJacobian(const Eigen::VectorXd& state,
                                           const Eigen::VectorXd& input) {
    return Eigen::Matrix3d::Zero();
  }

 protected:
  /** Default constructor should not be used. */
  SystemModelBase() = delete;
  /** @brief Constructor for a system model without an input. */
  SystemModelBase(const size_t& state_dim,
                  const DistributionInterface& system_noise);
  /** @brief Constructor for a system model with input. */
  SystemModelBase(const size_t& state_dim,
                  const DistributionInterface& system_noise,
                  const size_t& input_dim);

  /** @brief Function to set the system model parameters without an input. */
  void setSystemModelBaseParameters(const size_t& state_dim,
                                    const DistributionInterface& system_noise);
  /** @brief Function to set the system model parameters with an input. */
  void setSystemModelBaseParameters(const size_t& state_dim,
                                    const DistributionInterface& system_noise,
                                    const size_t& input_dim);

 private:
  size_t state_dim_;
  size_t input_dim_;
  std::unique_ptr<DistributionInterface> system_noise_;
};

}  // namespace refill

#endif  // REFILL_SYSTEM_MODELS_SYSTEM_MODEL_BASE_H_
