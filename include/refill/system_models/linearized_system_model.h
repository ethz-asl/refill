#ifndef REFILL_SYSTEM_MODELS_LINEARIZED_SYSTEM_MODEL_H_
#define REFILL_SYSTEM_MODELS_LINEARIZED_SYSTEM_MODEL_H_

#include <Eigen/Dense>

#include <cstdlib>

#include "refill/distributions/distribution_base.h"
#include "refill/system_models/system_model_base.h"

using std::size_t;

namespace refill {

/**
 * @brief Class that implements a linearized system model.
 *
 * This class is an interface for system models with Jacobians.
 *
 * Its intended purpose is to implement system models of the form:
 *
 * @f$ x_k = f_k(x_{k-1}, u_k, v_k) @f$
 *
 * With Jacobians:
 *
 * @f$ A_k = \frac{\partial f}{\partial x}(x_{k-1}, u_k, \mu_k) @f$
 *
 * and
 *
 * @f$ L_k = \frac{\partial f}{\partial v}(x_{k-1}, u_k, \mu_k) @f$
 *
 * Where @f$ x_k @f$ denotes the system state, @f$ u_k @f$ the system input,
 * @f$ v_k @f$ the system noise and @f$ \mu_k @f$ the noise mean at timestep @f$ k @f$.
 *
 * To implement a system model that works with the ExtendedKalmanFilter,
 * inherit from this class and implement your own propagate() function.
 *
 * If the Jacobians can be computed analytically, it is recommended to do so and overload the
 * getStateJacobian() and getNoiseJacobian() functions.
 */
class LinearizedSystemModel : public SystemModelBase {
 public:
    /**
   * @brief Function to get @f$ A_k @f$, which is the system Jacobian w.r.t.
   *        the system state.
   */
  virtual Eigen::MatrixXd getStateJacobian(
      const Eigen::VectorXd& state, const Eigen::VectorXd& input) const;
  /**
   * @brief Functoin to get @f$ L_k @f$, which is the system Jacobian w.r.t.
   *        the system noise.
   */
  virtual Eigen::MatrixXd getNoiseJacobian(
      const Eigen::VectorXd& state, const Eigen::VectorXd& input) const;

 protected:
  /** @brief Default constructor should not be used. */
  LinearizedSystemModel() = delete;
  /** @brief Constructor for a linearized system model without an input. */
  LinearizedSystemModel(const size_t& state_dim,
                        const DistributionInterface& system_noise);
  /** @brief Constructor for a linearized system model with an input. */
  LinearizedSystemModel(const size_t& state_dim,
                        const DistributionInterface& system_noise,
                        const size_t& input_dim);
};

}  // namespace refill

#endif  // REFILL_SYSTEM_MODELS_LINEARIZED_SYSTEM_MODEL_H_
