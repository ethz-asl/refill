#ifndef REFILL_SYSTEM_MODELS_LINEAR_SYSTEM_MODEL_H_
#define REFILL_SYSTEM_MODELS_LINEAR_SYSTEM_MODEL_H_

#include <Eigen/Dense>
#include <glog/logging.h>

#include "refill/distributions/gaussian_distribution.h"
#include "refill/system_models/linearized_system_model.h"

namespace refill {

/**
 * @brief Class that implements a linear system model.
 *
 * This class is an implementation of linear system models of the form:
 *
 * @f$ x_k = A_k \cdot x_{k-1} + B_k \cdot u_k + L_k \cdot v_k @f$
 *
 * Where @f$ x_k @f$ is the system state, @f$ u_k @f$ the system input,
 * @f$ v_k @f$ the system noise, @f$ A_k @f$ the system mapping, @f$ B_k @f$
 * the input mapping and @f$ L_k @f$ the noise mapping at time step @f$ k @f$.
 *
 * Use this class together with the LinearMeasurementModel, noise of type
 * GaussianDistribution and the ExtendedKalmanFilter if you want to implement
 * a simple kalman filter.
 */
class LinearSystemModel : public LinearizedSystemModel {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /** @brief Constructs an empty linear system model. */
  LinearSystemModel();
  /** @brief Constructor for a system model without an input and a
   *         simple noise mapping.
   */
  LinearSystemModel(const Eigen::MatrixXd& system_mapping,
                    const DistributionInterface& system_noise);
  /** @brief Constructor for a system model with an input and a
   *         simple noise mapping.
   */
  LinearSystemModel(const Eigen::MatrixXd& system_mapping,
                    const DistributionInterface& system_noise,
                    const Eigen::MatrixXd& input_mapping);
  /** @brief Constructor for a system model with an input and a
   *         specific noise mapping.
   */
  LinearSystemModel(const Eigen::MatrixXd& system_mapping,
                    const DistributionInterface& system_noise,
                    const Eigen::MatrixXd& input_mapping,
                    const Eigen::MatrixXd& noise_mapping);

  /** @brief Sets the system model parameters for a system without an input and
   *         a simple noise mapping.
   */
  void setSystemParameters(const Eigen::MatrixXd& system_mapping,
                           const DistributionInterface& system_noise);
  /** @brief Sets the system model parameters for a system with an input and
   *         a simple noise mapping.
   */
  void setSystemParameters(const Eigen::MatrixXd& system_mapping,
                           const DistributionInterface& system_noise,
                           const Eigen::MatrixXd& input_mapping);
  /** @brief Sets the system model parameters for a system with an input and
   *         a specific noise mapping.
   */
  void setSystemParameters(const Eigen::MatrixXd& system_mapping,
                           const DistributionInterface& system_noise,
                           const Eigen::MatrixXd& input_mapping,
                           const Eigen::MatrixXd& noise_mapping);

  /** @brief Propagates the state vector through the system model. */
  Eigen::VectorXd propagate(const Eigen::VectorXd& state) const;
  /** @brief Propagates the state and input vector through the system model. */
  Eigen::VectorXd propagate(const Eigen::VectorXd& state,
                            const Eigen::VectorXd& input) const;

  /**
   * @brief Function to get @f$ A_k @f$, which is the system Jacobian w.r.t.
   *        the system state.
   */
  Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd& state,
                                   const Eigen::VectorXd& input) const;
  /**
   * @brief Function to get @f$ L_k @f$, which is the system Jacobian w.r.t.
   *        the system noise.
   */
  Eigen::MatrixXd getNoiseJacobian(const Eigen::VectorXd& state,
                                   const Eigen::VectorXd& input) const;

 private:
  Eigen::MatrixXd system_mapping_;
  Eigen::MatrixXd input_mapping_;
  Eigen::MatrixXd noise_mapping_;
};

}  // namespace refill

#endif  // REFILL_SYSTEM_MODELS_LINEAR_SYSTEM_MODEL_H_
