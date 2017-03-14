#ifndef REFILL_MEASUREMENT_MODELS_LINEARIZED_MEASUREMENT_MODEL_H_
#define REFILL_MEASUREMENT_MODELS_LINEARIZED_MEASUREMENT_MODEL_H_

#include <Eigen/Dense>

#include <cstdlib>

#include "refill/distributions/distribution_base.h"
#include "refill/measurement_models/measurement_model_base.h"

using std::size_t;

namespace refill {

/**
 * @brief Class that implements a linearized measurement model.
 *
 * This class is an interface for measurement models with Jacobians.
 *
 * Its intended purpose is to implement measurement models of the form:
 *
 * @f$ y_k = h_k(x_k, w_k)@f$
 *
 * With Jacobians:
 *
 * @f$ H_K = \frac{\partial h}{\partial x}(x_k, \mu_k) @f$
 *
 * and
 *
 * @f$ M_k = \frac{\partial h}{\partial w}(x_k, \mu_k) @f$
 *
 * Where @f$ x_k @f$ denotes the system state, @f$ y_k @f$ the
 * measurement, @f$ w_k @f$ the measurement noise and @f$ \mu_k @f$ the
 * measurement noise mean at timestep @f$ k @f$.
 *
 * To implement a measurement model that works with the ExtendedKalmanFilter,
 * inherit from this class and implement your own observe() function.
 *
 * If the Jacobians can be computed analytically, it is recommended to do so and overload the
 * getMeasurementJacobian() and getNoiseJacobian() functions.
 */
class LinearizedMeasurementModel : public MeasurementModelBase {
 public:
  /**
   * @brief Function to get @f$ H_k @f$, which is the measurement model
   *        Jacobian w.r.t. the system state.
   */
  virtual Eigen::MatrixXd getMeasurementJacobian(
      const Eigen::VectorXd& state) const = 0;
  /**
   * @brief Function to get @f$ M_k @f$, which is the measurement model
   *        Jacobian w.r.t. the measurement noise.
   */
  virtual Eigen::MatrixXd getNoiseJacobian(
      const Eigen::VectorXd& state) const = 0;

 protected:
  /** @brief Default constructor should not be used. */
  LinearizedMeasurementModel() = delete;
  /** @brief Constructor for a linearized measurement model. */
  LinearizedMeasurementModel(const size_t& state_dim,
                             const size_t& measurement_dim,
                             const DistributionInterface& measurement_noise);
};

}  // namespace refill

#endif  // REFILL_MEASUREMENT_MODELS_LINEARIZED_MEASUREMENT_MODEL_H_
