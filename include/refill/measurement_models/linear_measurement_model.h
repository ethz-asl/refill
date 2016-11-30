#ifndef REFILL_MEASUREMENT_MODELS_LINEAR_MEASUREMENT_MODEL_H_
#define REFILL_MEASUREMENT_MODELS_LINEAR_MEASUREMENT_MODEL_H_

#include <glog/logging.h>
#include <Eigen/Dense>

#include "refill/distributions/gaussian_distribution.h"
#include "refill/measurement_models/linearized_measurement_model.h"

namespace refill {

/**
 * @brief Class that implements a linear measurement model.
 *
 * This class is an implementation of linear measurement models of the form:
 *
 * @f$ y_k = H_k \cdot x_k + M_k \cdot w_k @f$
 *
 * Where @f$ y_k @f$ is the measurement, @f$ x_k @f$ the system state,
 * @f$ w_k @f$ the measurement noise, @f$ H_k @f$ the measurement mapping and
 * @f$ M_k @f$ the noise mapping at time step @f$ k @f$.
 *
 * Use this class together with the LinearSystemModel, noise of type
 * GaussianDistribution and the ExtendedKalmanFilter if you want to implement
 * a simple kalman filter.
 */
class LinearMeasurementModel : public LinearizedMeasurementModel {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Standard constructor creates an one dimensional measurement model
  // with univariate standard normal gaussian noise.
  LinearMeasurementModel();
  /** @brief Constructor for a measurement model with a simple
   *         noise mapping.
   */
  LinearMeasurementModel(const Eigen::MatrixXd& measurement_mapping,
                         const DistributionInterface& measurement_noise);
  /** @brief Constructor for a measurement model with a specific
   *         noise mapping.
   */
  LinearMeasurementModel(const Eigen::MatrixXd& measurement_mapping,
                         const DistributionInterface& measurement_noise,
                         const Eigen::MatrixXd& noise_mapping);

  /** @brief Sets the measurement model parameters for a model with a simple
   *         noise mapping.
   */
  void setMeasurementParameters(const Eigen::MatrixXd& measurement_mapping,
                                const DistributionInterface& measurement_noise);
  /** @brief Sets the measurement model parameters for a model with a specific
   *         noise mapping.
   */
  void setMeasurementParameters(const Eigen::MatrixXd& measurement_mapping,
                                const DistributionInterface& measurement_noise,
                                const Eigen::MatrixXd& noise_mapping);

  /** @brief Use the measurement model to receive the expected measurement. */
  Eigen::VectorXd observe(const Eigen::VectorXd& state) const;

  /**
   * @brief Function to get @f$ H_k @f$, which is the measurement model
   *        Jacobian w.r.t. the system state.
   */
  Eigen::MatrixXd getMeasurementJacobian(const Eigen::VectorXd& state) const;
  /** @brief Function to get @f$ M_k @f$, which is the measurement model
   *         Jacobian w.r.t. the measurement noise.
   */
  Eigen::MatrixXd getNoiseJacobian(const Eigen::VectorXd& state) const;

 private:
  Eigen::MatrixXd measurement_mapping_;
  Eigen::MatrixXd noise_mapping_;
};

}  // namespace refill

#endif  // REFILL_MEASUREMENT_MODELS_LINEAR_MEASUREMENT_MODEL_H_
