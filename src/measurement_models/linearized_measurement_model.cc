#include "refill/measurement_models/linearized_measurement_model.h"

using std::size_t;

namespace refill {

/**
 * Uses a second order central finite difference approdiff_coeffimation for the
 * Jacobian computation, if not overloaded.
 *
 * @param state The current system state.
 * @return the measurement model Jacobian w.r.t. the system state.
 */
Eigen::MatrixXd LinearizedMeasurementModel::getMeasurementJacobian(
    const Eigen::VectorXd& state) const {
  CHECK_EQ(state.rows(), this->getStateDim());

  Eigen::MatrixXd jacobian(this->getMeasurementDim(), this->getStateDim());
  constexpr double eps = std::sqrt(std::numeric_limits<double>::epsilon());

  Eigen::VectorXd diff_coeff = state;
  Eigen::VectorXd evaluation_1(this->getStateDim());
  Eigen::VectorXd evaluation_2(this->getStateDim());
  for (std::size_t i = 0; i < this->getStateDim(); ++i) {
    double step_size = eps * std::abs(diff_coeff[i]);
    if (step_size == 0.0) {
      step_size = eps;
    }
    diff_coeff[i] += step_size;
    evaluation_1 = this->observe(diff_coeff,
                                 this->getNoise()->mean());
    diff_coeff[i] -= 2 * step_size;
    evaluation_2 = this->observe(diff_coeff,
                                 this->getNoise()->mean());
    diff_coeff[i] = state[i];
    jacobian.col(i) = (evaluation_1 - evaluation_2) / (2 * step_size);
  }
  return jacobian;
}

/**
 * Uses a second order central finite difference approximation for the
 * Jacobian computation, if not overloaded.
 *
 * @param state The current system state.
 * @return the measurement model Jacobian w.r.t. the measurement noise.
 */
Eigen::MatrixXd LinearizedMeasurementModel::getNoiseJacobian(
    const Eigen::VectorXd& state) const {
  CHECK_EQ(state.rows(), this->getStateDim());

  Eigen::MatrixXd jacobian(this->getMeasurementDim(),
                           this->getNoiseDim());
  constexpr double eps = std::sqrt(std::numeric_limits<double>::epsilon());

  const Eigen::VectorXd noise_mean = this->getNoise()->mean();
  Eigen::VectorXd diff_coeff = noise_mean;
  Eigen::VectorXd evaluation_1(this->getMeasurementDim());
  Eigen::VectorXd evaluation_2(this->getMeasurementDim());
  for (std::size_t i = 0; i < this->getNoiseDim(); ++i) {
    double step_size = eps * std::abs(diff_coeff[i]);
    if (step_size == 0.0) {
      step_size = eps;
    }
    diff_coeff[i] += step_size;
    evaluation_1 = this->observe(state, diff_coeff);
    diff_coeff[i] -= 2 * step_size;
    evaluation_2 = this->observe(state, diff_coeff);
    diff_coeff[i] = noise_mean[i];
    jacobian.col(i) = (evaluation_1 - evaluation_2) / (2 * step_size);
  }
  return jacobian;
}

/**
 * Use this constructor to create a linearized measurement model.
 * The constructor clones the measurement noise, so it can be used again.
 *
 * @param state_dim The measurement models state dimension.
 * @param measurement_dim The measurement models measurement dimension.
 * @param measurement_noise The measurement noise.
 */
LinearizedMeasurementModel::LinearizedMeasurementModel(
    const size_t& state_dim, const size_t& measurement_dim,
    const DistributionInterface& measurement_noise)
    : MeasurementModelBase(state_dim, measurement_dim, measurement_noise) {}

}  // namespace refill
