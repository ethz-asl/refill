#include "refill/measurement_models/linearized_measurement_model.h"

using std::size_t;

namespace refill {

/**
 * Uses a second order central finite difference approximation for the
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

  Eigen::VectorXd x = state;
  Eigen::VectorXd evaluation_1(this->getStateDim());
  Eigen::VectorXd evaluation_2(this->getStateDim());
  for (int i = 0; i < this->getStateDim(); ++i) {
    double h = eps * std::abs(x[i]);
    if (h == 0.0) {
      h = eps;
    }
    x[i] += h;
    evaluation_1 = this->observe(x, this->getMeasurementNoise()->mean());
    x[i] -= 2 * h;
    evaluation_2 = this->observe(x, this->getMeasurementNoise()->mean());
    x[i] = state[i];
    jacobian.col(i) = (evaluation_1 - evaluation_2) / (2 * h);
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
                           this->getMeasurementNoiseDim());
  constexpr double eps = std::sqrt(std::numeric_limits<double>::epsilon());

  const Eigen::VectorXd noise_mean = this->getMeasurementNoise()->mean();
  Eigen::VectorXd x = noise_mean;
  Eigen::VectorXd evaluation_1(this->getMeasurementDim());
  Eigen::VectorXd evaluation_2(this->getMeasurementDim());
  for (int i = 0; i < this->getMeasurementNoiseDim(); ++i) {
    double h = eps * std::abs(x[i]);
    if (h == 0.0) {
      h = eps;
    }
    x[i] += h;
    evaluation_1 = this->observe(state, x);
    x[i] -= 2 * h;
    evaluation_2 = this->observe(state, x);
    x[i] = noise_mean[i];
    jacobian.col(i) = (evaluation_1 - evaluation_2) / (2 * h);
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
