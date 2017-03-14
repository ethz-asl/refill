#include "refill/system_models/linearized_system_model.h"

using std::size_t;

namespace refill {


/**
 * Uses a second order central finite difference approximation for the
 * Jacobian computation, if not overloaded.
 *
 * @param state The current system state.
 * @param input The current system input.
 * @return the system model Jacobian w.r.t. the system state @f$x_k@f$.
 */
Eigen::MatrixXd LinearizedSystemModel::getStateJacobian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& input) const {
  CHECK_EQ(state.rows(), this->getStateDim());
  CHECK_EQ(input.rows(), this->getInputDim());

  Eigen::MatrixXd jacobian(this->getStateDim(), this->getStateDim());
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
    evaluation_1 = this->propagate(x, input, this->getSystemNoise()->mean());
    x[i] -= 2 * h;
    evaluation_2 = this->propagate(x, input, this->getSystemNoise()->mean());
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
 * @param input The current system input.
 * @return the system model Jacobian w.r.t. the system noise @f$v_k@f$.
 */
Eigen::MatrixXd LinearizedSystemModel::getNoiseJacobian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& input) const {
  CHECK_EQ(state.rows(), this->getStateDim());
  CHECK_EQ(input.rows(), this->getInputDim());

  Eigen::MatrixXd jacobian(this->getStateDim(), this->getSystemNoiseDim());
  constexpr double eps = std::sqrt(std::numeric_limits<double>::epsilon());

  const Eigen::VectorXd noise_mean = this->getSystemNoise()->mean();
  Eigen::VectorXd x = noise_mean;
  Eigen::VectorXd evaluation_1(this->getStateDim());
  Eigen::VectorXd evaluation_2(this->getStateDim());
  for (int i = 0; i < this->getSystemNoiseDim(); ++i) {
    double h = eps * std::abs(x[i]);
    if (h == 0.0) {
      h = eps;
    }
    x[i] += h;
    evaluation_1 = this->propagate(state, input, x);
    x[i] -= 2 * h;
    evaluation_2 = this->propagate(state, input, x);
    x[i] = noise_mean[i];
    jacobian.col(i) = (evaluation_1 - evaluation_2) / (2 * h);
  }
  return jacobian;
}

/**
 * Use this constructor if your system model does not have an input.
 * The constructor clones the system noise, so it can be used again.
 *
 * @param state_dim The systems state dimension.
 * @param system_noise The system noise.
 */
LinearizedSystemModel::LinearizedSystemModel(
    const size_t& state_dim, const DistributionInterface& system_noise)
    : SystemModelBase(state_dim, system_noise, 0) {}

/**
 * Use this constructor if your system does have an input.
 * The constructor clones the system noise, so it can be used again.
 *
 * @param state_dim The systems state dimension.
 * @param system_noise The system noise.
 * @param input_dim The systems input dimension.
 */
LinearizedSystemModel::LinearizedSystemModel(
    const size_t& state_dim, const DistributionInterface& system_noise,
    const size_t& input_dim)
    : SystemModelBase(state_dim, system_noise, input_dim) {}

}  // namespace refill
