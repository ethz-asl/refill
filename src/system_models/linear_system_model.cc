#include "refill/system_models/linear_system_model.h"

namespace refill {

LinearSystemModel::LinearSystemModel()
    : LinearSystemModel(Eigen::MatrixXd::Identity(0, 0), GaussianDistribution(),
                        Eigen::MatrixXd::Zero(0, 0),
                        Eigen::MatrixXd::Identity(0, 0)) {}

// If constructor is called without input matrix, we assume there is no input.
LinearSystemModel::LinearSystemModel(const Eigen::MatrixXd& system_mapping,
                                     const DistributionInterface& system_noise)
    : LinearSystemModel(system_mapping, system_noise,
                        Eigen::MatrixXd::Zero(0, 0),
                        Eigen::MatrixXd::Identity(system_mapping.rows(),
                                                  system_noise.mean().size())) {
}

LinearSystemModel::LinearSystemModel(const Eigen::MatrixXd& system_mapping,
                                     const DistributionInterface& system_noise,
                                     const Eigen::MatrixXd& input_mapping)
    : LinearSystemModel(system_mapping, system_noise, input_mapping,
                        Eigen::MatrixXd::Identity(system_mapping.rows(),
                                                  system_noise.mean().size())) {
}

LinearSystemModel::LinearSystemModel(const Eigen::MatrixXd& system_mapping,
                                     const DistributionInterface& system_noise,
                                     const Eigen::MatrixXd& input_mapping,
                                     const Eigen::MatrixXd& noise_mapping)
    : LinearizedSystemModel(system_mapping.rows(), system_noise,
                            input_mapping.cols()) {
  this->setSystemParameters(system_mapping, system_noise, input_mapping,
                            noise_mapping);
}

void LinearSystemModel::setSystemParameters(
    const Eigen::MatrixXd& system_mapping,
    const DistributionInterface& system_noise) {
  this->setSystemParameters(
      system_mapping, system_noise, Eigen::MatrixXd::Zero(0, 0),
      Eigen::MatrixXd::Identity(system_mapping.rows(),
                                system_noise.mean().size()));
}

void LinearSystemModel::setSystemParameters(
    const Eigen::MatrixXd& system_mapping,
    const DistributionInterface& system_noise,
    const Eigen::MatrixXd& input_mapping) {
  this->setSystemParameters(
      system_mapping, system_noise, input_mapping,
      Eigen::MatrixXd::Identity(system_mapping.rows(),
                                system_noise.mean().size()));
}

void LinearSystemModel::setSystemParameters(
    const Eigen::MatrixXd& system_mapping,
    const DistributionInterface& system_noise,
    const Eigen::MatrixXd& input_mapping,
    const Eigen::MatrixXd& noise_mapping) {
  CHECK_EQ(system_mapping.rows(), system_mapping.cols());
  CHECK_EQ(system_mapping.rows(), noise_mapping.rows());
  CHECK_EQ(noise_mapping.cols(), system_noise.mean().size());

  if (input_mapping.size() != 0) {
    CHECK_EQ(system_mapping.rows(), input_mapping.rows());
  }

  system_mapping_ = system_mapping;
  input_mapping_ = input_mapping;
  noise_mapping_ = noise_mapping;

  this->setSystemModelBaseParameters(system_mapping.rows(), system_noise,
                                     input_mapping.cols());
}

Eigen::VectorXd LinearSystemModel::propagate(
    const Eigen::VectorXd& state) const {
  return this->propagate(state, Eigen::VectorXd::Zero(this->getInputDim()),
                         this->getSystemNoise()->mean());
}

Eigen::VectorXd LinearSystemModel::propagate(
    const Eigen::VectorXd& state, const Eigen::VectorXd& input,
    const Eigen::VectorXd& noise) const {
  CHECK_NE(this->getStateDim(), 0) << "System model has not been initialized.";

  CHECK_EQ(state.size(), this->getStateDim());
  CHECK_EQ(input.size(), this->getInputDim());

  // If there is no input to the system,
  // we don't need to compute the matrix multiplication.
  if (input_mapping_.size() == 0 ||
      input == Eigen::VectorXd::Zero(this->getInputDim())) {
    return system_mapping_ * state + noise_mapping_ * noise;
  } else {
    return system_mapping_ * state + input_mapping_ * input +
           noise_mapping_ * noise;
  }
}

Eigen::MatrixXd LinearSystemModel::getStateJacobian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& input) const {
  CHECK_NE(this->getStateDim(), 0) << "System model has not been initialized.";
  return system_mapping_;
}

Eigen::MatrixXd LinearSystemModel::getNoiseJacobian(
    const Eigen::VectorXd& state, const Eigen::VectorXd& input) const {
  CHECK_NE(this->getStateDim(), 0) << "System model has not been initialized.";
  return noise_mapping_;
}

}  // namespace refill
