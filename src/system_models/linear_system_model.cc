#include "refill/system_models/linear_system_model.h"

namespace refill {

/**
 * This constructor sets up an empty linear system model.
 *
 * It can be useful for class member construction.
 *
 * To be able to use the model after using this constructor, first set the
 * parameters using the setModelParameters() function.
 */
LinearSystemModel::LinearSystemModel()
    : LinearSystemModel(Eigen::MatrixXd::Identity(0, 0),
                        GaussianDistribution(0), Eigen::MatrixXd::Zero(0, 0),
                        Eigen::MatrixXd::Identity(0, 0)) {
}

/**
 * @param system_model System model which will be copied.
 */
LinearSystemModel::LinearSystemModel(const LinearSystemModel& system_model)
    : LinearSystemModel(system_model.system_mapping_,
                        *(system_model.getNoise()),
                        system_model.input_mapping_,
                        system_model.noise_mapping_) {}

/**
 * This constructor assumes that there is no system input and sets the
 * noise mapping to an identity matrix.
 *
 * It also checks if @e system_mapping is a square matrix.
 *
 * @param system_mapping The matrix @f$ A_k @f$.
 * @param system_noise The system noise @f$ v_k @f$.
 */
LinearSystemModel::LinearSystemModel(const Eigen::MatrixXd& system_mapping,
                                     const DistributionInterface& system_noise)
    : LinearSystemModel(system_mapping, system_noise,
                        Eigen::MatrixXd::Zero(0, 0),
                        Eigen::MatrixXd::Identity(system_mapping.rows(),
                                                  system_noise.mean().size())) {
}

/**
 * This constructor sets the noise mapping to an identity matrix.
 *
 * It also checks whether @e system_mapping is a square matrix and
 * @e input_mapping is of right dimensions if it has a size different from 0.
 *
 * @param system_mapping The matrix @f$ A_k @f$.
 * @param system_noise The system noise @f$ v_k @f$.
 * @param input_mapping The input mapping @f$ B_k @f$.
 */
LinearSystemModel::LinearSystemModel(const Eigen::MatrixXd& system_mapping,
                                     const DistributionInterface& system_noise,
                                     const Eigen::MatrixXd& input_mapping)
    : LinearSystemModel(system_mapping, system_noise, input_mapping,
                        Eigen::MatrixXd::Identity(system_mapping.rows(),
                                                  system_noise.mean().size())) {
}

/**
 * This constructor sets all linear system parameters.
 *
 * It also checks whether @e system_mapping is a square matrix,
 * @e noise_mapping is of right dimensions and, if @e input_mapping has a
 * size different from 0, whether @e input_mapping has the right dimensions.
 *
 * @param system_mapping The system matrix @f$ A_k @f$.
 * @param system_noise The system noise @f$ v_k @f$.
 * @param input_mapping The input mapping @f$ B_k @f$.
 * @param noise_mapping The noise mapping @f$ L_k @f$.
 */
LinearSystemModel::LinearSystemModel(const Eigen::MatrixXd& system_mapping,
                                     const DistributionInterface& system_noise,
                                     const Eigen::MatrixXd& input_mapping,
                                     const Eigen::MatrixXd& noise_mapping)
    : LinearizedSystemModel(system_mapping.rows(), system_noise,
                            input_mapping.cols()) {
  this->setModelParameters(system_mapping, system_noise, input_mapping,
                            noise_mapping);
}

/**
 * Assumes that there is no input mapping and sets the noise mapping to an
 * identity matrix.
 *
 * Also checks whether @e system_mapping is a square matrix.
 *
 * @param system_mapping The system matrix @f$ A_k @f$.
 * @param system_noise The system noise @f$ v_k @f$
 */
void LinearSystemModel::setModelParameters(
    const Eigen::MatrixXd& system_mapping,
    const DistributionInterface& system_noise) {
  this->setModelParameters(
      system_mapping, system_noise, Eigen::MatrixXd::Zero(0, 0),
      Eigen::MatrixXd::Identity(system_mapping.rows(),
                                system_noise.mean().size()));
}

/**
 * Sets the noise mapping to an identity matrix.
 *
 * Also checks whether @e system_mapping is a square matrix and
 * @e input_mapping is of right dimensions if it has a size different from 0.
 *
 * @param system_mapping The system matrix @f$ A_k @f$.
 * @param system_noise The system noise @f$ v_k @f$.
 * @param input_mapping The input mapping @f$ B_k @f$.
 */
void LinearSystemModel::setModelParameters(
    const Eigen::MatrixXd& system_mapping,
    const DistributionInterface& system_noise,
    const Eigen::MatrixXd& input_mapping) {
  this->setModelParameters(
      system_mapping, system_noise, input_mapping,
      Eigen::MatrixXd::Identity(system_mapping.rows(),
                                system_noise.mean().size()));
}

/**
 * Sets all system parameters.
 *
 * Also checks whether @e system_mapping is a square matrix,
 * @e noise_mapping is of right dimensions and, if @e input_mapping has a
 * size different from 0, whether @e input_mapping has the right dimensions.
 *
 * @param system_mapping
 * @param system_noise
 * @param input_mapping
 * @param noise_mapping
 */
void LinearSystemModel::setModelParameters(
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

/**
 * Assumes that the system has no input.
 *
 * Also checks that the state vector has compatible size with the system model.
 *
 * @param state The current system state vector.
 * @return the new state vector.
 */
Eigen::VectorXd LinearSystemModel::propagate(
    const Eigen::VectorXd& state) const {
  return this->propagate(state, Eigen::VectorXd::Zero(this->getInputDim()),
                         this->getNoise()->mean());
}

/**
 * Also checks that the state and input vectors have compatible size with the
 * system model.
 *
 * @param state The current system state vector.
 * @param input The current system input vector.
 * @return the new state vector.
 */
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

/**
 * Since this is a linear system, it only returns @f$ A_k @f$.
 *
 * @param state The current system state.
 * @param input The current system input.
 * @return the system model Jacobian w.r.t. the system state @f$ x_k @f$.
 */
Eigen::MatrixXd LinearSystemModel::getStateJacobian(
    const Eigen::VectorXd&, const Eigen::VectorXd&) const {
  CHECK_NE(this->getStateDim(), 0) << "System model has not been initialized.";
  return system_mapping_;
}

/**
 * Since this is a linear system, it only returns @f$ L_k @f$.
 *
 * @param state The current system state.
 * @param input The current system input.
 * @return the system model Jacobian w.r.t. the system noise @f$ v_k @f$.
 */
Eigen::MatrixXd LinearSystemModel::getNoiseJacobian(
    const Eigen::VectorXd&, const Eigen::VectorXd&) const {
  CHECK_NE(this->getStateDim(), 0) << "System model has not been initialized.";
  return noise_mapping_;
}

/**
 * @return the current system mapping.
 */
Eigen::MatrixXd LinearSystemModel::getSystemMapping() const {
  CHECK_NE(this->getStateDim(), 0) << "System model has not been initialized.";
  return system_mapping_;
}

/**
 * @return the current input mapping.
 */
Eigen::MatrixXd LinearSystemModel::getInputMapping() const {
  CHECK_NE(this->getStateDim(), 0) << "System model has not been initialized.";
  return input_mapping_;
}

/**
 * @return the current noise mapping.
 */
Eigen::MatrixXd LinearSystemModel::getNoiseMapping() const {
  CHECK_NE(this->getStateDim(), 0) << "System model has not been initialized.";
  return noise_mapping_;
}

}  // namespace refill
