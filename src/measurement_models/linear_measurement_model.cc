#include "refill/measurement_models/linear_measurement_model.h"

namespace refill {

/**
 * This constructor sets up an empty linear measurement model.
 *
 * It can be useful for class member construction.
 *
 * To be able to use the model after using this constructor, first set the
 * parameters using the setModelParameters() function.
 */
LinearMeasurementModel::LinearMeasurementModel()
    : LinearMeasurementModel(Eigen::MatrixXd::Identity(0, 0),
                             GaussianDistribution(0),
                             Eigen::MatrixXd::Identity(0, 0)) {}

/**
 * @param measurement_model Measurement model which will be copied.
 */
LinearMeasurementModel::LinearMeasurementModel(
    const LinearMeasurementModel& measurement_model)
    : LinearMeasurementModel(measurement_model.measurement_mapping_,
                             *(measurement_model.getNoise()),
                             measurement_model.noise_mapping_) {}

/**
 * This constructor sets the noise mapping to an identity matrix.
 *
 * @param measurement_mapping The matrix @f$ H_k @f$.
 * @param measurement_noise The measurement noise @f$ w_k @f$.
 */
LinearMeasurementModel::LinearMeasurementModel(
    const Eigen::MatrixXd& measurement_mapping,
    const DistributionInterface& measurement_noise)
    : LinearMeasurementModel(
          measurement_mapping, measurement_noise,
          Eigen::MatrixXd::Identity(measurement_mapping.rows(),
                                    measurement_noise.mean().size())) {}

/**
 * This constructor sets all linear measurement model parameters.
 *
 * It also checks that the dimensions of @e noise_mapping are right.
 *
 * @param measurement_mapping The matrix @f$ H_k @f$.
 * @param measurement_noise The measurement noise @f$ w_k @f$
 * @param noise_mapping The matrix @f$ M_k @f$.
 */
LinearMeasurementModel::LinearMeasurementModel(
    const Eigen::MatrixXd& measurement_mapping,
    const DistributionInterface& measurement_noise,
    const Eigen::MatrixXd& noise_mapping)
    : LinearizedMeasurementModel(measurement_mapping.cols(),
                                 measurement_mapping.rows(),
                                 measurement_noise) {
  this->setModelParameters(measurement_mapping, measurement_noise,
                                 noise_mapping);
}

/**
 * Sets the noise mapping to an identity matrix.
 *
 * @param measurement_mapping The matrix @f$ H_k @f$.
 * @param measurement_noise The measurement noise @f$ w_k @f$.
 */
void LinearMeasurementModel::setModelParameters(
    const Eigen::MatrixXd& measurement_mapping,
    const DistributionInterface& measurement_noise) {
  this->setModelParameters(
      measurement_mapping, measurement_noise,
      Eigen::MatrixXd::Identity(measurement_mapping.rows(),
                                measurement_noise.mean().size()));
}

/**
 * Sets all parameters of the linear measurement model.
 *
 * Also checks that the dimensions of @e noise_mapping are right.
 *
 * @param measurement_mapping The matrix @f$ H_k @f$.
 * @param measurement_noise The measurement noise @f$ w_k @f$.
 * @param noise_mapping The matrix @f$ M_k @f$.
 */
void LinearMeasurementModel::setModelParameters(
    const Eigen::MatrixXd& measurement_mapping,
    const DistributionInterface& measurement_noise,
    const Eigen::MatrixXd& noise_mapping) {
  CHECK_EQ(measurement_mapping.rows(), noise_mapping.rows());
  CHECK_EQ(noise_mapping.cols(), measurement_noise.mean().size());

  measurement_mapping_ = measurement_mapping;
  noise_mapping_ = noise_mapping;

  this->setMeasurementModelBaseParameters(measurement_mapping.cols(),
                                          measurement_mapping.rows(),
                                          measurement_noise);
}

/**
 * Also checks that the state vector has compatible dimensions with the
 * measurement model.
 *
 * @param state The current system state.
 * @return the expected measurement.
 */
Eigen::VectorXd LinearMeasurementModel::observe(
    const Eigen::VectorXd& state, const Eigen::VectorXd& noise) const {
  CHECK_NE(this->getStateDim(), 0)
      << " Measurement model has not been initialized.";
  CHECK_EQ(state.size(), this->getStateDim());
  CHECK_EQ(noise.size(), this->getNoiseDim());

  return measurement_mapping_ * state +
         noise_mapping_ * noise;
}

/**
 * Since this is only a linear model, it only returns @f$ H_k @f$.
 *
 * @param state The current system state.
 * @return the measurement model Jacobian w.r.t. the system state.
 */
Eigen::MatrixXd LinearMeasurementModel::getMeasurementJacobian(
    const Eigen::VectorXd&) const {
  CHECK_NE(this->getStateDim(), 0)
      << " Measurement model has not been initialized.";
  return measurement_mapping_;
}

/**
 * Since this is only a linear model, it only returns @f$ M_k @f$.
 *
 * @param state The current system state.
 * @return the measurement model Jacobian w.r.t. the measurement noise.
 */
Eigen::MatrixXd LinearMeasurementModel::getNoiseJacobian(
    const Eigen::VectorXd&) const {
  CHECK_NE(this->getStateDim(), 0)
      << " Measurement model has not been initialized.";
  return noise_mapping_;
}

/**
 * @return the current measurement mapping.
 */
Eigen::MatrixXd LinearMeasurementModel::getMeasurementMapping() const {
  CHECK_NE(this->getStateDim(), 0)
      << "Measurement model has not been initialized.";
  return measurement_mapping_;
}

/**
 * @return the current noise mapping.
 */
Eigen::MatrixXd LinearMeasurementModel::getNoiseMapping() const {
  CHECK_NE(this->getStateDim(), 0)
      << "Measurement model has not been initialized.";
  return noise_mapping_;
}

double LinearMeasurementModel::getLikelihood(
    const Eigen::VectorXd& state, const Eigen::VectorXd& measurement) const {
  CHECK_EQ(this->getStateDim(), state.size());
  CHECK_EQ(this->getMeasurementDim(), measurement.size());

  // Solve M_k * w_k = y_k - H_k * x_k
  Eigen::VectorXd w_k = noise_mapping_.colPivHouseholderQr().solve(
      measurement - measurement_mapping_ * state);

  return this->getNoise()->evaluatePdf(w_k);
}

Eigen::VectorXd LinearMeasurementModel::getLikelihoodVectorized(
    const Eigen::MatrixXd& sampled_state,
    const Eigen::VectorXd& measurement) const {
  CHECK_EQ(this->getStateDim(), sampled_state.rows());
  CHECK_EQ(this->getMeasurementDim(), measurement.size());

  Eigen::MatrixXd expected_measurements = -measurement_mapping_ * sampled_state;

  Eigen::MatrixXd sampled_w_k = noise_mapping_.colPivHouseholderQr().solve(
      expected_measurements.colwise() + measurement);

  return this->getNoise()->evaluatePdfVectorized(sampled_w_k);
}

}  // namespace refill
