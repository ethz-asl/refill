#include "refill/measurement_models/linear_measurement_model.h"

namespace refill {

LinearMeasurementModel::LinearMeasurementModel() {
  // In case of call to standard constructor,
  // we assume a one dimensional system, with one measurement.
  measurement_mapping_ = Eigen::MatrixXd::Identity(1, 1);

  // In case of call to standard constructor,
  // we use univariate standard normal gaussian.
  measurement_noise_.reset(new GaussianDistribution(1));
}

LinearMeasurementModel::LinearMeasurementModel(
    const Eigen::MatrixXd& measurement_mat)
    : measurement_mapping_(measurement_mat) {
  measurement_noise_.reset(new GaussianDistribution(measurement_mat.rows()));
}

LinearMeasurementModel::LinearMeasurementModel(
    const Eigen::MatrixXd& measurement_mat,
    const DistributionInterface& measurement_noise)
    : measurement_mapping_(measurement_mat),
      measurement_noise_(measurement_noise.clone()) {
  CHECK_EQ(measurement_mapping_.rows(), measurement_noise.mean().size());
}

void LinearMeasurementModel::setMeasurementParameters(
    const Eigen::MatrixXd& measurement_mat,
    const DistributionInterface& measurement_noise) {
  CHECK_EQ(measurement_mat.rows(), measurement_noise.mean().size());

  measurement_mapping_ = measurement_mat;
  measurement_noise_.reset(measurement_noise.clone());
}

Eigen::VectorXd LinearMeasurementModel::observe(
    const Eigen::VectorXd& state) const {
  CHECK_EQ(state.size(), measurement_mapping_.cols());

  return measurement_mapping_ * state + measurement_noise_->mean();
}

int LinearMeasurementModel::getStateDim() const {
  return measurement_mapping_.cols();
}

int LinearMeasurementModel::getMeasurementDim() const {
  return measurement_mapping_.rows();
}

DistributionInterface* LinearMeasurementModel::getMeasurementNoise() const {
  return measurement_noise_.get();
}

Eigen::MatrixXd LinearMeasurementModel::getJacobian() const {
  return measurement_mapping_;
}

}  // namespace refill
