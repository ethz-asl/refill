#ifndef REFILL_MEASUREMENT_MODELS_LINEAR_MEASUREMENT_MODEL_H_
#define REFILL_MEASUREMENT_MODELS_LINEAR_MEASUREMENT_MODEL_H_

#include <glog/logging.h>
#include <memory>

#include "refill/distributions/gaussian_distribution.h"
#include "refill/measurement_models/measurement_model_base.h"

namespace refill {

template<int STATE_DIM = Eigen::Dynamic, int MEAS_DIM = Eigen::Dynamic>
class LinearMeasurementModel
    : public MeasurementModelBase<STATE_DIM, MEAS_DIM> {
 public:
  LinearMeasurementModel();
  explicit LinearMeasurementModel(
      const Eigen::Matrix<double, MEAS_DIM, STATE_DIM>& measurement_mat);
  LinearMeasurementModel(
      const Eigen::Matrix<double, MEAS_DIM, STATE_DIM>& measurement_mat,
      const DistributionInterface<MEAS_DIM>& measurement_noise);

  int getStateDim() const {
    return measurement_mat_.cols();
  }
  int getMeasurementDim() const {
    return measurement_mat_.rows();
  }

  Eigen::Matrix<double, MEAS_DIM, 1> observe(
      const Eigen::Matrix<double, STATE_DIM, 1>& state) const;

  DistributionInterface<MEAS_DIM>* getMeasurementNoise() const {
    CHECK_NE(measurement_noise_.get(),
             static_cast<DistributionInterface<MEAS_DIM>*>(nullptr));
    return measurement_noise_.get();
  }

  Eigen::Matrix<double, MEAS_DIM, STATE_DIM> getJacobian() const {
    return measurement_mat_;
  }

 private:
  Eigen::Matrix<double, MEAS_DIM, STATE_DIM> measurement_mat_;
  std::unique_ptr<DistributionInterface<MEAS_DIM>> measurement_noise_;
};

typedef LinearMeasurementModel<Eigen::Dynamic, Eigen::Dynamic>
        LinearMeasurementModelXd;

// Function definitions

template<int STATE_DIM, int MEAS_DIM>
LinearMeasurementModel<STATE_DIM, MEAS_DIM>::LinearMeasurementModel() {
  constexpr int kCurrentSystemDim =
      (STATE_DIM == Eigen::Dynamic) ? 1 : STATE_DIM;
  constexpr int kCurrentMeasurementDim =
      (MEAS_DIM == Eigen::Dynamic) ? 1 : MEAS_DIM;

  measurement_mat_ = Eigen::Matrix<double, kCurrentMeasurementDim,
      kCurrentSystemDim>::Identity(kCurrentMeasurementDim, kCurrentSystemDim);

  // In case of no declaration of measurement noise,
  // we use standard normal gaussian.
  measurement_noise_.reset(new GaussianDistribution<kCurrentMeasurementDim>());
}

template<int STATE_DIM, int MEAS_DIM>
LinearMeasurementModel<STATE_DIM, MEAS_DIM>::LinearMeasurementModel(
    const Eigen::Matrix<double, MEAS_DIM, STATE_DIM>& measurement_mat)
    : measurement_mat_(measurement_mat) {
  measurement_noise_.reset(new GaussianDistribution<MEAS_DIM>());
}

template<int STATE_DIM, int MEAS_DIM>
LinearMeasurementModel<STATE_DIM, MEAS_DIM>::LinearMeasurementModel(
    const Eigen::Matrix<double, MEAS_DIM, STATE_DIM>& measurement_mat,
    const DistributionInterface<MEAS_DIM>& measurement_noise)
    : measurement_mat_(measurement_mat),
      measurement_noise_(measurement_noise.clone()) {

  if (MEAS_DIM == Eigen::Dynamic) {
    CHECK_EQ(measurement_mat_.rows(), measurement_noise_->mean().size());
  }
}

template<int STATE_DIM, int MEAS_DIM>
Eigen::Matrix<double, MEAS_DIM, 1>
  LinearMeasurementModel<STATE_DIM, MEAS_DIM>::observe(
    const Eigen::Matrix<double, STATE_DIM, 1>& state) const {
  if (STATE_DIM == Eigen::Dynamic) {
    CHECK_EQ(state.size(), measurement_mat_.cols());
  }

  return measurement_mat_ * state + measurement_noise_->mean();
}

}  // namespace refill

#endif  // REFILL_MEASUREMENT_MODELS_LINEAR_MEASUREMENT_MODEL_H_
