#ifndef REFILL_MEASUREMENT_MODELS_LINEAR_MEASUREMENT_MODEL_H_
#define REFILL_MEASUREMENT_MODELS_LINEAR_MEASUREMENT_MODEL_H_

#include <memory>

#include "refill/measurement_models/measurement_model_base.h"
#include "refill/distributions/gaussian_distribution.h"

namespace refill {

template<int STATE_DIM = Eigen::Dynamic, int MEAS_DIM = Eigen::Dynamic>
class LinearMeasurementModel :
    public MeasurementModelBase<STATE_DIM, MEAS_DIM> {
 public:
  LinearMeasurementModel();
  LinearMeasurementModel(
      const Eigen::Matrix<double, MEAS_DIM, STATE_DIM>& measurement_mat,
      const DistributionInterface<MEAS_DIM>& measurement_noise =
          GaussianDistribution<MEAS_DIM>());

  int GetStateDim() const {
    return measurement_mat_.cols();
  }
  int GetMeasurementDim() const {
    return measurement_mat_.rows();
  }

  Eigen::Matrix<double, MEAS_DIM, 1> Observe(
      const Eigen::Matrix<double, STATE_DIM, 1>& state) const;

  DistributionInterface<MEAS_DIM>* GetMeasurementNoise() const {
    return measurement_noise_.get();
  }

  Eigen::Matrix<double, MEAS_DIM, STATE_DIM> GetJacobian() const {
    return measurement_mat_;
  }

 private:
  Eigen::Matrix<double, MEAS_DIM, STATE_DIM> measurement_mat_;
  std::unique_ptr<DistributionInterface<MEAS_DIM>> measurement_noise_;
};

typedef LinearMeasurementModel<Eigen::Dynamic, Eigen::Dynamic>
        LinearMeasurementModelXd;

}  // namespace refill

#include "./linear_measurement_model-inl.h"

#endif  // REFILL_MEASUREMENT_MODELS_LINEAR_MEASUREMENT_MODEL_H_
