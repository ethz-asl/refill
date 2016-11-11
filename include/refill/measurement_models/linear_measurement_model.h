#ifndef REFILL_MEASUREMENT_MODELS_LINEAR_MEASUREMENT_MODEL_H_
#define REFILL_MEASUREMENT_MODELS_LINEAR_MEASUREMENT_MODEL_H_

#include <memory>

#include "refill/measurement_models/measurement_model_base.h"
#include "refill/distributions/gaussian_distribution.h"

namespace refill {

template<int STATEDIM = Eigen::Dynamic, int MEASDIM = Eigen::Dynamic>
class LinearMeasurementModel : public MeasurementModelBase<STATEDIM, MEASDIM> {
 public:
  LinearMeasurementModel();
  LinearMeasurementModel(
      const Eigen::Matrix<double, MEASDIM, STATEDIM>& measurement_mat,
      const DistributionInterface<MEASDIM>& measurement_noise =
          GaussianDistribution<MEASDIM>());

  int GetStateDim() const {
    return measurement_mat_.cols();
  }
  int GetMeasurementDim() const {
    return measurement_mat_.rows();
  }

  Eigen::Matrix<double, MEASDIM, 1> Observe(
      const Eigen::Matrix<double, STATEDIM, 1>& state) const;

  DistributionInterface<MEASDIM>* GetMeasurementNoise() const {
    return measurement_noise_.get();
  }

  Eigen::Matrix<double, MEASDIM, STATEDIM> GetJacobian() const {
    return measurement_mat_;
  }

 private:
  Eigen::Matrix<double, MEASDIM, STATEDIM> measurement_mat_;
  std::unique_ptr<DistributionInterface<MEASDIM>> measurement_noise_;
};

typedef LinearMeasurementModel<Eigen::Dynamic, Eigen::Dynamic>
        LinearMeasurementModelXd;

}  // namespace refill

#include "./linear_measurement_model-inl.h"

#endif  // REFILL_MEASUREMENT_MODELS_LINEAR_MEASUREMENT_MODEL_H_
