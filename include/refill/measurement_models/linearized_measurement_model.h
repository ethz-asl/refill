#ifndef REFILL_MEASUREMENT_MODELS_LINEARIZED_MEASUREMENT_MODEL_H_
#define REFILL_MEASUREMENT_MODELS_LINEARIZED_MEASUREMENT_MODEL_H_

#include <Eigen/Dense>
#include <memory>

#include "refill/measurement_models/measurement_model_base.h"

namespace refill {

class LinearizedMeasurementModel : public MeasurementModelBase {
 public:
  LinearizedMeasurementModel() {}

  // TODO(jwidauer): Add comment
  virtual Eigen::MatrixXd getMeasurementJacobian(
      const Eigen::VectorXd& state) const = 0;
  virtual Eigen::MatrixXd getNoiseJacobian(
      const Eigen::VectorXd& state) const = 0;

  int getMeasurementNoiseDim() const;
 protected:
  std::unique_ptr<DistributionInterface> measurement_noise_;
};

}  // namespace refill

#endif  // REFILL_MEASUREMENT_MODELS_LINEARIZED_MEASUREMENT_MODEL_H_
