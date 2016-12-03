#ifndef REFILL_MEASUREMENT_MODELS_LINEARIZED_MEASUREMENT_MODEL_H_
#define REFILL_MEASUREMENT_MODELS_LINEARIZED_MEASUREMENT_MODEL_H_

#include <Eigen/Dense>

#include <cstdlib>

#include "refill/measurement_models/measurement_model_base.h"

namespace refill {

class LinearizedMeasurementModel : public MeasurementModelBase {
 public:
  virtual Eigen::MatrixXd getMeasurementJacobian(
      const Eigen::VectorXd& state) const = 0;
  virtual Eigen::MatrixXd getNoiseJacobian(
      const Eigen::VectorXd& state) const = 0;

 protected:
  LinearizedMeasurementModel() = delete;
  LinearizedMeasurementModel(const std::size_t& state_dim,
                             const std::size_t& measurement_dim,
                             const DistributionInterface& measurement_noise);
};

}  // namespace refill

#endif  // REFILL_MEASUREMENT_MODELS_LINEARIZED_MEASUREMENT_MODEL_H_
