#ifndef REFILL_MEASUREMENT_MODELS_LINEARIZED_MEASUREMENT_MODEL_H_
#define REFILL_MEASUREMENT_MODELS_LINEARIZED_MEASUREMENT_MODEL_H_

#include <Eigen/Dense>

#include <memory>

#include "refill/measurement_models/measurement_model_base.h"

namespace refill {

class LinearizedMeasurementModel : public MeasurementModelBase {
 public:
  LinearizedMeasurementModel(const unsigned int& state_dim,
                             const unsigned int& measurement_dim,
                             const DistributionInterface& measurement_noise);

  void setLinearizedMeasurementModelParameters(
      const unsigned int& state_dim, const unsigned int& measurement_dim,
      const DistributionInterface& measurement_noise);

  // TODO(jwidauer): Add comment
  virtual Eigen::MatrixXd getMeasurementJacobian(
      const Eigen::VectorXd& state) const = 0;
  virtual Eigen::MatrixXd getNoiseJacobian(
      const Eigen::VectorXd& state) const = 0;

  unsigned int getStateDim() const;
  unsigned int getMeasurementDim() const;
  unsigned int getMeasurementNoiseDim() const;
  DistributionInterface* getMeasurementNoise() const;
 protected:
  std::unique_ptr<DistributionInterface> measurement_noise_;
  unsigned int state_dim_;
  unsigned int measurement_dim_;
};

}  // namespace refill

#endif  // REFILL_MEASUREMENT_MODELS_LINEARIZED_MEASUREMENT_MODEL_H_
