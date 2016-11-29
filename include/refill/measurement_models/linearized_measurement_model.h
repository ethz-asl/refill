#ifndef REFILL_MEASUREMENT_MODELS_LINEARIZED_MEASUREMENT_MODEL_H_
#define REFILL_MEASUREMENT_MODELS_LINEARIZED_MEASUREMENT_MODEL_H_

#include <Eigen/Dense>
#include <memory>

#include "refill/measurement_models/measurement_model_base.h"

namespace refill {

class LinearizedMeasurementModel : public MeasurementModelBase {
 public:
  virtual Eigen::MatrixXd getMeasurementJacobian(
      const Eigen::VectorXd& state) const = 0;
  virtual Eigen::MatrixXd getNoiseJacobian(
      const Eigen::VectorXd& state) const = 0;

  std::size_t getStateDim() const;
  std::size_t getMeasurementDim() const;
  std::size_t getMeasurementNoiseDim() const;
  DistributionInterface* getMeasurementNoise() const;

 protected:
  LinearizedMeasurementModel() = delete;
  LinearizedMeasurementModel(const std::size_t& state_dim,
                             const std::size_t& measurement_dim,
                             const DistributionInterface& measurement_noise);

  void setLinearizedMeasurementModelParameters(
      const std::size_t& state_dim, const std::size_t& measurement_dim,
      const DistributionInterface& measurement_noise);

  std::unique_ptr<DistributionInterface> measurement_noise_;
  std::size_t state_dim_;
  std::size_t measurement_dim_;
};

}  // namespace refill

#endif  // REFILL_MEASUREMENT_MODELS_LINEARIZED_MEASUREMENT_MODEL_H_
