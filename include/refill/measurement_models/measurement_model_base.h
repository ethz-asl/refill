#ifndef REFILL_MEASUREMENT_MODELS_MEASUREMENT_MODEL_BASE_H_
#define REFILL_MEASUREMENT_MODELS_MEASUREMENT_MODEL_BASE_H_

#include <Eigen/Dense>
#include <glog/logging.h>

#include <cstdlib>
#include <memory>

#include "refill/distributions/distribution_base.h"

namespace refill {

class MeasurementModelBase {
 public:
  virtual Eigen::VectorXd observe(const Eigen::VectorXd& state) const = 0;

  std::size_t getStateDim() const;
  std::size_t getMeasurementDim() const;
  std::size_t getMeasurementNoiseDim() const;
  DistributionInterface* getMeasurementNoise() const;

 protected:
  MeasurementModelBase() = delete;
  MeasurementModelBase(const std::size_t& state_dim,
                       const std::size_t& measurement_dim,
                       const DistributionInterface& measurement_noise);

  void setMeasurementModelBaseParameters(
      const std::size_t& state_dim, const std::size_t& measurement_dim,
      const DistributionInterface& measurement_noise);

 private:
  std::size_t state_dim_;
  std::size_t measurement_dim_;
  std::unique_ptr<DistributionInterface> measurement_noise_;
};

}  // namespace refill

#endif  // REFILL_MEASUREMENT_MODELS_MEASUREMENT_MODEL_BASE_H_
