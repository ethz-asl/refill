#ifndef REFILL_MEASUREMENT_MODELS_MEASUREMENT_MODEL_BASE_H_
#define REFILL_MEASUREMENT_MODELS_MEASUREMENT_MODEL_BASE_H_

#include <Eigen/Dense>

#include "refill/distributions/distribution_base.h"

namespace refill {

class MeasurementModelBase {
 public:
  virtual Eigen::VectorXd observe(
      const Eigen::VectorXd& state) const = 0;
  virtual int getStateDim() const = 0;
  virtual int getMeasurementDim() const = 0;
  virtual DistributionInterface* getMeasurementNoise() const = 0;
  virtual Eigen::MatrixXd getJacobian() const = 0;
};

}  // namespace refill

#endif  // REFILL_MEASUREMENT_MODELS_MEASUREMENT_MODEL_BASE_H_
