#ifndef REFILL_MEASUREMENT_MODELS_MEASUREMENT_MODEL_BASE_H_
#define REFILL_MEASUREMENT_MODELS_MEASUREMENT_MODEL_BASE_H_

#include <Eigen/Dense>

#include "refill/distributions/distribution_base.h"

namespace refill {

template<int STATEDIM, int MEASDIM>
class MeasurementModelBase {
 public:
  virtual Eigen::Matrix<double, MEASDIM, 1> Observe(
      const Eigen::Matrix<double, STATEDIM, 1>& state) const = 0;
  virtual int GetStateDim() const = 0;
  virtual int GetMeasurementDim() const = 0;
  virtual DistributionInterface<MEASDIM>* GetMeasurementNoise() const = 0;
  virtual Eigen::Matrix<double, MEASDIM, STATEDIM> GetJacobian() const = 0;
};

}  // namespace refill

#endif  // REFILL_MEASUREMENT_MODELS_MEASUREMENT_MODEL_BASE_H_
