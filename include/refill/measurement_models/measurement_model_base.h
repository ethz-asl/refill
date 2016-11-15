#ifndef REFILL_MEASUREMENT_MODELS_MEASUREMENT_MODEL_BASE_H_
#define REFILL_MEASUREMENT_MODELS_MEASUREMENT_MODEL_BASE_H_

#include <Eigen/Dense>

#include "refill/distributions/distribution_base.h"

namespace refill {

template<int STATE_DIM, int MEAS_DIM>
class MeasurementModelBase {
 public:
  virtual Eigen::Matrix<double, MEAS_DIM, 1> observe(
      const Eigen::Matrix<double, STATE_DIM, 1>& state) const = 0;
  virtual int getStateDim() const = 0;
  virtual int getMeasurementDim() const = 0;
  virtual DistributionInterface<MEAS_DIM>* getMeasurementNoise() const = 0;
  virtual Eigen::Matrix<double, MEAS_DIM, STATE_DIM> getJacobian() const = 0;
};

}  // namespace refill

#endif  // REFILL_MEASUREMENT_MODELS_MEASUREMENT_MODEL_BASE_H_
