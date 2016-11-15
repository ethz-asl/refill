#ifndef REFILL_MEASUREMENT_MODELS_MEASUREMENT_MODEL_BASE_H_
#define REFILL_MEASUREMENT_MODELS_MEASUREMENT_MODEL_BASE_H_

#include <Eigen/Dense>

namespace refill {

template<int STATE_DIM, int MEAS_DIM>
class MeasurementModelBase {
 public:
  virtual Eigen::Matrix<double, MEAS_DIM, 1> observe(
      const Eigen::Matrix<double, STATE_DIM, 1>& state) = 0;
  virtual int dimension() const = 0;
};

}  // namespace refill

#endif  // REFILL_MEASUREMENT_MODELS_MEASUREMENT_MODEL_BASE_H_
