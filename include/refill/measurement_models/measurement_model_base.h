#ifndef REFILL_MEASUREMENT_MODELS_MEASUREMENT_MODEL_BASE_H_
#define REFILL_MEASUREMENT_MODELS_MEASUREMENT_MODEL_BASE_H_

#include <Eigen/Dense>

namespace refill {

template<int STATEDIM, int MEASDIM>
class MeasurementModelBase {
 public:
  virtual Eigen::Matrix<double, MEASDIM, 1> Observe(
      const Eigen::Matrix<double, STATEDIM, 1>& state) = 0;
  virtual int dim() const = 0;
};

}  // namespace refill

#endif  // REFILL_MEASUREMENT_MODELS_MEASUREMENT_MODEL_BASE_H_
