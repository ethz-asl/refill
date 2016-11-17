#ifndef REFILL_MEASUREMENT_MODELS_LINEARIZED_MEASUREMENT_MODEL_H_
#define REFILL_MEASUREMENT_MODELS_LINEARIZED_MEASUREMENT_MODEL_H_

#include "refill/measurement_models/measurement_model_base.h"

namespace refill {

class LinearizedMeasurementModel : public MeasurementModelBase {
 public:
  // TODO(jwidauer): Add comment
  virtual Eigen::MatrixXd getJacobian() const = 0;
};

}  // namespace refill

#endif  // REFILL_MEASUREMENT_MODELS_LINEARIZED_MEASUREMENT_MODEL_H_
