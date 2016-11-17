#ifndef REFILL_SYSTEM_MODELS_LINEARIZABLE_SYSTEM_MODEL_H_
#define REFILL_SYSTEM_MODELS_LINEARIZABLE_SYSTEM_MODEL_H_

#include <Eigen/Dense>

#include "refill/system_models/system_model_base.h"

namespace refill {

class LinearizableSystemModel : public SystemModelBase {
 public:
  // TODO(jwidauer): Add comment
  virtual Eigen::MatrixXd getJacobian() const = 0;
};

}  // namespace refill

#endif  // REFILL_SYSTEM_MODELS_LINEARIZABLE_SYSTEM_MODEL_H_
