#ifndef REFILL_SYSTEM_MODELS_LINEARIZED_SYSTEM_MODEL_H_
#define REFILL_SYSTEM_MODELS_LINEARIZED_SYSTEM_MODEL_H_

#include <Eigen/Dense>

#include <cstdlib>

#include "refill/distributions/distribution_base.h"
#include "refill/system_models/system_model_base.h"

using std::size_t;

namespace refill {

class LinearizedSystemModel : public SystemModelBase {
 public:
  virtual Eigen::MatrixXd getStateJacobian(
      const Eigen::VectorXd& state, const Eigen::VectorXd& input) const = 0;
  virtual Eigen::MatrixXd getNoiseJacobian(
      const Eigen::VectorXd& state, const Eigen::VectorXd& input) const = 0;

 protected:
  LinearizedSystemModel() = delete;
  LinearizedSystemModel(const size_t& state_dim,
                        const DistributionInterface& system_noise);
  LinearizedSystemModel(const size_t& state_dim,
                        const DistributionInterface& system_noise,
                        const size_t& input_dim);
};

}  // namespace refill

#endif  // REFILL_SYSTEM_MODELS_LINEARIZED_SYSTEM_MODEL_H_
