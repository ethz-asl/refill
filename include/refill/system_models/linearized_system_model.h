#ifndef REFILL_SYSTEM_MODELS_LINEARIZED_SYSTEM_MODEL_H_
#define REFILL_SYSTEM_MODELS_LINEARIZED_SYSTEM_MODEL_H_

#include <Eigen/Dense>

#include "refill/distributions/distribution_base.h"
#include "refill/system_models/system_model_base.h"

namespace refill {

/**
 * @brief Class that implements a linearized system model.
 *
 * This class is an interface for system models with Jacobians.
 *
 * Its intended purpose is to implement system models of the form:
 *
 * \f$ x_{k+1} = f(x_k, \nu_k)\f$
 */
class LinearizedSystemModel : public SystemModelBase {
 public:
  virtual Eigen::MatrixXd getStateJacobian(
      const Eigen::VectorXd& state, const Eigen::VectorXd& input) const = 0;
  virtual Eigen::MatrixXd getNoiseJacobian(
      const Eigen::VectorXd& state, const Eigen::VectorXd& input) const = 0;

 protected:
  /** @brief Default constructor should not be used. */
  LinearizedSystemModel() = delete;

  /** @brief Constructor for a linearized system model without an input. */
  LinearizedSystemModel(const std::size_t& state_dim,
                        const DistributionInterface& system_noise);

  /** @brief Constructor for a linearized system model with an input. */
  LinearizedSystemModel(const std::size_t& state_dim,
                        const DistributionInterface& system_noise,
                        const std::size_t& input_dim);
};

}  // namespace refill

#endif  // REFILL_SYSTEM_MODELS_LINEARIZED_SYSTEM_MODEL_H_
