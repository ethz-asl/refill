#ifndef REFILL_SYSTEM_MODELS_LINEAR_SYSTEM_MODEL_H_
#define REFILL_SYSTEM_MODELS_LINEAR_SYSTEM_MODEL_H_

#include <glog/logging.h>
#include <Eigen/Dense>

#include "refill/distributions/gaussian_distribution.h"
#include "refill/system_models/linearized_system_model.h"

namespace refill {

// Class for a system model of the form:
//   x(k+1) = A * x(k) + B * u(k) + L * v(k)
// Where x(k) is the state, u(k) the input and v(k) the noise at timestep k.
class LinearSystemModel : public LinearizedSystemModel {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  LinearSystemModel();
  LinearSystemModel(const Eigen::MatrixXd& system_mapping,
                    const DistributionInterface& system_noise);
  LinearSystemModel(const Eigen::MatrixXd& system_mapping,
                    const DistributionInterface& system_noise,
                    const Eigen::MatrixXd& input_mapping);
  LinearSystemModel(const Eigen::MatrixXd& system_mapping,
                    const DistributionInterface& system_noise,
                    const Eigen::MatrixXd& input_mapping,
                    const Eigen::MatrixXd& noise_mapping);

  void setSystemParameters(const Eigen::MatrixXd& system_mapping,
                           const DistributionInterface& system_noise);
  void setSystemParameters(const Eigen::MatrixXd& system_mapping,
                           const DistributionInterface& system_noise,
                           const Eigen::MatrixXd& input_mapping);
  void setSystemParameters(const Eigen::MatrixXd& system_mapping,
                           const DistributionInterface& system_noise,
                           const Eigen::MatrixXd& input_mapping,
                           const Eigen::MatrixXd& noise_mapping);

  // Propagate a state vector through the linear system model
  Eigen::VectorXd propagate(const Eigen::VectorXd& state) const;
  Eigen::VectorXd propagate(const Eigen::VectorXd& state,
                            const Eigen::VectorXd& input) const;

  int getStateDim() const;
  int getInputDim() const;
  Eigen::MatrixXd getStateJacobian(const Eigen::VectorXd& state,
                                   const Eigen::VectorXd& input) const;
  Eigen::MatrixXd getNoiseJacobian(const Eigen::VectorXd& state,
                                   const Eigen::VectorXd& input) const;
  DistributionInterface* getSystemNoise() const;

 private:
  Eigen::MatrixXd system_mapping_;
  Eigen::MatrixXd input_mapping_;
  Eigen::MatrixXd noise_mapping_;
};

}  // namespace refill

#endif  // REFILL_SYSTEM_MODELS_LINEAR_SYSTEM_MODEL_H_
