#ifndef REFILL_SYSTEM_MODELS_LINEAR_SYSTEM_MODEL_H_
#define REFILL_SYSTEM_MODELS_LINEAR_SYSTEM_MODEL_H_

#include <glog/logging.h>
#include <memory>

#include "refill/distributions/gaussian_distribution.h"
#include "refill/system_models/system_model_base.h"

namespace refill {

class LinearSystemModel : public SystemModelBase {
 public:
  LinearSystemModel();
  LinearSystemModel(const Eigen::MatrixXd& system_mat,
                    const DistributionInterface& system_noise);
  LinearSystemModel(const Eigen::MatrixXd& system_mat,
                    const DistributionInterface& system_noise,
                    const Eigen::MatrixXd& input_mat);

  void setSystemParameters(const Eigen::MatrixXd& system_mat,
                           const DistributionInterface& system_noise);
  void setSystemParameters(const Eigen::MatrixXd& system_mat,
                           const DistributionInterface& system_noise,
                           const Eigen::MatrixXd& input_mat);

  // Propagate a state vector through the linear system model
  Eigen::VectorXd propagate(const Eigen::VectorXd& state) const;
  Eigen::VectorXd propagate(const Eigen::VectorXd& state,
                            const Eigen::VectorXd& input) const;

  int getStateDim() const;
  int getInputDim() const;
  Eigen::MatrixXd getJacobian() const;
  DistributionInterface* getSystemNoise() const;

 private:
  Eigen::MatrixXd system_mapping_;
  Eigen::MatrixXd input_mapping_;
  std::unique_ptr<DistributionInterface> system_noise_;
};

}  // namespace refill

#endif  // REFILL_SYSTEM_MODELS_LINEAR_SYSTEM_MODEL_H_
