#ifndef REFILL_SYSTEM_MODELS_SYSTEM_MODEL_BASE_H_
#define REFILL_SYSTEM_MODELS_SYSTEM_MODEL_BASE_H_

#include <Eigen/Dense>

#include "refill/distributions/distribution_base.h"

namespace refill {

class SystemModelBase {
 public:
  virtual Eigen::VectorXd propagate(const Eigen::VectorXd& state,
                                    const Eigen::VectorXd& input) const = 0;

  // TODO(igilitschenski): Remove above interface.
  virtual Eigen::VectorXd propagate(const Eigen::VectorXd& state,
                                    const Eigen::VectorXd& input,
                                    const Eigen::VectorXd& noise) {
    return Eigen::VectorXd::Zero(3);
  }

  // Performs propagate on each column of state. Can be reimplemented in a
  // subclass for improving performance of sample based filters.
  virtual Eigen::MatrixXd propagateVectorized(
      const Eigen::MatrixXd& sampled_state, const Eigen::VectorXd& input,
      const Eigen::MatrixXd& sampled_noise);

  virtual int getStateDim() const = 0;
  virtual int getInputDim() const = 0;
  virtual int getSystemNoiseDim() const = 0;
  virtual DistributionInterface* getSystemNoise() const = 0;
};

}  // namespace refill

#endif  // REFILL_SYSTEM_MODELS_SYSTEM_MODEL_BASE_H_
