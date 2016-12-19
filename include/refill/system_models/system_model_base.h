#ifndef REFILL_SYSTEM_MODELS_SYSTEM_MODEL_BASE_H_
#define REFILL_SYSTEM_MODELS_SYSTEM_MODEL_BASE_H_

#include <glog/logging.h>

#include <Eigen/Dense>
#include <cstdlib>
#include <memory>

#include "refill/distributions/distribution_base.h"

using std::size_t;

namespace refill {

class SystemModelBase {
 public:
  SystemModelBase() = delete;
  virtual Eigen::VectorXd propagate(const Eigen::VectorXd& state,
                                    const Eigen::VectorXd& input,
                                    const Eigen::VectorXd& noise) const = 0;

  // A vectorized version of the propagation.
  // sampled_state is a N_s x M_s Matrix
  // input is the intput to the system function.
  // sampled_noise is a N_n x M_n Matrix
  // Returns an N_s x (M_s * M_n) Matrix
  // N_s / N_n denotes state and noise dimensions respectively.
  // M_s / M_n user defined number of state/noise samples.
  virtual Eigen::MatrixXd propagateVectorized(
      const Eigen::MatrixXd& sampled_state, const Eigen::VectorXd& input,
      const Eigen::MatrixXd& sampled_noise) const;

  size_t getStateDim() const;
  size_t getInputDim() const;
  size_t getSystemNoiseDim() const;
  DistributionInterface* getSystemNoise() const;

 protected:
  SystemModelBase(const size_t& state_dim,
                  const DistributionInterface& system_noise);
  SystemModelBase(const size_t& state_dim,
                  const DistributionInterface& system_noise,
                  const size_t& input_dim);

  void setSystemModelBaseParameters(const size_t& state_dim,
                                    const DistributionInterface& system_noise);
  void setSystemModelBaseParameters(const size_t& state_dim,
                                    const DistributionInterface& system_noise,
                                    const size_t& input_dim);

 private:
  size_t state_dim_;
  size_t input_dim_;
  std::unique_ptr<DistributionInterface> system_noise_;
};

}  // namespace refill

#endif  // REFILL_SYSTEM_MODELS_SYSTEM_MODEL_BASE_H_
