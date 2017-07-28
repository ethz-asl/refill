#include "refill/distributions/likelihood.h"

namespace refill {

Eigen::VectorXd Likelihood::getLikelihoodVectorized(
    const Eigen::MatrixXd& sampled_state,
    const Eigen::VectorXd& measurement) const {
  Eigen::VectorXd likelihoods(sampled_state.cols());

  for (int i = 0; i < sampled_state.cols(); ++i) {
    likelihoods[i] = this->getLikelihood(sampled_state.col(i), measurement);
  }

  return likelihoods;
}

}  // namespace refill
