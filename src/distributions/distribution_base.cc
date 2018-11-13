#include "refill/distributions/distribution_base.h"

namespace refill {

/**
 * Assumes the samples where the pdf should be evaluated as the columns of @e x.
 *
 * @param sampled_x Samples at which the pdf should be evaluated.
 *          A @f$ N \times M @f$ matrix.
 * @return the pdfs values at the samples defined by x's column vectors.
 */
Eigen::VectorXd DistributionInterface::evaluatePdfVectorized(
    const Eigen::MatrixXd& sampled_x) const {
  Eigen::VectorXd pdf_values(sampled_x.cols());

  for (int i = 0; i < sampled_x.cols(); ++i) {
    pdf_values[i] = this->evaluatePdf(sampled_x.col(i));
  }

  return pdf_values;
}

void DistributionInterface::setRng(std::mt19937 rng) {
  rng_ = rng;
}

std::mt19937 DistributionInterface::getRng() const {
  return rng_;
}

}  // namespace refill

