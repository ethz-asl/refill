#include "refill/distributions/distribution_base.h"

namespace refill {

/**
 * Assumes the points where the pdf should be evaluated as the columns of @e x.
 *
 * @param x Points at which the pdf should be evaluated.
 *          A @f$ N \times M @f$ matrix.
 * @return the pdfs values at the points defined by x's column vectors.
 */
Eigen::VectorXd DistributionInterface::evaluatePdfVectorized(
    const Eigen::MatrixXd& sampled_x) const {
  Eigen::VectorXd pdf_values(sampled_x.cols());

  for (int i = 0; i < sampled_x.cols(); ++i) {
    pdf_values[i] = this->evaluatePdf(sampled_x.col(i));
  }

  return pdf_values;
}

}  // namespace refill

