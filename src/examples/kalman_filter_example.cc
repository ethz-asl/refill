#include <Eigen/Dense>

#include <iostream>

#include "refill/distributions/gaussian_distribution.h"
#include "refill/filters/extended_kalman_filter.h"
#include "refill/measurement_models/linear_measurement_model.h"
#include "refill/system_models/linear_system_model.h"

/*
 * This is an example program for using Refill to estimate the 3D position
 * using a constant position model and assuming measurements are 3D position
 * measurements of the real position.
 *
 * The system model can then be written as:
 *
 * x(k) = I * x(k-1) + v(k)
 *
 * with
 *
 * x(k) element of R^3
 *
 * I = Identity matrix element of R^3x3
 *
 * v(k) random variable distributed with N(0, dt * Q)
 *
 * where Q element of R^3x3 is the system model covariance.
 *
 * The measurement model can be written as:
 *
 * y(k) = I * x(k) + w(k)
 *
 * with
 *
 * w(k) ~ N(0, R)
 *
 * where R element of R^3x3 is the measurement model covariance.
 */

int main(int argc, char **argv) {
  /* initialize Q and R
   * Q = I * 2.0
   * R = I */
  Eigen::Matrix3d system_noise_cov = Eigen::Matrix3d::Identity() * 2.0;
  Eigen::Matrix3d measurement_noise_cov = Eigen::Matrix3d::Identity() * 1.0;

  /* initialize v(k) */
  refill::GaussianDistribution system_noise(Eigen::Vector3d::Zero(),
                                            system_noise_cov);
  /* initialize w(k) */
  refill::GaussianDistribution measurement_noise(
      Eigen::Vector3d::Zero(), measurement_noise_cov);

  /* initialize the system model */
  refill::LinearSystemModel system_model(Eigen::Matrix3d::Identity(),
                                         system_noise);
  /* initialize the measurement model */
  refill::LinearMeasurementModel measurement_model(Eigen::Matrix3d::Identity(),
                                                   measurement_noise);

  /* initialize the initial state distribution
   * Assumed to be at position [1, 1, 1]^T with cov[I * 5] */
  refill::GaussianDistribution initial_state(Eigen::Vector3d::Ones(),
                                             Eigen::Matrix3d::Identity() * 5.0);
  /* initialize the kf with the initial state */
  refill::ExtendedKalmanFilter ekf(initial_state);

  /* assume that 1.0 seconds has passed and we get a position measurement at
   * [1.5, 1.5, 1.5]^T
   * t = 1.0 */
  double dt = 1.0;
  Eigen::Vector3d measurement = Eigen::Vector3d::Constant(1.5);

  /* adapt the system model noise according to the time step */
  system_noise.setCov(system_noise_cov * dt);
  /* adapt the system model */
  system_model.setModelParameters(Eigen::Matrix3d::Identity(), system_noise);

  /* predict the kf to the current time */
  ekf.predict(system_model);
  /* update the kf with the measurement and the measurement model */
  ekf.update(measurement_model, measurement);

  /* print the current state */
  std::cout << "State at t = 1.0:\n";
  std::cout << "Mean:\n\n" << ekf.state().mean() << "\n\n";
  std::cout << "Covariance:\n\n" << ekf.state().cov() << "\n\n";

  /* Assume that another 0.5 seconds have passed and another measurement
   * is received
   * t = 1.5 */
  dt = 0.5;
  measurement = Eigen::Vector3d::Constant(1.5);

  // adapt the system noise according to the time step
  system_noise.setCov(system_noise_cov * dt);
  // adapt the system model
  system_model.setModelParameters(Eigen::Matrix3d::Identity(), system_noise);

  /* predict the kf to the current time */
  ekf.predict(system_model);
  /* update the kf with the measurement and the measurement model */
  ekf.update(measurement_model, measurement);

  /* print the current state */
  std::cout << "State at t = 1.5:\n";
  std::cout << "Mean:\n\n" << ekf.state().mean() << "\n\n";
  std::cout << "Covariance:\n\n" << ekf.state().cov() << "\n\n";

  return 0;
}
