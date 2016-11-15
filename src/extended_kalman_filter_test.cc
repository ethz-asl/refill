#include <gtest/gtest.h>
#include <Eigen/Dense>

#include "refill/filters/extended_kalman_filter.h"
#include "refill/system_models/linear_system_model.h"
#include "refill/measurement_models/linear_measurement_model.h"

namespace refill {

TEST(KalmanFilterTest, FullRun) {
  // Tests with templated size
  constexpr int kStateDim = 3;
  constexpr int kMeasurementDim = 3;
  constexpr int kInputDim = 3;

  // Set up distributions
  GaussianDistribution<kStateDim> initial_state;
  GaussianDistribution<kStateDim> system_noise;
  GaussianDistribution<kMeasurementDim> measurement_noise;

  // Set up syste, input and measurement matrix
  Eigen::Matrix<double, kStateDim, kStateDim> system_mat;
  Eigen::Matrix<double, kStateDim, kInputDim> input_mat;
  Eigen::Matrix<double, kMeasurementDim, kStateDim> measurement_mat;

  system_mat = Eigen::Matrix<double, kStateDim, kStateDim>::Identity(kStateDim,
                                                                     kStateDim);
  input_mat = Eigen::Matrix<double, kStateDim, kInputDim>::Identity(kStateDim,
                                                                    kInputDim);
  measurement_mat = Eigen::Matrix<double, kMeasurementDim, kStateDim>::Identity(
      kMeasurementDim, kStateDim);

  // Set up system and measurement model
  LinearSystemModel<kStateDim, kInputDim> system_model(system_mat, system_noise,
                                                       input_mat);
  LinearMeasurementModel<kStateDim, kMeasurementDim> measurement_model(
      measurement_mat, measurement_noise);

  // Set up kalman filter
  ExtendedKalmanFilter<kStateDim> KF(initial_state);

  // Set up input and measurement vector
  Eigen::Matrix<double, kInputDim, 1> input;
  Eigen::Matrix<double, kMeasurementDim, 1> measurement;

  input = Eigen::Matrix<double, kInputDim, 1>::Ones(kInputDim);
  measurement = Eigen::Matrix<double, kMeasurementDim, 1>::Ones(kMeasurementDim,
                                                                1);

  // Predict one step
  KF.Predict(system_model, input);

  ASSERT_EQ(KF.state().mean(), Eigen::VectorXd::Ones(kStateDim));
  ASSERT_EQ(KF.state().cov(),
            Eigen::MatrixXd::Identity(kStateDim, kStateDim) * 2.0);

  // Update with measurement
  KF.Update(measurement_model, measurement);

  ASSERT_EQ(KF.state().mean(), Eigen::VectorXd::Ones(kStateDim));

  // Tests with dynamic size
  GaussianDistributionXd initial_stateXd(
      Eigen::VectorXd::Zero(kStateDim),
      Eigen::MatrixXd::Identity(kStateDim, kStateDim));
  GaussianDistributionXd system_noiseXd(
      Eigen::VectorXd::Zero(kStateDim),
      Eigen::MatrixXd::Identity(kStateDim, kStateDim));
  GaussianDistributionXd measurement_noiseXd(
      Eigen::VectorXd::Zero(kMeasurementDim),
      Eigen::MatrixXd::Identity(kMeasurementDim, kMeasurementDim));

  Eigen::MatrixXd system_matXd;
  Eigen::MatrixXd input_matXd;
  Eigen::MatrixXd measurement_matXd;

  system_matXd = Eigen::MatrixXd::Identity(kStateDim, kStateDim);
  input_matXd = Eigen::MatrixXd::Identity(kStateDim, kInputDim);
  measurement_matXd = Eigen::MatrixXd::Identity(kMeasurementDim, kStateDim);

  LinearSystemModelXd system_modelXd(system_matXd, system_noiseXd, input_matXd);
  LinearMeasurementModelXd measurement_modelXd(measurement_matXd,
                                               measurement_noiseXd);

  ExtendedKalmanFilterXd KFXd(initial_stateXd);

  Eigen::VectorXd inputXd;
  Eigen::VectorXd measurementXd;

  inputXd = Eigen::VectorXd::Ones(kInputDim);
  measurementXd = Eigen::VectorXd::Ones(kMeasurementDim);

  KFXd.Predict(system_modelXd, inputXd);

  ASSERT_EQ(KFXd.state().mean(), Eigen::VectorXd::Ones(kStateDim));
  ASSERT_EQ(KFXd.state().cov(),
            Eigen::MatrixXd::Identity(kStateDim, kStateDim) * 2.0);

  KFXd.Update(measurement_modelXd, measurementXd);

  ASSERT_EQ(KFXd.state().mean(), Eigen::VectorXd::Ones(kStateDim));

//  KFXd.Update(measurement_modelXd, measurement);
}

}  // namespace refill
