#include <gtest/gtest.h>
#include <Eigen/Dense>

#include "refill/filters/extended_kalman_filter.h"
#include "refill/system_models/linear_system_model.h"
#include "refill/measurement_models/linear_measurement_model.h"

namespace refill {

TEST(ExtendedKalmanFilterTest, ConstructorTest) {
  ExtendedKalmanFilter filter_1;

  EXPECT_EQ(GaussianDistribution().mean(), filter_1.state().mean());
  EXPECT_EQ(GaussianDistribution().cov(), filter_1.state().cov());

  GaussianDistribution initial_state(Eigen::Vector2d::Zero(),
                                     Eigen::Matrix2d::Identity());

  ExtendedKalmanFilter filter_2(initial_state);

  EXPECT_EQ(Eigen::Vector2d::Zero(), filter_2.state().mean());
  EXPECT_EQ(Eigen::Matrix2d::Identity(), filter_2.state().cov());

  GaussianDistribution system_noise(Eigen::Vector2d::Zero(),
                                    Eigen::Matrix2d::Identity());
  GaussianDistribution measurement_noise(Eigen::Vector2d::Zero(),
                                         Eigen::Matrix2d::Identity() * 2.0);

  LinearSystemModel system_model(Eigen::Matrix2d::Identity(), system_noise);
  LinearMeasurementModel measurement_model(Eigen::Matrix2d::Identity(),
                                           measurement_noise);
  ExtendedKalmanFilter filter_3(
      initial_state, std::make_unique < LinearSystemModel > (system_model),
      std::make_unique < LinearMeasurementModel > (measurement_model));

  EXPECT_EQ(Eigen::Vector2d::Zero(), filter_3.state().mean());
  EXPECT_EQ(Eigen::Matrix2d::Identity(), filter_3.state().cov());

  filter_3.predict();

  EXPECT_EQ(Eigen::Vector2d::Zero(), filter_3.state().mean());
  EXPECT_EQ(Eigen::Matrix2d::Identity() * 2.0, filter_3.state().cov());

  filter_3.update(Eigen::Vector2d::Constant(3.0));

  EXPECT_EQ(Eigen::Vector2d::Constant(1.5), filter_3.state().mean());
  EXPECT_EQ(Eigen::Matrix2d::Identity(), filter_3.state().cov());
}

TEST(ExtendedKalmanFilter, SetterTest) {
  GaussianDistribution initial_state(Eigen::Vector2d::Zero(),
                                     Eigen::Matrix2d::Identity());

  ExtendedKalmanFilter filter_1(initial_state);

  initial_state.setDistributionParameters(Eigen::Vector2d::Ones(),
                             Eigen::Matrix2d::Identity() * 2.0);
  filter_1.setState(initial_state);

  EXPECT_EQ(Eigen::Vector2d::Ones(), filter_1.state().mean());
  EXPECT_EQ(Eigen::Matrix2d::Identity() * 2.0, filter_1.state().cov());
}

TEST(ExtendedKalmanFilterTest, PredictionTest) {
  GaussianDistribution initial_state(Eigen::Vector2d::Zero(),
                                     Eigen::Matrix2d::Identity());
  GaussianDistribution system_noise(Eigen::Vector2d::Zero(),
                                    Eigen::Matrix2d::Identity());
  GaussianDistribution measurement_noise(Eigen::Vector2d::Zero(),
                                         Eigen::Matrix2d::Identity());

  LinearSystemModel system_model(Eigen::Matrix2d::Identity(), system_noise,
                                 Eigen::Matrix2d::Identity());
  LinearMeasurementModel measurement_model(Eigen::Matrix2d::Identity(),
                                           measurement_noise);

  ExtendedKalmanFilter filter_1(
      initial_state, std::make_unique < LinearSystemModel > (system_model),
      std::make_unique < LinearMeasurementModel > (measurement_model));

  filter_1.predict();

  EXPECT_EQ(Eigen::Vector2d::Zero(), filter_1.state().mean());
  EXPECT_EQ(Eigen::Matrix2d::Identity() * 2.0, filter_1.state().cov());

  ExtendedKalmanFilter filter_2(
      initial_state, std::make_unique < LinearSystemModel > (system_model),
      std::make_unique < LinearMeasurementModel > (measurement_model));

  filter_2.predict(Eigen::Vector2d::Ones());

  EXPECT_EQ(Eigen::Vector2d::Ones(), filter_2.state().mean());
  EXPECT_EQ(Eigen::Matrix2d::Identity() * 2.0, filter_2.state().cov());

  ExtendedKalmanFilter filter_3(initial_state);

  filter_3.predict(system_model);

  EXPECT_EQ(Eigen::Vector2d::Zero(), filter_3.state().mean());
  EXPECT_EQ(Eigen::Matrix2d::Identity() * 2.0, filter_3.state().cov());

  ExtendedKalmanFilter filter_4(initial_state);

  filter_4.predict(system_model, Eigen::Vector2d::Ones());

  EXPECT_EQ(Eigen::Vector2d::Ones(), filter_4.state().mean());
  EXPECT_EQ(Eigen::Matrix2d::Identity() * 2.0, filter_4.state().cov());
}

TEST(ExtendedKalmanFilterTest, FullRun) {
  constexpr int kStateDim = 3;
  constexpr int kMeasurementDim = 3;
  constexpr int kInputDim = 3;

  // Set up initial state, as well as system and measurement noise
  GaussianDistribution initial_state(kStateDim);
  GaussianDistribution system_noise(kStateDim);
  GaussianDistribution measurement_noiseXd(kMeasurementDim);

  // Set up system, input and measurement matrix
  const Eigen::MatrixXd system_mat = Eigen::MatrixXd::Identity(kStateDim,
                                                               kStateDim);
  const Eigen::MatrixXd input_mat = Eigen::MatrixXd::Identity(kStateDim,
                                                              kInputDim);
  const Eigen::MatrixXd measurement_mat = Eigen::MatrixXd::Identity(
      kMeasurementDim, kStateDim);

  // Set up linear system and measurement model
  LinearSystemModel system_model(system_mat, system_noise, input_mat);
  LinearMeasurementModel measurement_model(measurement_mat,
                                           measurement_noiseXd);

  // Set up kalman filter
  ExtendedKalmanFilter KF(initial_state);

  // Set up input and measurement vectors
  const Eigen::VectorXd input = Eigen::VectorXd::Ones(kInputDim);
  const Eigen::VectorXd measurement = Eigen::VectorXd::Ones(kMeasurementDim);

  // Perform one prediction step
  KF.predict(system_model, input);

  // Check if prediction was correct
  ASSERT_EQ(KF.state().mean(), Eigen::VectorXd::Ones(kStateDim));
  ASSERT_EQ(KF.state().cov(),
            Eigen::MatrixXd::Identity(kStateDim, kStateDim) * 2.0);

  // Perform one update step
  KF.update(measurement_model, measurement);

  // Check if update was correct
  ASSERT_EQ(KF.state().mean(), Eigen::VectorXd::Ones(kStateDim));

  // TODO(jwidauer): Add more test cases
}

}  // namespace refill
