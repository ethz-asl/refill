#include <gtest/gtest.h>
#include <Eigen/Dense>

// Pre-commit linter doesn't like chrono,
// #include <chrono>

#include "refill/filters/extended_kalman_filter.h"
#include "refill/system_models/linear_system_model.h"
#include "refill/measurement_models/linear_measurement_model.h"

namespace refill {

TEST(KalmanFilterTest, FullRun) {
  constexpr int loop_runs = 10000;

  // Testing speed with as much templating as possible
  constexpr int state_dim = 6;
  constexpr int meas_dim = 3;
  constexpr int input_dim = 1;

  GaussianDistribution<state_dim> init_state;
  GaussianDistribution<state_dim> sys_noise;
  GaussianDistribution<meas_dim> meas_noise;
  Eigen::Matrix<double, state_dim, state_dim> sys_mat;
  Eigen::Matrix<double, state_dim, input_dim> input_mat;
  Eigen::Matrix<double, meas_dim, state_dim> meas_mat;

  sys_mat = Eigen::Matrix<double, state_dim, state_dim>::Identity(state_dim,
                                                                  state_dim);
  input_mat = Eigen::Matrix<double, state_dim, input_dim>::Identity(state_dim,
                                                                    input_dim);
  meas_mat = Eigen::Matrix<double, meas_dim, state_dim>::Identity(meas_dim,
                                                                  state_dim);

  LinearSystemModel<state_dim, input_dim> sys_mod(sys_mat, sys_noise,
                                                  input_mat);
  LinearMeasurementModel<state_dim, meas_dim> meas_mod(meas_mat, meas_noise);

  ExtendedKalmanFilter<state_dim> kf(init_state);

  Eigen::Matrix<double, input_dim, 1> input;
  Eigen::Matrix<double, meas_dim, 1> measurement;

  input = Eigen::Matrix<double, input_dim, 1>::Ones(input_dim);
  measurement = Eigen::Matrix<double, meas_dim, 1>::Ones(meas_dim, 1);

//  std::chrono::time_point<std::chrono::system_clock> start, end;

//  start = std::chrono::system_clock::now();
  for (int i = 0; i < loop_runs; ++i) {
    kf.Predict(sys_mod, input);
    kf.Update(meas_mod, measurement);
  }
//  end = std::chrono::system_clock::now();

//  std::chrono::duration<double, std::milli> elapsed_ms_t = end - start;
//  std::cout << "Elapsed time with templating: " << elapsed_ms_t.count()
//            << " ms" << std::endl;

  // Testing speed with as little templating as possible
  int s_dim = state_dim;
  int m_dim = meas_dim;
  int i_dim = input_dim;

  GaussianDistributionXd initial_state(Eigen::VectorXd::Zero(s_dim),
                                       Eigen::MatrixXd::Identity(s_dim, s_dim));
  GaussianDistributionXd system_noise(Eigen::VectorXd::Zero(s_dim),
                                      Eigen::MatrixXd::Identity(s_dim, s_dim));
  GaussianDistributionXd measurement_noise(
      Eigen::VectorXd::Zero(m_dim), Eigen::MatrixXd::Identity(m_dim, m_dim));

  Eigen::MatrixXd system_mat;
  Eigen::MatrixXd in_mat;
  Eigen::MatrixXd measurement_mat;

  system_mat = Eigen::MatrixXd::Identity(s_dim, s_dim);
  in_mat = Eigen::MatrixXd::Identity(s_dim, i_dim);
  measurement_mat = Eigen::MatrixXd::Identity(m_dim, s_dim);

  LinearSystemModelXd system_mod(system_mat, system_noise, in_mat);
  LinearMeasurementModelXd measurement_mod(measurement_mat, measurement_noise);

  ExtendedKalmanFilterXd kfXd(initial_state);

  Eigen::VectorXd in;
  Eigen::VectorXd meas;

  in = Eigen::VectorXd::Ones(i_dim);
  meas = Eigen::VectorXd::Ones(m_dim);

//  start = std::chrono::system_clock::now();
  for (int i = 0; i < loop_runs; ++i) {
    kfXd.Predict(system_mod, in);
    kfXd.Update(measurement_mod, meas);
  }
//  end = std::chrono::system_clock::now();

//  std::chrono::duration<double, std::milli> elapsed_ms_nt = end - start;
//  std::cout << "Elapsed time without templating: " << elapsed_ms_nt.count()
//            << " ms" << std::endl;
//  std::cout << "Not templated took "
//            << elapsed_ms_nt.count() / elapsed_ms_t.count()
//            << " times as long as templated." << std::endl;
}

}  // namespace refill
