#include <gtest/gtest.h>
#include <Eigen/Dense>

#include "refill/filters/particle_filter.h"
#include "refill/measurement_models/linear_measurement_model.h"
#include "refill/system_models/linear_system_model.h"
#include "refill/utility/resample_methods.h"

namespace refill {

TEST(ParticleFilterTest, ConstructorTest) {
  GaussianDistribution initial_dist(1);
  initial_dist.rng_.seed(1);

  GaussianDistribution measurement_noise(1);
  GaussianDistribution system_noise(1);
  system_noise.rng_.seed(1);

  LinearMeasurementModel measurement_model(Eigen::MatrixXd::Identity(1, 1),
                                           measurement_noise);
  LinearSystemModel system_model(2 * Eigen::MatrixXd::Identity(1, 1),
                                 system_noise);

  Eigen::MatrixXd expected_initial_particles(1, 2);
  for (int i = 0; i < 2; ++i) {
    expected_initial_particles.col(i) = initial_dist.drawSample();
  }

  Eigen::VectorXd measurement(Eigen::VectorXd::Zero(1));

  Eigen::Vector2d expected_updated_weights(Eigen::Vector2d::Constant(0.5));

  Eigen::VectorXd likelihoods = measurement_model.getLikelihoodVectorized(
      expected_initial_particles, measurement);
  expected_updated_weights = expected_updated_weights.cwiseProduct(likelihoods);
  expected_updated_weights /= expected_updated_weights.sum();

  Eigen::MatrixXd expected_propagated_particles(expected_initial_particles);
  for (int i = 0; i < 2; ++i) {
    expected_propagated_particles.col(i) = 2
        * expected_propagated_particles.col(i)
        + system_model.getNoise()->drawSample();
  }

  ParticleFilter filter_1;

  Eigen::MatrixXd particles;
  Eigen::VectorXd weights;

  filter_1.getParticlesAndWeights(&particles, &weights);

  EXPECT_EQ(Eigen::MatrixXd::Zero(0, 0), particles);
  EXPECT_EQ(Eigen::VectorXd::Zero(0), weights);

  initial_dist.rng_.seed(1);
  ParticleFilter filter_2(2, &initial_dist);
  filter_2.getParticlesAndWeights(&particles, &weights);

  EXPECT_EQ(Eigen::Vector2d::Constant(0.5), weights);
  EXPECT_EQ(expected_initial_particles, particles);

  initial_dist.rng_.seed(1);
  ParticleFilter filter_3(2, &initial_dist, SamplingFunctorBase());

  filter_3.getParticlesAndWeights(&particles, &weights);

  EXPECT_EQ(Eigen::Vector2d::Constant(0.5), weights);
  EXPECT_EQ(expected_initial_particles, particles);

  filter_3.update(measurement_model, measurement);
  filter_3.getParticlesAndWeights(&particles, &weights);

  EXPECT_EQ(expected_updated_weights, weights);
  EXPECT_EQ(expected_initial_particles, particles);

  initial_dist.rng_.seed(1);
  system_model.getNoise()->rng_.seed(1);
  ParticleFilter filter_4(
      2,
      &initial_dist,
      SamplingFunctorBase(),
      std::unique_ptr<LinearSystemModel>(new LinearSystemModel(system_model)),
      std::unique_ptr<Likelihood>(
          new LinearMeasurementModel(measurement_model)));

  filter_4.getParticlesAndWeights(&particles, &weights);

  EXPECT_EQ(Eigen::Vector2d::Constant(0.5), weights);
  EXPECT_EQ(expected_initial_particles, particles);

  filter_4.predict();
  filter_4.getParticlesAndWeights(&particles, &weights);

  EXPECT_EQ(Eigen::Vector2d::Constant(0.5), weights);
  EXPECT_EQ(expected_propagated_particles, particles);

  filter_4.setParticles(expected_initial_particles);
  filter_4.update(measurement);

  filter_4.getParticlesAndWeights(&particles, &weights);

  EXPECT_EQ(expected_updated_weights, weights);
  EXPECT_EQ(expected_initial_particles, particles);
}

TEST(ParticleFilterTest, SetterTest) {
  GaussianDistribution initial_dist(1);
  initial_dist.rng_.seed(1);

  GaussianDistribution measurement_noise(1);
  GaussianDistribution system_noise(1);
  system_noise.rng_.seed(1);

  LinearMeasurementModel measurement_model(Eigen::MatrixXd::Identity(1, 1),
                                           measurement_noise);
  LinearSystemModel system_model(2 * Eigen::MatrixXd::Identity(1, 1),
                                 system_noise);

  Eigen::MatrixXd expected_initial_particles(1, 2);
  for (int i = 0; i < 2; ++i) {
    expected_initial_particles.col(i) = initial_dist.drawSample();
  }

  Eigen::VectorXd measurement(Eigen::VectorXd::Zero(1));

  Eigen::Vector2d expected_updated_weights(Eigen::Vector2d::Constant(0.5));

  Eigen::VectorXd likelihoods = measurement_model.getLikelihoodVectorized(
      expected_initial_particles, measurement);
  expected_updated_weights = expected_updated_weights.cwiseProduct(likelihoods);
  expected_updated_weights /= expected_updated_weights.sum();

  Eigen::MatrixXd expected_propagated_particles(expected_initial_particles);
  for (int i = 0; i < 2; ++i) {
    expected_propagated_particles.col(i) = 2
        * expected_propagated_particles.col(i)
        + system_model.getNoise()->drawSample();
  }

  ParticleFilter filter_1;

  initial_dist.rng_.seed(1);
  filter_1.setFilterParameters(2, &initial_dist);

  Eigen::VectorXd weights;
  Eigen::MatrixXd particles;

  filter_1.getParticlesAndWeights(&particles, &weights);

  EXPECT_EQ(Eigen::Vector2d::Constant(0.5), weights);
  EXPECT_EQ(expected_initial_particles, particles);

  ParticleFilter filter_2;

  initial_dist.rng_.seed(1);
  filter_2.setFilterParameters(2, &initial_dist, SamplingFunctorBase());

  filter_2.update(measurement_model, measurement);
  filter_2.getParticlesAndWeights(&particles, &weights);

  EXPECT_EQ(expected_updated_weights, weights);
  EXPECT_EQ(expected_initial_particles, particles);

  ParticleFilter filter_3;

  initial_dist.rng_.seed(1);
  system_model.getNoise()->rng_.seed(1);
  filter_3.setFilterParameters(
      2,
      &initial_dist,
      SamplingFunctorBase(),
      std::unique_ptr<LinearSystemModel>(new LinearSystemModel(system_model)),
      std::unique_ptr<Likelihood>(
          new LinearMeasurementModel(measurement_model)));

  filter_3.getParticlesAndWeights(&particles, &weights);

  EXPECT_EQ(Eigen::Vector2d::Constant(0.5), weights);
  EXPECT_EQ(expected_initial_particles, particles);

  filter_3.predict();

  filter_3.getParticlesAndWeights(&particles, &weights);

  EXPECT_EQ(Eigen::Vector2d::Constant(0.5), weights);
  EXPECT_EQ(expected_propagated_particles, particles);

  filter_3.setParticles(expected_initial_particles);
  filter_3.update(measurement);

  filter_3.getParticlesAndWeights(&particles, &weights);

  EXPECT_EQ(expected_updated_weights, weights);
  EXPECT_EQ(expected_initial_particles, particles);

  ParticleFilter filter_4(2, &initial_dist);
  filter_4.setParticles(Eigen::MatrixXd::Zero(1, 2));

  filter_4.getParticlesAndWeights(&particles, &weights);

  EXPECT_EQ(Eigen::Vector2d::Constant(0.5), weights);
  EXPECT_EQ(Eigen::MatrixXd::Zero(1, 2), particles);

  filter_4.setParticlesAndWeights(Eigen::MatrixXd::Constant(1, 2, 2),
                                  Eigen::Vector2d::Constant(0.5));

  filter_4.getParticlesAndWeights(&particles, &weights);

  EXPECT_EQ(Eigen::Vector2d::Constant(0.5), weights);
  EXPECT_EQ(Eigen::MatrixXd::Constant(1, 2, 2), particles);
}

}  // namespace refill
