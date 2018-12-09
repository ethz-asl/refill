#include <gtest/gtest.h>
#include <Eigen/Dense>

#include "refill/filters/particle_filter.h"
#include "refill/measurement_models/linear_measurement_model.h"
#include "refill/system_models/linear_system_model.h"
#include "refill/utility/resample_methods.h"

namespace refill {

class ParticleFilterTest : public ::testing::Test {
 public:
  ParticleFilterTest()
      : initial_dist_(1),
        system_noise_(1),
        measurement_noise_(1),
        system_model_(2 * Eigen::MatrixXd::Identity(1, 1),
                      system_noise_, Eigen::MatrixXd::Identity(1, 1)),
        measurement_model_(Eigen::MatrixXd::Identity(1, 1),
                           measurement_noise_),
        expected_initial_particles_(1, 2),
        expected_propagated_particles_(1, 2),
        expected_propagated_particles_with_input_(1, 2),
        expected_updated_weights_(Eigen::Vector2d::Constant(0.5)),
        input_(Eigen::VectorXd::Constant(1, 1.0)),
        measurement_(Eigen::VectorXd::Zero(1)) {
    this->ResetRngs();

    for (int i = 0; i < 2; ++i) {
      expected_initial_particles_.col(i) = initial_dist_.drawSample();
    }

    Eigen::VectorXd likelihoods = measurement_model_.getLikelihoodVectorized(
        expected_initial_particles_, measurement_);
    expected_updated_weights_ =
        expected_updated_weights_.cwiseProduct(likelihoods);
    expected_updated_weights_ /= expected_updated_weights_.sum();

    for (int i = 0; i < 2; ++i) {
      expected_propagated_particles_.col(i) = 2
          * expected_initial_particles_.col(i)
          + system_model_.getNoise()->drawSample();
      expected_propagated_particles_with_input_.col(i) =
          expected_propagated_particles_.col(i) + input_;
    }

    this->ResetRngs();
  }

  void ResetRngs() {
    std::mt19937 rng(1);
    initial_dist_.setRng(rng);
    system_noise_.setRng(rng);
    measurement_noise_.setRng(rng);

    system_model_.getNoise()->setRng(rng);
    measurement_model_.getNoise()->setRng(rng);
  }

  GaussianDistribution initial_dist_;
  GaussianDistribution system_noise_;
  GaussianDistribution measurement_noise_;

  LinearSystemModel system_model_;
  LinearMeasurementModel measurement_model_;

  Eigen::MatrixXd expected_initial_particles_;
  Eigen::MatrixXd expected_propagated_particles_;
  Eigen::MatrixXd expected_propagated_particles_with_input_;
  Eigen::Vector2d expected_updated_weights_;

  Eigen::VectorXd input_;
  Eigen::VectorXd measurement_;
};

TEST_F(ParticleFilterTest, DefaultConstructorTest) {
  ParticleFilter filter;

  Eigen::MatrixXd particles;
  Eigen::VectorXd weights;

  filter.getParticlesAndWeights(&particles, &weights);

  EXPECT_EQ(Eigen::MatrixXd::Zero(0, 0), particles);
  EXPECT_EQ(Eigen::VectorXd::Zero(0), weights);
}

TEST_F(ParticleFilterTest, TwoArgumentsConstructorTest) {
  ParticleFilter filter(2, &initial_dist_);

  Eigen::MatrixXd particles;
  Eigen::VectorXd weights;

  filter.getParticlesAndWeights(&particles, &weights);

  EXPECT_EQ(Eigen::Vector2d::Constant(0.5), weights);
  EXPECT_EQ(expected_initial_particles_, particles);
}

TEST_F(ParticleFilterTest, ThreeArgumentConstructorTest) {
  ParticleFilter filter(2, &initial_dist_, SamplingFunctorBase());

  Eigen::MatrixXd particles;
  Eigen::VectorXd weights;

  filter.getParticlesAndWeights(&particles, &weights);

  EXPECT_EQ(Eigen::Vector2d::Constant(0.5), weights);
  EXPECT_EQ(expected_initial_particles_, particles);

  filter.update(measurement_model_, measurement_);
  filter.getParticlesAndWeights(&particles, &weights);

  EXPECT_EQ(expected_updated_weights_, weights);
  EXPECT_EQ(expected_initial_particles_, particles);
}

TEST_F(ParticleFilterTest, FiveArgumentsConstructorTest) {
  ParticleFilter filter(
      2,
      &initial_dist_,
      SamplingFunctorBase(),
      std::unique_ptr<LinearSystemModel>(new LinearSystemModel(system_model_)),
      std::unique_ptr<Likelihood>(
          new LinearMeasurementModel(measurement_model_)));

  Eigen::MatrixXd particles;
  Eigen::VectorXd weights;

  filter.getParticlesAndWeights(&particles, &weights);

  EXPECT_EQ(Eigen::Vector2d::Constant(0.5), weights);
  EXPECT_EQ(expected_initial_particles_, particles);

  filter.predict();
  filter.getParticlesAndWeights(&particles, &weights);

  EXPECT_EQ(Eigen::Vector2d::Constant(0.5), weights);
  EXPECT_EQ(expected_propagated_particles_, particles);

  filter.setParticles(expected_initial_particles_);
  filter.update(measurement_);

  filter.getParticlesAndWeights(&particles, &weights);

  EXPECT_EQ(expected_updated_weights_, weights);
  EXPECT_EQ(expected_initial_particles_, particles);
}

TEST_F(ParticleFilterTest, TwoArgumentsParameterSetterTest) {
  ParticleFilter filter;
  filter.setFilterParameters(2, &initial_dist_);

  Eigen::VectorXd weights;
  Eigen::MatrixXd particles;

  filter.getParticlesAndWeights(&particles, &weights);

  EXPECT_EQ(Eigen::Vector2d::Constant(0.5), weights);
  EXPECT_EQ(expected_initial_particles_, particles);
}

TEST_F(ParticleFilterTest, ThreeArgumentsParameterSetterTest) {
  ParticleFilter filter;
  filter.setFilterParameters(2, &initial_dist_, SamplingFunctorBase());

  Eigen::MatrixXd particles;
  Eigen::VectorXd weights;

  filter.update(measurement_model_, measurement_);
  filter.getParticlesAndWeights(&particles, &weights);

  EXPECT_EQ(expected_updated_weights_, weights);
  EXPECT_EQ(expected_initial_particles_, particles);
}

TEST_F(ParticleFilterTest, FiveArgumentsParameterSetterTest) {
  ParticleFilter filter;
  filter.setFilterParameters(
      2,
      &initial_dist_,
      SamplingFunctorBase(),
      std::unique_ptr<LinearSystemModel>(new LinearSystemModel(system_model_)),
      std::unique_ptr<Likelihood>(
          new LinearMeasurementModel(measurement_model_)));

  Eigen::MatrixXd particles;
  Eigen::VectorXd weights;

  filter.getParticlesAndWeights(&particles, &weights);

  EXPECT_EQ(Eigen::Vector2d::Constant(0.5), weights);
  EXPECT_EQ(expected_initial_particles_, particles);

  filter.predict();

  filter.getParticlesAndWeights(&particles, &weights);

  EXPECT_EQ(Eigen::Vector2d::Constant(0.5), weights);
  EXPECT_EQ(expected_propagated_particles_, particles);

  filter.setParticles(expected_initial_particles_);
  filter.update(measurement_);

  filter.getParticlesAndWeights(&particles, &weights);

  EXPECT_EQ(expected_updated_weights_, weights);
  EXPECT_EQ(expected_initial_particles_, particles);
}

TEST_F(ParticleFilterTest, SetParticlesTest) {
  ParticleFilter filter(2, &initial_dist_);
  filter.setParticles(Eigen::MatrixXd::Zero(1, 2));

  Eigen::MatrixXd particles;
  Eigen::VectorXd weights;

  filter.getParticlesAndWeights(&particles, &weights);

  EXPECT_EQ(Eigen::Vector2d::Constant(0.5), weights);
  EXPECT_EQ(Eigen::MatrixXd::Zero(1, 2), particles);
}

TEST_F(ParticleFilterTest, SetParticlesAndWeightsTest) {
  ParticleFilter filter(2, &initial_dist_);
  filter.setParticlesAndWeights(Eigen::MatrixXd::Constant(1, 2, 2),
                                Eigen::Vector2d::Constant(0.5));

  Eigen::MatrixXd particles;
  Eigen::VectorXd weights;

  filter.getParticlesAndWeights(&particles, &weights);

  EXPECT_EQ(Eigen::Vector2d::Constant(0.5), weights);
  EXPECT_EQ(Eigen::MatrixXd::Constant(1, 2, 2), particles);
}

TEST_F(ParticleFilterTest, ReinitializeParticlesTest) {
  ParticleFilter filter(2, &initial_dist_);
  filter.setParticlesAndWeights(Eigen::MatrixXd::Constant(1, 2, 2),
                                Eigen::Vector2d::Constant(0.5));

  ResetRngs();
  filter.reinitializeParticles(&initial_dist_);

  Eigen::MatrixXd particles;
  Eigen::VectorXd weights;

  filter.getParticlesAndWeights(&particles, &weights);

  EXPECT_EQ(Eigen::Vector2d::Constant(0.5), weights);
  EXPECT_EQ(expected_initial_particles_, particles);
}

TEST_F(ParticleFilterTest, DefaultPredictorTest) {
  ParticleFilter filter(
      2,
      &initial_dist_,
      SamplingFunctorBase(),
      std::unique_ptr<LinearSystemModel>(new LinearSystemModel(system_model_)),
      std::unique_ptr<Likelihood>(
          new LinearMeasurementModel(measurement_model_)));

  filter.predict();

  Eigen::MatrixXd particles;
  Eigen::VectorXd weights;

  filter.getParticlesAndWeights(&particles, &weights);

  EXPECT_EQ(Eigen::Vector2d::Constant(0.5), weights);
  EXPECT_EQ(expected_propagated_particles_, particles);
}

TEST_F(ParticleFilterTest, DefaultPredictorWithInputTest) {
  ParticleFilter filter(
      2,
      &initial_dist_,
      SamplingFunctorBase(),
      std::unique_ptr<LinearSystemModel>(new LinearSystemModel(system_model_)),
      std::unique_ptr<Likelihood>(
          new LinearMeasurementModel(measurement_model_)));

  filter.predict(input_);

  Eigen::MatrixXd particles;
  Eigen::VectorXd weights;

  filter.getParticlesAndWeights(&particles, &weights);

  EXPECT_EQ(Eigen::Vector2d::Constant(0.5), weights);
  EXPECT_EQ(expected_propagated_particles_with_input_, particles);
}

TEST_F(ParticleFilterTest, SystemModelPredictionTest) {
  ParticleFilter filter(2, &initial_dist_);
  filter.predict(system_model_);

  Eigen::MatrixXd particles;
  Eigen::VectorXd weights;

  filter.getParticlesAndWeights(&particles, &weights);

  EXPECT_EQ(Eigen::Vector2d::Constant(0.5), weights);
  EXPECT_EQ(expected_propagated_particles_, particles);
}

TEST_F(ParticleFilterTest, SystemModelWithInputPredictionTest) {
  ParticleFilter filter(2, &initial_dist_);
  filter.predict(system_model_, input_);

  Eigen::MatrixXd particles;
  Eigen::VectorXd weights;

  filter.getParticlesAndWeights(&particles, &weights);

  EXPECT_EQ(Eigen::Vector2d::Constant(0.5), weights);
  EXPECT_EQ(expected_propagated_particles_with_input_, particles);
}

TEST_F(ParticleFilterTest, DefaultUpdateTest) {
  ParticleFilter filter(
      2,
      &initial_dist_,
      SamplingFunctorBase(),
      std::unique_ptr<LinearSystemModel>(new LinearSystemModel(system_model_)),
      std::unique_ptr<Likelihood>(
          new LinearMeasurementModel(measurement_model_)));
  filter.update(measurement_);

  Eigen::MatrixXd particles;
  Eigen::VectorXd weights;

  filter.getParticlesAndWeights(&particles, &weights);

  EXPECT_EQ(expected_updated_weights_, weights);
  EXPECT_EQ(expected_initial_particles_, particles);
}

TEST_F(ParticleFilterTest, MeasurementModelUpdateTest) {
  ParticleFilter filter(2, &initial_dist_);
  filter.update(measurement_model_, measurement_);

  Eigen::MatrixXd particles;
  Eigen::VectorXd weights;

  filter.getParticlesAndWeights(&particles, &weights);

  EXPECT_EQ(expected_updated_weights_, weights);
  EXPECT_EQ(expected_initial_particles_, particles);
}

TEST_F(ParticleFilterTest, GetExpectationTest) {
  ParticleFilter filter(2, &initial_dist_);
  filter.setParticlesAndWeights(Eigen::MatrixXd::Constant(1, 2, 1.0),
                                Eigen::VectorXd::Constant(2, 0.5));

  Eigen::VectorXd expectation = filter.getExpectation();

  EXPECT_EQ(Eigen::VectorXd::Constant(1, 1.0), expectation);
}

TEST_F(ParticleFilterTest, GetMaxWeightSampleTest) {
  ParticleFilter filter(2, &initial_dist_);

  Eigen::MatrixXd particles(1, 2);
  Eigen::VectorXd weights(2);

  particles << 1.0, 2.0;
  weights << 0.75, 0.25;

  filter.setParticlesAndWeights(particles, weights);

  Eigen::VectorXd max_weight_particle = filter.getMaxWeightParticle();

  EXPECT_EQ(Eigen::VectorXd::Constant(1, 1.0), max_weight_particle);
}

TEST_F(ParticleFilterTest, GetParticlesTest) {
  ParticleFilter filter(2, &initial_dist_);

  Eigen::MatrixXd particles = filter.getParticles();

  EXPECT_EQ(expected_initial_particles_, particles);
}

TEST_F(ParticleFilterTest, GetParticlesAndWeightsTest) {
  ParticleFilter filter(2, &initial_dist_);

  Eigen::MatrixXd particles;
  Eigen::VectorXd weights;

  filter.getParticlesAndWeights(&particles, &weights);

  EXPECT_EQ(Eigen::Vector2d::Constant(0.5), weights);
  EXPECT_EQ(expected_initial_particles_, particles);
}

}  // namespace refill
