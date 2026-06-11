#include <gtest/gtest.h>
#include "srbd_model.h"
#include <cmath>

using namespace quadruped;

class SRBDModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        model = std::make_unique<SRBDModel>();
        model->setMass(12.0);
        
        Eigen::Matrix3d inertia;
        inertia << 0.2, 0.0, 0.0,
                   0.0, 0.3, 0.0,
                   0.0, 0.0, 0.4;
        model->setInertia(inertia);

        for (int i = 0; i < NUM_LEGS; ++i) {
            foot_positions_[i] << 0.2, 0.1, -0.3;
        }
        foot_positions_[1].y() = -0.1;
        foot_positions_[2].x() = -0.2;
        foot_positions_[3].x() = -0.2;
        foot_positions_[3].y() = -0.1;

        for (int i = 0; i < NUM_LEGS; ++i) {
            contact_[i] = true;
        }

        state_ = StateVector::Zero();
        state_(POS_Z) = 0.5;
        state_(QUAT_W) = 1.0;

        input_ = InputVector::Zero();
        for (int i = 0; i < NUM_LEGS; ++i) {
            input_(i * 3 + 2) = 12.0 * 9.81 / 4.0;
        }
    }

    std::unique_ptr<SRBDModel> model;
    FootPosArray foot_positions_;
    ContactArray contact_;
    StateVector state_;
    InputVector input_;
};

TEST_F(SRBDModelTest, SkewSymmetricMatrix) {
    Eigen::Vector3d v(1.0, 2.0, 3.0);
    Eigen::Matrix3d skew = model->skewSymmetric(v);

    EXPECT_NEAR(skew(0, 0), 0.0, 1e-10);
    EXPECT_NEAR(skew(0, 1), -3.0, 1e-10);
    EXPECT_NEAR(skew(0, 2), 2.0, 1e-10);
    EXPECT_NEAR(skew(1, 0), 3.0, 1e-10);
    EXPECT_NEAR(skew(1, 1), 0.0, 1e-10);
    EXPECT_NEAR(skew(1, 2), -1.0, 1e-10);
    EXPECT_NEAR(skew(2, 0), -2.0, 1e-10);
    EXPECT_NEAR(skew(2, 1), 1.0, 1e-10);
    EXPECT_NEAR(skew(2, 2), 0.0, 1e-10);

    EXPECT_TRUE(skew.isApprox(-skew.transpose(), 1e-10));
}

TEST_F(SRBDModelTest, StateConstruction) {
    Eigen::Vector3d pos(1.0, 2.0, 3.0);
    Eigen::Quaterniond quat(1.0, 0.0, 0.0, 0.0);
    Eigen::Vector3d lin_vel(0.5, 0.3, 0.1);
    Eigen::Vector3d ang_vel(0.1, 0.2, 0.3);

    StateVector state = model->buildState(pos, quat, lin_vel, ang_vel);

    EXPECT_NEAR(state(POS_X), 1.0, 1e-10);
    EXPECT_NEAR(state(POS_Y), 2.0, 1e-10);
    EXPECT_NEAR(state(POS_Z), 3.0, 1e-10);
    EXPECT_NEAR(state(QUAT_W), 1.0, 1e-10);
    EXPECT_NEAR(state(QUAT_X), 0.0, 1e-10);
    EXPECT_NEAR(state(QUAT_Y), 0.0, 1e-10);
    EXPECT_NEAR(state(QUAT_Z), 0.0, 1e-10);
    EXPECT_NEAR(state(VEL_X), 0.5, 1e-10);
    EXPECT_NEAR(state(VEL_Y), 0.3, 1e-10);
    EXPECT_NEAR(state(VEL_Z), 0.1, 1e-10);
    EXPECT_NEAR(state(ANG_VEL_X), 0.1, 1e-10);
    EXPECT_NEAR(state(ANG_VEL_Y), 0.2, 1e-10);
    EXPECT_NEAR(state(ANG_VEL_Z), 0.3, 1e-10);
}

TEST_F(SRBDModelTest, GravityFallsCorrectly) {
    InputVector zero_input = InputVector::Zero();
    ContactArray no_contact;
    for (int i = 0; i < NUM_LEGS; ++i) no_contact[i] = false;

    StateVector deriv = model->continuousDynamics(
        state_, zero_input, foot_positions_
    );

    EXPECT_NEAR(deriv(VEL_X), 0.0, 1e-10);
    EXPECT_NEAR(deriv(VEL_Y), 0.0, 1e-10);
    EXPECT_NEAR(deriv(VEL_Z), -9.81, 1e-6);
}

TEST_F(SRBDModelTest, BalancedForcesEquilibrium) {
    StateVector deriv = model->continuousDynamics(
        state_, input_, foot_positions_
    );

    EXPECT_NEAR(deriv(VEL_Z), 0.0, 1e-6);
    EXPECT_NEAR(deriv(VEL_X), 0.0, 1e-6);
    EXPECT_NEAR(deriv(VEL_Y), 0.0, 1e-6);
}

TEST_F(SRBDModelTest, AMatrixDimensions) {
    StateMatrix A = model->continuousAMatrix(
        state_, foot_positions_, contact_
    );

    EXPECT_EQ(A.rows(), STATE_DIM);
    EXPECT_EQ(A.cols(), STATE_DIM);
}

TEST_F(SRBDModelTest, BMatrixDimensions) {
    InputMatrix B = model->continuousBMatrix(
        state_, foot_positions_, contact_
    );

    EXPECT_EQ(B.rows(), STATE_DIM);
    EXPECT_EQ(B.cols(), INPUT_DIM);
}

TEST_F(SRBDModelTest, BMatrixZeroForNoContact) {
    ContactArray no_contact;
    for (int i = 0; i < NUM_LEGS; ++i) no_contact[i] = false;

    InputMatrix B = model->continuousBMatrix(
        state_, foot_positions_, no_contact
    );

    EXPECT_TRUE(B.isZero(1e-10));
}

TEST_F(SRBDModelTest, RK4IntegrationConservesQuaternionNorm) {
    double dt = 0.01;
    StateVector new_state = model->integrateRK4(
        state_, input_, foot_positions_, dt
    );

    Eigen::Quaterniond q(
        new_state(QUAT_W), new_state(QUAT_X),
        new_state(QUAT_Y), new_state(QUAT_Z)
    );

    EXPECT_NEAR(q.norm(), 1.0, 1e-10);
}

TEST_F(SRBDModelTest, DiscreteAMatrixIdentityForZeroDt) {
    double dt = 1e-6;
    StateMatrix A_disc = model->discreteAMatrix(
        state_, foot_positions_, contact_, dt
    );

    StateMatrix expected = StateMatrix::Identity();
    double error = (A_disc - expected).norm();
    EXPECT_LT(error, 1e-3);
}

TEST_F(SRBDModelTest, DiscreteBMatrixSmallForZeroDt) {
    double dt = 1e-6;
    InputMatrix B_disc = model->discreteBMatrix(
        state_, foot_positions_, contact_, dt
    );

    EXPECT_LT(B_disc.norm(), 1e-3);
}

TEST_F(SRBDModelTest, RotationMatrixIdentityForDefaultOrientation) {
    Eigen::Matrix3d R = model->rotationMatrix(state_);
    EXPECT_TRUE(R.isApprox(Eigen::Matrix3d::Identity(), 1e-10));
}

TEST_F(SRBDModelTest, InvalidMassThrows) {
    EXPECT_THROW(model->setMass(-1.0), std::invalid_argument);
    EXPECT_THROW(model->setMass(0.0), std::invalid_argument);
}

TEST_F(SRBDModelTest, AsymmetricInertiaThrows) {
    Eigen::Matrix3d bad_inertia;
    bad_inertia << 0.2, 0.1, 0.0,
                   0.0, 0.3, 0.0,
                   0.0, 0.0, 0.4;
    EXPECT_THROW(model->setInertia(bad_inertia), std::invalid_argument);
}

TEST_F(SRBDModelTest, QuaternionJacobianCorrectSize) {
    Eigen::Quaterniond q(1.0, 0.0, 0.0, 0.0);
    auto Jq = model->quaternionJacobian(q);
    EXPECT_EQ(Jq.rows(), 3);
    EXPECT_EQ(Jq.cols(), 4);
}

TEST_F(SRBDModelTest, RK4FreeFallMatchesAnalytical) {
    InputVector zero_input = InputVector::Zero();
    double dt = 0.01;
    int steps = 100;

    StateVector s = state_;
    for (int i = 0; i < steps; ++i) {
        s = model->integrateRK4(s, zero_input, foot_positions_, dt);
    }

    double t = steps * dt;
    double expected_z = state_(POS_Z) + 0.5 * (-9.81) * t * t;
    double expected_vz = -9.81 * t;

    EXPECT_NEAR(s(POS_Z), expected_z, 1e-6);
    EXPECT_NEAR(s(VEL_Z), expected_vz, 1e-6);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
