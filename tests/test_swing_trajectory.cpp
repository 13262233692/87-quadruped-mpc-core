#include <gtest/gtest.h>
#include "bezier_curve.h"
#include "swing_trajectory.h"
#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace quadruped;

class BezierCurveTest : public ::testing::Test {
protected:
    void SetUp() override {}
};

TEST_F(BezierCurveTest, CubicEndpoints) {
    Eigen::Vector3d p0(0, 0, 0);
    Eigen::Vector3d p1(1, 2, 3);
    Eigen::Vector3d p2(4, 5, 6);
    Eigen::Vector3d p3(7, 8, 9);

    auto curve = BezierCurve::cubic(p0, p1, p2, p3);

    EXPECT_NEAR(curve.evaluate(0.0).x(), p0.x(), 1e-12);
    EXPECT_NEAR(curve.evaluate(0.0).y(), p0.y(), 1e-12);
    EXPECT_NEAR(curve.evaluate(0.0).z(), p0.z(), 1e-12);
    EXPECT_NEAR(curve.evaluate(1.0).x(), p3.x(), 1e-12);
    EXPECT_NEAR(curve.evaluate(1.0).y(), p3.y(), 1e-12);
    EXPECT_NEAR(curve.evaluate(1.0).z(), p3.z(), 1e-12);
}

TEST_F(BezierCurveTest, QuinticEndpoints) {
    Eigen::Vector3d p0(0, 0, 0);
    Eigen::Vector3d p1(1, 0, 2);
    Eigen::Vector3d p2(2, 0, 4);
    Eigen::Vector3d p3(3, 0, 4);
    Eigen::Vector3d p4(4, 0, 2);
    Eigen::Vector3d p5(5, 0, 0);

    auto curve = BezierCurve::quintic(p0, p1, p2, p3, p4, p5);

    EXPECT_NEAR(curve.evaluate(0.0).x(), p0.x(), 1e-12);
    EXPECT_NEAR(curve.evaluate(1.0).x(), p5.x(), 1e-12);
    EXPECT_NEAR(curve.evaluate(0.0).z(), p0.z(), 1e-12);
    EXPECT_NEAR(curve.evaluate(1.0).z(), p5.z(), 1e-12);
}

TEST_F(BezierCurveTest, LinearBezierIsLinear) {
    Eigen::Vector3d p0(0, 0, 0);
    Eigen::Vector3d p1(10, 10, 10);

    BezierCurve linear({p0, p1});

    for (double t = 0.0; t <= 1.0; t += 0.1) {
        Eigen::Vector3d pt = linear.evaluate(t);
        EXPECT_NEAR(pt.x(), 10.0 * t, 1e-10);
        EXPECT_NEAR(pt.y(), 10.0 * t, 1e-10);
        EXPECT_NEAR(pt.z(), 10.0 * t, 1e-10);
    }
}

TEST_F(BezierCurveTest, CubicMidpoint) {
    Eigen::Vector3d p0(0, 0, 0);
    Eigen::Vector3d p1(1, 0, 0);
    Eigen::Vector3d p2(1, 0, 0);
    Eigen::Vector3d p3(2, 0, 0);

    auto curve = BezierCurve::cubic(p0, p1, p2, p3);
    Eigen::Vector3d mid = curve.evaluate(0.5);
    EXPECT_NEAR(mid.x(), 1.0, 1e-10);
}

TEST_F(BezierCurveTest, DerivativeAtEndpoints) {
    Eigen::Vector3d p0(0, 0, 0);
    Eigen::Vector3d p1(3, 0, 0);
    Eigen::Vector3d p2(3, 0, 0);
    Eigen::Vector3d p3(6, 0, 0);

    auto curve = BezierCurve::cubic(p0, p1, p2, p3);

    Eigen::Vector3d d0 = curve.derivative(0.0);
    Eigen::Vector3d d1 = curve.derivative(1.0);

    EXPECT_NEAR(d0.x(), 9.0, 1e-10);
    EXPECT_NEAR(d1.x(), 9.0, 1e-10);
}

TEST_F(BezierCurveTest, DerivativeAtStartMatchesControlPointDiff) {
    Eigen::Vector3d p0(0, 0, 0);
    Eigen::Vector3d p1(2, 3, 4);
    Eigen::Vector3d p2(5, 6, 7);
    Eigen::Vector3d p3(8, 9, 10);

    auto curve = BezierCurve::cubic(p0, p1, p2, p3);
    Eigen::Vector3d d0 = curve.derivative(0.0);

    Eigen::Vector3d expected = 3.0 * (p1 - p0);
    EXPECT_NEAR(d0.x(), expected.x(), 1e-10);
    EXPECT_NEAR(d0.y(), expected.y(), 1e-10);
    EXPECT_NEAR(d0.z(), expected.z(), 1e-10);
}

TEST_F(BezierCurveTest, SecondDerivative) {
    Eigen::Vector3d p0(0, 0, 0);
    Eigen::Vector3d p1(1, 0, 0);
    Eigen::Vector3d p2(2, 0, 0);
    Eigen::Vector3d p3(3, 0, 0);

    auto curve = BezierCurve::cubic(p0, p1, p2, p3);
    Eigen::Vector3d dd = curve.secondDerivative(0.5);
    EXPECT_NEAR(dd.x(), 0.0, 1e-10);
}

TEST_F(BezierCurveTest, ConvexHullProperty) {
    Eigen::Vector3d p0(0, 0, 0);
    Eigen::Vector3d p1(1, 2, 3);
    Eigen::Vector3d p2(4, 5, 6);
    Eigen::Vector3d p3(7, 8, 9);

    auto curve = BezierCurve::cubic(p0, p1, p2, p3);

    for (double t = 0.0; t <= 1.0; t += 0.05) {
        Eigen::Vector3d pt = curve.evaluate(t);
        EXPECT_GE(pt.x(), 0.0);
        EXPECT_LE(pt.x(), 7.0);
    }
}

TEST_F(BezierCurveTest, ArcLengthPositive) {
    Eigen::Vector3d p0(0, 0, 0);
    Eigen::Vector3d p1(1, 2, 3);
    Eigen::Vector3d p2(4, 5, 6);
    Eigen::Vector3d p3(7, 8, 9);

    auto curve = BezierCurve::cubic(p0, p1, p2, p3);
    double len = curve.arcLength();
    EXPECT_GT(len, 0.0);

    double straight = (p3 - p0).norm();
    EXPECT_GE(len, straight - 1e-6);
}

TEST_F(BezierCurveTest, SampleCount) {
    Eigen::Vector3d p0(0, 0, 0);
    Eigen::Vector3d p1(1, 0, 1);
    Eigen::Vector3d p2(2, 0, 1);
    Eigen::Vector3d p3(3, 0, 0);

    auto curve = BezierCurve::cubic(p0, p1, p2, p3);
    auto samples = curve.sample(20);
    EXPECT_EQ(static_cast<int>(samples.size()), 21);
}

TEST_F(BezierCurveTest, MinMaxZ) {
    Eigen::Vector3d p0(0, 0, 0);
    Eigen::Vector3d p1(1, 0, 5);
    Eigen::Vector3d p2(2, 0, -3);
    Eigen::Vector3d p3(3, 0, 0);

    auto curve = BezierCurve::cubic(p0, p1, p2, p3);
    EXPECT_NEAR(curve.maxZ(), 5.0, 1e-10);
    EXPECT_NEAR(curve.minZ(), -3.0, 1e-10);
}

TEST_F(BezierCurveTest, GroundPenetration) {
    Eigen::Vector3d p0(0, 0, 0.1);
    Eigen::Vector3d p1(1, 0, 0.2);
    Eigen::Vector3d p2(2, 0, 0.15);
    Eigen::Vector3d p3(3, 0, 0.1);

    auto curve = BezierCurve::cubic(p0, p1, p2, p3);
    EXPECT_FALSE(curve.hasGroundPenetration(0.0));

    Eigen::Vector3d q0(0, 0, -0.1);
    Eigen::Vector3d q1(1, 0, -0.2);
    Eigen::Vector3d q2(2, 0, -0.15);
    Eigen::Vector3d q3(3, 0, -0.1);

    auto curve2 = BezierCurve::cubic(q0, q1, q2, q3);
    EXPECT_TRUE(curve2.hasGroundPenetration(0.0));
}

TEST_F(BezierCurveTest, ElevateDegree) {
    Eigen::Vector3d p0(0, 0, 0);
    Eigen::Vector3d p1(1, 0, 2);
    Eigen::Vector3d p2(2, 0, 2);
    Eigen::Vector3d p3(3, 0, 0);

    auto cubic = BezierCurve::cubic(p0, p1, p2, p3);
    EXPECT_EQ(cubic.order(), 3);

    auto quartic = cubic.elevateDegree();
    EXPECT_EQ(quartic.order(), 4);
    EXPECT_EQ(quartic.numControlPoints(), 5);

    for (double t = 0.0; t <= 1.0; t += 0.1) {
        Eigen::Vector3d p_orig = cubic.evaluate(t);
        Eigen::Vector3d p_elev = quartic.evaluate(t);
        EXPECT_NEAR((p_orig - p_elev).norm(), 0.0, 1e-10);
    }
}

TEST_F(BezierCurveTest, SetControlPoint) {
    Eigen::Vector3d p0(0, 0, 0);
    Eigen::Vector3d p1(1, 0, 0);
    Eigen::Vector3d p2(2, 0, 0);
    Eigen::Vector3d p3(3, 0, 0);

    auto curve = BezierCurve::cubic(p0, p1, p2, p3);
    curve.setControlPoint(1, Eigen::Vector3d(1, 0, 5));

    auto cp = curve.controlPoints();
    EXPECT_NEAR(cp[1].z(), 5.0, 1e-12);
}


class SwingTrajectoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        planner_ = std::make_unique<SwingTrajectoryPlanner>();
    }
    std::unique_ptr<SwingTrajectoryPlanner> planner_;
};

TEST_F(SwingTrajectoryTest, BasicCubicSwing) {
    Eigen::Vector3d start(0.2, 0.1, 0.0);
    Eigen::Vector3d target(0.3, 0.1, 0.0);

    auto config = planner_->getConfig();
    config.bezier_order = BezierOrder::CUBIC;
    config.swing_height = 0.08;
    planner_->setConfig(config);

    auto curve = planner_->buildBodyFrameBezier(start, target);

    EXPECT_NEAR(curve.evaluate(0.0).x(), start.x(), 1e-10);
    EXPECT_NEAR(curve.evaluate(1.0).x(), target.x(), 1e-10);
    EXPECT_GT(curve.evaluate(0.5).z(), start.z());
}

TEST_F(SwingTrajectoryTest, BasicQuinticSwing) {
    Eigen::Vector3d start(0.2, 0.1, 0.0);
    Eigen::Vector3d target(0.3, 0.1, 0.0);

    auto config = planner_->getConfig();
    config.bezier_order = BezierOrder::QUINTIC;
    config.swing_height = 0.1;
    planner_->setConfig(config);

    auto curve = planner_->buildBodyFrameBezier(start, target);

    EXPECT_NEAR(curve.evaluate(0.0).x(), start.x(), 1e-10);
    EXPECT_NEAR(curve.evaluate(1.0).x(), target.x(), 1e-10);
    EXPECT_GT(curve.evaluate(0.5).z(), start.z());
    EXPECT_EQ(curve.order(), 5);
}

TEST_F(SwingTrajectoryTest, SwingHeightRespected) {
    Eigen::Vector3d start(0.2, 0.1, 0.0);
    Eigen::Vector3d target(0.3, 0.1, 0.0);
    double desired_height = 0.12;

    auto config = planner_->getConfig();
    config.swing_height = desired_height;
    config.bezier_order = BezierOrder::CUBIC;
    planner_->setConfig(config);

    auto curve = planner_->buildBodyFrameBezier(start, target);

    auto samples = curve.sample(100);
    double max_z = 0.0;
    for (const auto& p : samples) {
        max_z = std::max(max_z, p.z());
    }

    EXPECT_GT(max_z, desired_height * 0.5);
    EXPECT_LT(max_z, desired_height * 2.0);
}

TEST_F(SwingTrajectoryTest, BodyToWorldTransform) {
    Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
    Eigen::Vector3d pos(1.0, 2.0, 3.0);

    Eigen::Vector3d body_pt(0.1, 0.2, 0.0);
    Eigen::Vector3d world_pt = planner_->bodyToWorld(body_pt, q, pos);

    EXPECT_NEAR(world_pt.x(), 1.1, 1e-10);
    EXPECT_NEAR(world_pt.y(), 2.2, 1e-10);
    EXPECT_NEAR(world_pt.z(), 3.0, 1e-10);
}

TEST_F(SwingTrajectoryTest, BodyToWorldWithTilt) {
    Eigen::Quaterniond q(Eigen::AngleAxisd(M_PI / 6, Eigen::Vector3d::UnitY()));
    Eigen::Vector3d pos(0.0, 0.0, 0.5);

    Eigen::Vector3d body_pt(0.2, 0.0, -0.3);
    Eigen::Vector3d world_pt = planner_->bodyToWorld(body_pt, q, pos);

    Eigen::Vector3d roundtrip = planner_->worldToBody(world_pt, q, pos);
    EXPECT_NEAR(roundtrip.x(), body_pt.x(), 1e-10);
    EXPECT_NEAR(roundtrip.y(), body_pt.y(), 1e-10);
    EXPECT_NEAR(roundtrip.z(), body_pt.z(), 1e-10);
}

TEST_F(SwingTrajectoryTest, GroundPenetrationDetection) {
    Eigen::Vector3d start(0.2, 0.1, 0.0);
    Eigen::Vector3d target(0.3, 0.1, 0.0);

    auto config = planner_->getConfig();
    config.swing_height = 0.08;
    planner_->setConfig(config);

    auto curve = planner_->buildBodyFrameBezier(start, target);

    Eigen::Quaterniond q(Eigen::AngleAxisd(M_PI / 4, Eigen::Vector3d::UnitX()));
    Eigen::Vector3d pos(0.0, 0.0, 0.05);

    auto world_curve = planner_->transformBezierToWorld(curve, q, pos);

    bool has_pen = planner_->checkGroundPenetration(world_curve, 0.0);
    double max_pen = planner_->computeMaxPenetration(world_curve, 0.0);

    if (has_pen) {
        EXPECT_GT(max_pen, 0.0);
    }
}

TEST_F(SwingTrajectoryTest, GroundPenaltyCorrection) {
    Eigen::Vector3d start(0.2, 0.1, 0.0);
    Eigen::Vector3d target(0.3, 0.1, 0.0);

    auto config = planner_->getConfig();
    config.swing_height = 0.08;
    config.ground_height = 0.0;
    config.ground_clearance = 0.005;
    config.penetration_penalty_gain = 2.0;
    planner_->setConfig(config);

    Eigen::Quaterniond q(Eigen::AngleAxisd(M_PI / 4, Eigen::Vector3d::UnitX()));
    Eigen::Vector3d pos(0.0, 0.0, 0.05);

    auto result = planner_->generateSwingTrajectory(start, target, q, pos);

    EXPECT_TRUE(result.is_valid);
    auto world_samples = result.bezier_world.sample(100);
    for (const auto& p : world_samples) {
        EXPECT_GE(p.z(), -0.01);
    }
}

TEST_F(SwingTrajectoryTest, GenerateWithState) {
    Eigen::Vector3d start(0.2, 0.1, 0.0);
    Eigen::Vector3d target(0.3, 0.1, 0.0);

    StateVector state = StateVector::Zero();
    state(POS_X) = 0.0;
    state(POS_Y) = 0.0;
    state(POS_Z) = 0.5;
    state(QUAT_W) = 1.0;

    auto result = planner_->generateSwingTrajectory(start, target, state);

    EXPECT_TRUE(result.is_valid);
    EXPECT_GT(static_cast<int>(result.trajectory_world.size()), 0);

    auto body_samples = result.bezier_body.sample(50);
    for (const auto& p : body_samples) {
        EXPECT_GT(p.z(), -0.1);
    }
}

TEST_F(SwingTrajectoryTest, LegPhaseManagement) {
    EXPECT_EQ(planner_->getLegPhase(0), LegPhase::STANCE);
    EXPECT_EQ(planner_->getLegPhase(3), LegPhase::STANCE);

    planner_->setLegPhase(0, LegPhase::SWING);
    EXPECT_EQ(planner_->getLegPhase(0), LegPhase::SWING);
    EXPECT_EQ(planner_->getLegPhase(1), LegPhase::STANCE);

    planner_->resetLeg(0);
    EXPECT_EQ(planner_->getLegPhase(0), LegPhase::STANCE);
}

TEST_F(SwingTrajectoryTest, SwingPhaseProgress) {
    planner_->setLegPhase(0, LegPhase::SWING);
    EXPECT_DOUBLE_EQ(planner_->getSwingPhaseProgress(0), 0.0);

    EXPECT_DOUBLE_EQ(planner_->getSwingPhaseProgress(1), 0.0);
}

TEST_F(SwingTrajectoryTest, UprightBodyNoPenetration) {
    Eigen::Vector3d start(0.2, 0.1, 0.0);
    Eigen::Vector3d target(0.3, 0.1, 0.0);

    Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
    Eigen::Vector3d pos(0.0, 0.0, 0.5);

    auto config = planner_->getConfig();
    config.swing_height = 0.08;
    config.ground_height = 0.0;
    planner_->setConfig(config);

    auto result = planner_->generateSwingTrajectory(start, target, q, pos);

    EXPECT_TRUE(result.is_valid);

    auto world_samples = result.bezier_world.sample(100);
    for (const auto& p : world_samples) {
        EXPECT_GT(p.z(), -0.01);
    }
}

TEST_F(SwingTrajectoryTest, ZeroStepNoSelfIntersection) {
    Eigen::Vector3d start(0.2, 0.1, 0.0);
    Eigen::Vector3d target = start;

    auto config = planner_->getConfig();
    config.swing_height = 0.05;
    planner_->setConfig(config);

    auto curve = planner_->buildBodyFrameBezier(start, target);

    EXPECT_NEAR(curve.evaluate(0.0).x(), start.x(), 1e-10);
    EXPECT_NEAR(curve.evaluate(1.0).x(), target.x(), 1e-10);

    auto samples = curve.sample(50);
    double max_z = 0.0;
    for (const auto& p : samples) {
        max_z = std::max(max_z, p.z());
    }
    EXPECT_GT(max_z, 0.0);
}

TEST_F(SwingTrajectoryTest, LargeTiltPenaltyCorrection) {
    Eigen::Vector3d start(0.2, 0.15, 0.0);
    Eigen::Vector3d target(0.3, 0.15, 0.0);

    Eigen::Quaterniond q(Eigen::AngleAxisd(M_PI / 3, Eigen::Vector3d::UnitX()));
    Eigen::Vector3d pos(0.0, 0.0, 0.15);

    auto config = planner_->getConfig();
    config.swing_height = 0.06;
    config.ground_height = 0.0;
    config.ground_clearance = 0.01;
    config.penetration_penalty_gain = 3.0;
    config.penalty_max_iterations = 15;
    planner_->setConfig(config);

    auto result = planner_->generateSwingTrajectory(start, target, q, pos);

    EXPECT_TRUE(result.is_valid);
}

TEST_F(SwingTrajectoryTest, BezierOrderSwitch) {
    Eigen::Vector3d start(0.2, 0.1, 0.0);
    Eigen::Vector3d target(0.3, 0.1, 0.0);

    auto config = planner_->getConfig();
    config.bezier_order = BezierOrder::CUBIC;
    planner_->setConfig(config);
    auto cubic_curve = planner_->buildBodyFrameBezier(start, target);
    EXPECT_EQ(cubic_curve.order(), 3);

    config.bezier_order = BezierOrder::QUINTIC;
    planner_->setConfig(config);
    auto quintic_curve = planner_->buildBodyFrameBezier(start, target);
    EXPECT_EQ(quintic_curve.order(), 5);
}

TEST_F(SwingTrajectoryTest, TransformBezierToWorldPreservesShape) {
    Eigen::Vector3d start(0.2, 0.1, 0.0);
    Eigen::Vector3d target(0.3, 0.1, 0.0);

    auto curve = planner_->buildBodyFrameBezier(start, target);

    Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
    Eigen::Vector3d pos(0.0, 0.0, 0.0);

    auto world_curve = planner_->transformBezierToWorld(curve, q, pos);

    for (double t = 0.0; t <= 1.0; t += 0.1) {
        Eigen::Vector3d body_pt = curve.evaluate(t);
        Eigen::Vector3d world_pt = world_curve.evaluate(t);
        EXPECT_NEAR(body_pt.x(), world_pt.x(), 1e-10);
        EXPECT_NEAR(body_pt.y(), world_pt.y(), 1e-10);
        EXPECT_NEAR(body_pt.z(), world_pt.z(), 1e-10);
    }
}
