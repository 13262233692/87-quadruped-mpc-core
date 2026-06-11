#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <array>
#include <vector>

#include "bezier_curve.h"
#include "srbd_model.h"

namespace quadruped {

enum class BezierOrder {
    CUBIC = 3,
    QUINTIC = 5
};

enum class LegPhase {
    STANCE = 0,
    SWING = 1
};

enum class LegIndex {
    FL = 0,
    FR = 1,
    RL = 2,
    RR = 3
};

struct SwingTrajectoryConfig {
    double swing_height = 0.08;
    double swing_duration = 0.25;
    double lift_ratio = 0.3;
    double descent_ratio = 0.3;
    BezierOrder bezier_order = BezierOrder::QUINTIC;
    double ground_height = 0.0;
    double ground_clearance = 0.005;
    double penetration_penalty_gain = 2.0;
    int penalty_max_iterations = 10;
    double z_velocity_at_start = 0.0;
    double z_velocity_at_end = 0.0;
    double x_velocity_at_start = 0.0;
    double x_velocity_at_end = 0.0;
};

struct SwingTrajectoryResult {
    BezierCurve bezier_body;
    BezierCurve bezier_world;
    std::vector<Eigen::Vector3d> trajectory_body;
    std::vector<Eigen::Vector3d> trajectory_world;
    double ground_penetration_max = 0.0;
    bool had_penetration = false;
    int penalty_iterations = 0;
    bool is_valid = true;
};

struct LegSwingState {
    LegPhase phase = LegPhase::STANCE;
    double phase_time = 0.0;
    Eigen::Vector3d start_position = Eigen::Vector3d::Zero();
    Eigen::Vector3d target_position = Eigen::Vector3d::Zero();
    SwingTrajectoryResult trajectory;
};

class SwingTrajectoryPlanner {
public:
    SwingTrajectoryPlanner();

    void setConfig(const SwingTrajectoryConfig& config);
    const SwingTrajectoryConfig& getConfig() const { return config_; }

    SwingTrajectoryResult generateSwingTrajectory(
        const Eigen::Vector3d& start_body,
        const Eigen::Vector3d& target_body,
        const Eigen::Quaterniond& body_orientation,
        const Eigen::Vector3d& body_position
    ) const;

    SwingTrajectoryResult generateSwingTrajectory(
        const Eigen::Vector3d& start_body,
        const Eigen::Vector3d& target_body,
        const StateVector& state
    ) const;

    BezierCurve buildBodyFrameBezier(
        const Eigen::Vector3d& start,
        const Eigen::Vector3d& target
    ) const;

    BezierCurve buildCubicBodyBezier(
        const Eigen::Vector3d& start,
        const Eigen::Vector3d& target
    ) const;

    BezierCurve buildQuinticBodyBezier(
        const Eigen::Vector3d& start,
        const Eigen::Vector3d& target
    ) const;

    Eigen::Vector3d bodyToWorld(
        const Eigen::Vector3d& point_body,
        const Eigen::Quaterniond& orientation,
        const Eigen::Vector3d& position
    ) const;

    Eigen::Vector3d worldToBody(
        const Eigen::Vector3d& point_world,
        const Eigen::Quaterniond& orientation,
        const Eigen::Vector3d& position
    ) const;

    BezierCurve transformBezierToWorld(
        const BezierCurve& curve_body,
        const Eigen::Quaterniond& orientation,
        const Eigen::Vector3d& position
    ) const;

    bool checkGroundPenetration(
        const BezierCurve& curve_world,
        double ground_height
    ) const;

    double computeMaxPenetration(
        const BezierCurve& curve_world,
        double ground_height,
        int num_samples = 50
    ) const;

    BezierCurve applyGroundPenalty(
        const BezierCurve& curve_body,
        const Eigen::Quaterniond& orientation,
        const Eigen::Vector3d& position,
        double ground_height,
        int max_iterations = 10
    ) const;

    void updateLegState(
        int leg_index,
        double dt,
        const Eigen::Vector3d& current_foot_body,
        const Eigen::Vector3d& target_foot_body,
        const StateVector& state
    );

    Eigen::Vector3d getCurrentFootPosition(
        int leg_index,
        const StateVector& state
    ) const;

    Eigen::Vector3d getCurrentFootVelocity(
        int leg_index,
        const StateVector& state
    ) const;

    LegPhase getLegPhase(int leg_index) const;
    double getSwingPhaseProgress(int leg_index) const;

    void setLegPhase(int leg_index, LegPhase phase);
    void resetLeg(int leg_index);

    const LegSwingState& getLegState(int leg_index) const;

private:
    SwingTrajectoryConfig config_;
    std::array<LegSwingState, NUM_LEGS> leg_states_;
};

}
