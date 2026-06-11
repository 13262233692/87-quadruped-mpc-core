#include "swing_trajectory.h"
#include <algorithm>
#include <cmath>

namespace quadruped {

SwingTrajectoryPlanner::SwingTrajectoryPlanner() {
    for (int i = 0; i < NUM_LEGS; ++i) {
        leg_states_[i] = LegSwingState{};
    }
}

void SwingTrajectoryPlanner::setConfig(const SwingTrajectoryConfig& config) {
    config_ = config;
}

BezierCurve SwingTrajectoryPlanner::buildCubicBodyBezier(
    const Eigen::Vector3d& start,
    const Eigen::Vector3d& target
) const {
    double h = config_.swing_height;
    Eigen::Vector3d mid = 0.5 * (start + target);
    double step_length = (target - start).norm();

    Eigen::Vector3d p0 = start;
    Eigen::Vector3d p3 = target;

    Eigen::Vector3d p1 = start;
    p1.x() += config_.lift_ratio * (target.x() - start.x());
    p1.y() += config_.lift_ratio * (target.y() - start.y());
    p1.z() = start.z() + h;

    Eigen::Vector3d p2 = target;
    p2.x() -= config_.descent_ratio * (target.x() - start.x());
    p2.y() -= config_.descent_ratio * (target.y() - start.y());
    p2.z() = target.z() + h;

    return BezierCurve::cubic(p0, p1, p2, p3);
}

BezierCurve SwingTrajectoryPlanner::buildQuinticBodyBezier(
    const Eigen::Vector3d& start,
    const Eigen::Vector3d& target
) const {
    double h = config_.swing_height;
    double dx = target.x() - start.x();
    double dy = target.y() - start.y();
    double dz = target.z() - start.z();

    Eigen::Vector3d p0 = start;
    Eigen::Vector3d p5 = target;

    double dt_total = config_.swing_duration;
    double vx_start = config_.x_velocity_at_start;
    double vy_start = 0.0;
    double vz_start = config_.z_velocity_at_start;
    double vx_end = config_.x_velocity_at_end;
    double vy_end = 0.0;
    double vz_end = config_.z_velocity_at_end;

    Eigen::Vector3d p1 = start + (dt_total / 5.0) * Eigen::Vector3d(vx_start, vy_start, vz_start);
    Eigen::Vector3d p4 = target - (dt_total / 5.0) * Eigen::Vector3d(vx_end, vy_end, vz_end);

    p1.z() = std::max(p1.z(), start.z() + h * 0.4);
    p4.z() = std::max(p4.z(), target.z() + h * 0.4);

    Eigen::Vector3d p2 = start;
    p2.x() += 0.4 * dx;
    p2.y() += 0.4 * dy;
    p2.z() = start.z() + h;

    Eigen::Vector3d p3 = target;
    p3.x() -= 0.4 * dx;
    p3.y() -= 0.4 * dy;
    p3.z() = target.z() + h;

    return BezierCurve::quintic(p0, p1, p2, p3, p4, p5);
}

BezierCurve SwingTrajectoryPlanner::buildBodyFrameBezier(
    const Eigen::Vector3d& start,
    const Eigen::Vector3d& target
) const {
    if (config_.bezier_order == BezierOrder::CUBIC) {
        return buildCubicBodyBezier(start, target);
    } else {
        return buildQuinticBodyBezier(start, target);
    }
}

Eigen::Vector3d SwingTrajectoryPlanner::bodyToWorld(
    const Eigen::Vector3d& point_body,
    const Eigen::Quaterniond& orientation,
    const Eigen::Vector3d& position
) const {
    return orientation * point_body + position;
}

Eigen::Vector3d SwingTrajectoryPlanner::worldToBody(
    const Eigen::Vector3d& point_world,
    const Eigen::Quaterniond& orientation,
    const Eigen::Vector3d& position
) const {
    return orientation.inverse() * (point_world - position);
}

BezierCurve SwingTrajectoryPlanner::transformBezierToWorld(
    const BezierCurve& curve_body,
    const Eigen::Quaterniond& orientation,
    const Eigen::Vector3d& position
) const {
    auto cp = curve_body.controlPoints();
    std::vector<Eigen::Vector3d> world_cp;
    world_cp.reserve(cp.size());
    for (const auto& p : cp) {
        world_cp.push_back(bodyToWorld(p, orientation, position));
    }
    return BezierCurve(world_cp);
}

bool SwingTrajectoryPlanner::checkGroundPenetration(
    const BezierCurve& curve_world,
    double ground_height
) const {
    return curve_world.hasGroundPenetration(ground_height);
}

double SwingTrajectoryPlanner::computeMaxPenetration(
    const BezierCurve& curve_world,
    double ground_height,
    int num_samples
) const {
    double max_pen = 0.0;
    auto samples = curve_world.sample(num_samples);
    for (const auto& p : samples) {
        double pen = ground_height - p.z();
        if (pen > max_pen) {
            max_pen = pen;
        }
    }
    return max_pen;
}

BezierCurve SwingTrajectoryPlanner::applyGroundPenalty(
    const BezierCurve& curve_body,
    const Eigen::Quaterniond& orientation,
    const Eigen::Vector3d& position,
    double ground_height,
    int max_iterations
) const {
    BezierCurve corrected = curve_body;
    double gain = config_.penetration_penalty_gain;

    for (int iter = 0; iter < max_iterations; ++iter) {
        BezierCurve world_curve = transformBezierToWorld(corrected, orientation, position);
        auto samples = world_curve.sample(50);

        double max_pen = 0.0;
        double max_pen_t = 0.0;
        for (int i = 0; i <= 50; ++i) {
            double t = static_cast<double>(i) / 50.0;
            double pen = ground_height - samples[i].z();
            if (pen > max_pen) {
                max_pen = pen;
                max_pen_t = t;
            }
        }

        if (max_pen <= 1e-6) {
            break;
        }

        double correction_body_z = max_pen + config_.ground_clearance;
        correction_body_z *= gain;

        int n = corrected.numControlPoints();
        for (int i = 0; i < n; ++i) {
            double cp_t = static_cast<double>(i) / static_cast<double>(n - 1);
            double weight = 1.0 - 2.0 * std::abs(cp_t - max_pen_t);
            weight = std::max(0.0, weight);
            weight = weight * weight;

            Eigen::Vector3d cp = corrected.controlPoints()[i];
            cp.z() += correction_body_z * weight;
            corrected.setControlPoint(i, cp);
        }
    }

    return corrected;
}

SwingTrajectoryResult SwingTrajectoryPlanner::generateSwingTrajectory(
    const Eigen::Vector3d& start_body,
    const Eigen::Vector3d& target_body,
    const Eigen::Quaterniond& body_orientation,
    const Eigen::Vector3d& body_position
) const {
    SwingTrajectoryResult result;

    BezierCurve curve_body = buildBodyFrameBezier(start_body, target_body);

    Eigen::Quaterniond q_normalized = body_orientation.normalized();

    curve_body = applyGroundPenalty(
        curve_body, q_normalized, body_position,
        config_.ground_height, config_.penalty_max_iterations
    );

    result.bezier_body = curve_body;

    result.bezier_world = transformBezierToWorld(curve_body, q_normalized, body_position);

    result.had_penetration = checkGroundPenetration(
        result.bezier_world, config_.ground_height
    );
    result.ground_penetration_max = computeMaxPenetration(
        result.bezier_world, config_.ground_height
    );

    if (result.had_penetration) {
        curve_body = applyGroundPenalty(
            curve_body, q_normalized, body_position,
            config_.ground_height + config_.ground_clearance,
            config_.penalty_max_iterations
        );
        result.bezier_body = curve_body;
        result.bezier_world = transformBezierToWorld(curve_body, q_normalized, body_position);
        result.had_penetration = checkGroundPenetration(
            result.bezier_world, config_.ground_height
        );
        result.ground_penetration_max = computeMaxPenetration(
            result.bezier_world, config_.ground_height
        );
        result.penalty_iterations = config_.penalty_max_iterations;
    }

    result.trajectory_body = curve_body.sample(50);
    result.trajectory_world = result.bezier_world.sample(50);
    result.is_valid = !result.had_penetration;

    return result;
}

SwingTrajectoryResult SwingTrajectoryPlanner::generateSwingTrajectory(
    const Eigen::Vector3d& start_body,
    const Eigen::Vector3d& target_body,
    const StateVector& state
) const {
    Eigen::Vector3d pos = state.segment<3>(POS_X);
    Eigen::Quaterniond quat(state(QUAT_W), state(QUAT_X), state(QUAT_Y), state(QUAT_Z));
    return generateSwingTrajectory(start_body, target_body, quat, pos);
}

void SwingTrajectoryPlanner::updateLegState(
    int leg_index,
    double dt,
    const Eigen::Vector3d& current_foot_body,
    const Eigen::Vector3d& target_foot_body,
    const StateVector& state
) {
    if (leg_index < 0 || leg_index >= NUM_LEGS) return;

    auto& ls = leg_states_[leg_index];

    if (ls.phase == LegPhase::SWING) {
        ls.phase_time += dt;
        if (ls.phase_time >= config_.swing_duration) {
            ls.phase = LegPhase::STANCE;
            ls.phase_time = 0.0;
            ls.start_position = current_foot_body;
            ls.target_position = current_foot_body;
        }
    } else {
        ls.start_position = current_foot_body;
        ls.target_position = target_foot_body;
    }
}

Eigen::Vector3d SwingTrajectoryPlanner::getCurrentFootPosition(
    int leg_index,
    const StateVector& state
) const {
    if (leg_index < 0 || leg_index >= NUM_LEGS) {
        return Eigen::Vector3d::Zero();
    }

    const auto& ls = leg_states_[leg_index];

    if (ls.phase == LegPhase::STANCE) {
        return ls.start_position;
    }

    double t = std::clamp(ls.phase_time / config_.swing_duration, 0.0, 1.0);
    return ls.trajectory.bezier_body.evaluate(t);
}

Eigen::Vector3d SwingTrajectoryPlanner::getCurrentFootVelocity(
    int leg_index,
    const StateVector& state
) const {
    if (leg_index < 0 || leg_index >= NUM_LEGS) {
        return Eigen::Vector3d::Zero();
    }

    const auto& ls = leg_states_[leg_index];

    if (ls.phase == LegPhase::STANCE) {
        return Eigen::Vector3d::Zero();
    }

    double t = std::clamp(ls.phase_time / config_.swing_duration, 0.0, 1.0);
    Eigen::Vector3d dpos_dt = ls.trajectory.bezier_body.derivative(t);
    return dpos_dt / config_.swing_duration;
}

LegPhase SwingTrajectoryPlanner::getLegPhase(int leg_index) const {
    if (leg_index < 0 || leg_index >= NUM_LEGS) return LegPhase::STANCE;
    return leg_states_[leg_index].phase;
}

double SwingTrajectoryPlanner::getSwingPhaseProgress(int leg_index) const {
    if (leg_index < 0 || leg_index >= NUM_LEGS) return 0.0;
    const auto& ls = leg_states_[leg_index];
    if (ls.phase != LegPhase::SWING) return 0.0;
    return std::clamp(ls.phase_time / config_.swing_duration, 0.0, 1.0);
}

void SwingTrajectoryPlanner::setLegPhase(int leg_index, LegPhase phase) {
    if (leg_index < 0 || leg_index >= NUM_LEGS) return;
    leg_states_[leg_index].phase = phase;
    if (phase == LegPhase::SWING) {
        leg_states_[leg_index].phase_time = 0.0;
    }
}

void SwingTrajectoryPlanner::resetLeg(int leg_index) {
    if (leg_index < 0 || leg_index >= NUM_LEGS) return;
    leg_states_[leg_index] = LegSwingState{};
}

const LegSwingState& SwingTrajectoryPlanner::getLegState(int leg_index) const {
    return leg_states_[leg_index];
}

}
