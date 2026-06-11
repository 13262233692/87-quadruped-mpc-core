#include "srbd_model.h"
#include "bezier_curve.h"
#include "swing_trajectory.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <sstream>

namespace py = pybind11;
using namespace quadruped;

PYBIND11_MODULE(quadruped_srbd, m) {
    m.doc() = "Quadruped Single Rigid Body Dynamics (SRBD) and Swing Trajectory library";

    m.attr("STATE_DIM") = STATE_DIM;
    m.attr("INPUT_DIM") = INPUT_DIM;
    m.attr("NUM_LEGS") = NUM_LEGS;
    m.attr("SPACE_DIM") = SPACE_DIM;

    py::enum_<DiscretizationMethod>(m, "DiscretizationMethod")
        .value("FORWARD_EULER", DiscretizationMethod::FORWARD_EULER)
        .value("MATRIX_EXPONENTIAL", DiscretizationMethod::MATRIX_EXPONENTIAL)
        .value("TUSTIN", DiscretizationMethod::TUSTIN)
        .value("ZOH", DiscretizationMethod::ZOH);

    py::class_<DiscretizationResult>(m, "DiscretizationResult")
        .def_readwrite("A_d", &DiscretizationResult::A_d)
        .def_readwrite("B_d", &DiscretizationResult::B_d)
        .def_readwrite("matrix_exp_series_terms", 
                       &DiscretizationResult::matrix_exp_series_terms)
        .def_readwrite("spectral_radius", 
                       &DiscretizationResult::spectral_radius)
        .def_readwrite("is_stable", &DiscretizationResult::is_stable)
        .def("__repr__", [](const DiscretizationResult& r) {
            std::ostringstream oss;
            oss << "DiscretizationResult("
                << "spectral_radius=" << r.spectral_radius
                << ", is_stable=" << (r.is_stable ? "True" : "False")
                << ", terms=" << r.matrix_exp_series_terms
                << ")";
            return oss.str();
        });

    py::class_<SRBDModel>(m, "SRBDModel")
        .def(py::init<>())
        .def("set_mass", &SRBDModel::setMass, py::arg("mass"))
        .def("set_inertia", &SRBDModel::setInertia, py::arg("inertia"))
        .def("set_gravity", &SRBDModel::setGravity, py::arg("gravity"))
        .def("get_mass", &SRBDModel::getMass)
        .def("get_inertia", &SRBDModel::getInertia)
        .def("get_gravity", &SRBDModel::getGravity)
        .def("get_com_position", &SRBDModel::getComPosition, py::arg("state"))
        .def("get_orientation", [](const SRBDModel& self, const StateVector& state) {
            Eigen::Quaterniond q = self.getOrientation(state);
            return Eigen::Vector4d(q.w(), q.x(), q.y(), q.z());
        }, py::arg("state"))
        .def("get_linear_velocity", &SRBDModel::getLinearVelocity, py::arg("state"))
        .def("get_angular_velocity", &SRBDModel::getAngularVelocity, py::arg("state"))
        .def("build_state", &SRBDModel::buildState,
             py::arg("pos"), py::arg("quat"), py::arg("lin_vel"), py::arg("ang_vel"))
        .def("continuous_dynamics", &SRBDModel::continuousDynamics,
             py::arg("state"), py::arg("input"), py::arg("foot_positions"))
        .def("continuous_A_matrix", &SRBDModel::continuousAMatrix,
             py::arg("state"), py::arg("foot_positions"), py::arg("contact"))
        .def("continuous_B_matrix", &SRBDModel::continuousBMatrix,
             py::arg("state"), py::arg("foot_positions"), py::arg("contact"))
        .def("continuous_AB_matrices", [](const SRBDModel& self,
             const StateVector& state,
             const FootPosArray& foot_positions,
             const ContactArray& contact) {
            StateMatrix A;
            InputMatrix B;
            self.continuousABMatrices(state, foot_positions, contact, A, B);
            return std::make_pair(A, B);
        }, py::arg("state"), py::arg("foot_positions"), py::arg("contact"))
        .def("skew_symmetric", &SRBDModel::skewSymmetric, py::arg("v"))
        .def("rotation_matrix", &SRBDModel::rotationMatrix, py::arg("state"))
        .def("quaternion_jacobian", &SRBDModel::quaternionJacobian, py::arg("q"))
        .def("integrate_rk4", &SRBDModel::integrateRK4,
             py::arg("state"), py::arg("input"), py::arg("foot_positions"), py::arg("dt"))
        .def("discrete_A_matrix", &SRBDModel::discreteAMatrix,
             py::arg("state"), py::arg("foot_positions"), py::arg("contact"), py::arg("dt"))
        .def("discrete_B_matrix", &SRBDModel::discreteBMatrix,
             py::arg("state"), py::arg("foot_positions"), py::arg("contact"), py::arg("dt"))
        .def("discretize", [](const SRBDModel& self,
             const StateVector& state,
             const FootPosArray& foot_positions,
             const ContactArray& contact,
             double dt,
             DiscretizationMethod method) {
            return self.discretize(state, foot_positions, contact, dt, method);
        }, py::arg("state"), py::arg("foot_positions"),
           py::arg("contact"), py::arg("dt"),
           py::arg("method") = DiscretizationMethod::MATRIX_EXPONENTIAL)
        .def("matrix_exponential", [](const SRBDModel& self, const StateMatrix& A) {
            int terms = 0;
            StateMatrix exp_A = self.matrixExponential(A, terms);
            return std::make_pair(exp_A, terms);
        }, py::arg("A"))
        .def("spectral_radius", &SRBDModel::spectralRadius, py::arg("A"))
        .def("check_stability", &SRBDModel::checkStability, py::arg("A_disc"))
        .def("set_discretization_method", &SRBDModel::setDiscretizationMethod, py::arg("method"))
        .def("get_discretization_method", &SRBDModel::getDiscretizationMethod);

    m.def("skew_symmetric", [](const Eigen::Vector3d& v) {
        Eigen::Matrix3d skew;
        skew <<  0.0, -v.z(),  v.y(),
                 v.z(),  0.0, -v.x(),
                -v.y(),  v.x(),  0.0;
        return skew;
    }, py::arg("v"));

    // ---- BezierCurve ----
    py::class_<BezierCurve>(m, "BezierCurve")
        .def(py::init<>())
        .def(py::init<const std::vector<Eigen::Vector3d>&>(), py::arg("control_points"))
        .def_static("cubic", &BezierCurve::cubic,
             py::arg("p0"), py::arg("p1"), py::arg("p2"), py::arg("p3"))
        .def_static("quintic", &BezierCurve::quintic,
             py::arg("p0"), py::arg("p1"), py::arg("p2"),
             py::arg("p3"), py::arg("p4"), py::arg("p5"))
        .def("evaluate", &BezierCurve::evaluate, py::arg("t"))
        .def("derivative", &BezierCurve::derivative, py::arg("t"))
        .def("second_derivative", &BezierCurve::secondDerivative, py::arg("t"))
        .def("sample", &BezierCurve::sample, py::arg("num_samples") = 50)
        .def("arc_length", &BezierCurve::arcLength, py::arg("num_segments") = 100)
        .def("control_points", &BezierCurve::controlPoints)
        .def("order", &BezierCurve::order)
        .def("num_control_points", &BezierCurve::numControlPoints)
        .def("start_point", &BezierCurve::startPoint)
        .def("end_point", &BezierCurve::endPoint)
        .def("set_control_point", &BezierCurve::setControlPoint,
             py::arg("index"), py::arg("point"))
        .def("min_z", &BezierCurve::minZ)
        .def("max_z", &BezierCurve::maxZ)
        .def("has_ground_penetration", &BezierCurve::hasGroundPenetration,
             py::arg("ground_height") = 0.0)
        .def("elevate_degree", &BezierCurve::elevateDegree)
        .def("__repr__", [](const BezierCurve& c) {
            std::ostringstream oss;
            oss << "BezierCurve(order=" << c.order()
                << ", n_points=" << c.numControlPoints() << ")";
            return oss.str();
        });

    // ---- Enums for swing trajectory ----
    py::enum_<BezierOrder>(m, "BezierOrder")
        .value("CUBIC", BezierOrder::CUBIC)
        .value("QUINTIC", BezierOrder::QUINTIC);

    py::enum_<LegPhase>(m, "LegPhase")
        .value("STANCE", LegPhase::STANCE)
        .value("SWING", LegPhase::SWING);

    py::enum_<LegIndex>(m, "LegIndex")
        .value("FL", LegIndex::FL)
        .value("FR", LegIndex::FR)
        .value("RL", LegIndex::RL)
        .value("RR", LegIndex::RR);

    // ---- SwingTrajectoryConfig ----
    py::class_<SwingTrajectoryConfig>(m, "SwingTrajectoryConfig")
        .def(py::init<>())
        .def_readwrite("swing_height", &SwingTrajectoryConfig::swing_height)
        .def_readwrite("swing_duration", &SwingTrajectoryConfig::swing_duration)
        .def_readwrite("lift_ratio", &SwingTrajectoryConfig::lift_ratio)
        .def_readwrite("descent_ratio", &SwingTrajectoryConfig::descent_ratio)
        .def_readwrite("bezier_order", &SwingTrajectoryConfig::bezier_order)
        .def_readwrite("ground_height", &SwingTrajectoryConfig::ground_height)
        .def_readwrite("ground_clearance", &SwingTrajectoryConfig::ground_clearance)
        .def_readwrite("penetration_penalty_gain", &SwingTrajectoryConfig::penetration_penalty_gain)
        .def_readwrite("penalty_max_iterations", &SwingTrajectoryConfig::penalty_max_iterations)
        .def_readwrite("z_velocity_at_start", &SwingTrajectoryConfig::z_velocity_at_start)
        .def_readwrite("z_velocity_at_end", &SwingTrajectoryConfig::z_velocity_at_end)
        .def_readwrite("x_velocity_at_start", &SwingTrajectoryConfig::x_velocity_at_start)
        .def_readwrite("x_velocity_at_end", &SwingTrajectoryConfig::x_velocity_at_end)
        .def("__repr__", [](const SwingTrajectoryConfig& c) {
            std::ostringstream oss;
            oss << "SwingTrajectoryConfig("
                << "swing_height=" << c.swing_height
                << ", bezier_order=" << static_cast<int>(c.bezier_order)
                << ", swing_duration=" << c.swing_duration
                << ")";
            return oss.str();
        });

    // ---- SwingTrajectoryResult ----
    py::class_<SwingTrajectoryResult>(m, "SwingTrajectoryResult")
        .def_readwrite("bezier_body", &SwingTrajectoryResult::bezier_body)
        .def_readwrite("bezier_world", &SwingTrajectoryResult::bezier_world)
        .def_readwrite("trajectory_body", &SwingTrajectoryResult::trajectory_body)
        .def_readwrite("trajectory_world", &SwingTrajectoryResult::trajectory_world)
        .def_readwrite("ground_penetration_max", &SwingTrajectoryResult::ground_penetration_max)
        .def_readwrite("had_penetration", &SwingTrajectoryResult::had_penetration)
        .def_readwrite("penalty_iterations", &SwingTrajectoryResult::penalty_iterations)
        .def_readwrite("is_valid", &SwingTrajectoryResult::is_valid)
        .def("__repr__", [](const SwingTrajectoryResult& r) {
            std::ostringstream oss;
            oss << "SwingTrajectoryResult("
                << "valid=" << (r.is_valid ? "True" : "False")
                << ", had_penetration=" << (r.had_penetration ? "True" : "False")
                << ", max_penetration=" << r.ground_penetration_max
                << ")";
            return oss.str();
        });

    // ---- LegSwingState ----
    py::class_<LegSwingState>(m, "LegSwingState")
        .def_readwrite("phase", &LegSwingState::phase)
        .def_readwrite("phase_time", &LegSwingState::phase_time)
        .def_readwrite("start_position", &LegSwingState::start_position)
        .def_readwrite("target_position", &LegSwingState::target_position)
        .def_readwrite("trajectory", &LegSwingState::trajectory);

    // ---- SwingTrajectoryPlanner ----
    py::class_<SwingTrajectoryPlanner>(m, "SwingTrajectoryPlanner")
        .def(py::init<>())
        .def("set_config", &SwingTrajectoryPlanner::setConfig, py::arg("config"))
        .def("get_config", &SwingTrajectoryPlanner::getConfig,
             py::return_value_policy::reference_internal)
        .def("generate_swing_trajectory",
             [](const SwingTrajectoryPlanner& self,
                const Eigen::Vector3d& start_body,
                const Eigen::Vector3d& target_body,
                const Eigen::Vector4d& quat_vec,
                const Eigen::Vector3d& body_position) {
                Eigen::Quaterniond q(quat_vec(0), quat_vec(1), quat_vec(2), quat_vec(3));
                return self.generateSwingTrajectory(start_body, target_body, q, body_position);
             },
             py::arg("start_body"), py::arg("target_body"),
             py::arg("body_quaternion"), py::arg("body_position"))
        .def("generate_swing_trajectory_state",
             [](const SwingTrajectoryPlanner& self,
                const Eigen::Vector3d& start_body,
                const Eigen::Vector3d& target_body,
                const StateVector& state) {
                return self.generateSwingTrajectory(start_body, target_body, state);
             },
             py::arg("start_body"), py::arg("target_body"), py::arg("state"))
        .def("build_body_frame_bezier", &SwingTrajectoryPlanner::buildBodyFrameBezier,
             py::arg("start"), py::arg("target"))
        .def("body_to_world", [](const SwingTrajectoryPlanner& self,
             const Eigen::Vector3d& point_body,
             const Eigen::Vector4d& quat_vec,
             const Eigen::Vector3d& position) {
                Eigen::Quaterniond q(quat_vec(0), quat_vec(1), quat_vec(2), quat_vec(3));
                return self.bodyToWorld(point_body, q, position);
             },
             py::arg("point_body"), py::arg("body_quaternion"), py::arg("position"))
        .def("world_to_body", [](const SwingTrajectoryPlanner& self,
             const Eigen::Vector3d& point_world,
             const Eigen::Vector4d& quat_vec,
             const Eigen::Vector3d& position) {
                Eigen::Quaterniond q(quat_vec(0), quat_vec(1), quat_vec(2), quat_vec(3));
                return self.worldToBody(point_world, q, position);
             },
             py::arg("point_world"), py::arg("body_quaternion"), py::arg("position"))
        .def("check_ground_penetration", &SwingTrajectoryPlanner::checkGroundPenetration,
             py::arg("curve_world"), py::arg("ground_height") = 0.0)
        .def("compute_max_penetration", &SwingTrajectoryPlanner::computeMaxPenetration,
             py::arg("curve_world"), py::arg("ground_height") = 0.0, py::arg("num_samples") = 50)
        .def("apply_ground_penalty", [](const SwingTrajectoryPlanner& self,
             const BezierCurve& curve_body,
             const Eigen::Vector4d& quat_vec,
             const Eigen::Vector3d& position,
             double ground_height,
             int max_iterations) {
                Eigen::Quaterniond q(quat_vec(0), quat_vec(1), quat_vec(2), quat_vec(3));
                return self.applyGroundPenalty(curve_body, q, position, ground_height, max_iterations);
             },
             py::arg("curve_body"), py::arg("body_quaternion"), py::arg("position"),
             py::arg("ground_height"), py::arg("max_iterations") = 10)
        .def("update_leg_state", &SwingTrajectoryPlanner::updateLegState,
             py::arg("leg_index"), py::arg("dt"),
             py::arg("current_foot_body"), py::arg("target_foot_body"),
             py::arg("state"))
        .def("get_current_foot_position", &SwingTrajectoryPlanner::getCurrentFootPosition,
             py::arg("leg_index"), py::arg("state"))
        .def("get_current_foot_velocity", &SwingTrajectoryPlanner::getCurrentFootVelocity,
             py::arg("leg_index"), py::arg("state"))
        .def("get_leg_phase", &SwingTrajectoryPlanner::getLegPhase, py::arg("leg_index"))
        .def("get_swing_phase_progress", &SwingTrajectoryPlanner::getSwingPhaseProgress,
             py::arg("leg_index"))
        .def("set_leg_phase", &SwingTrajectoryPlanner::setLegPhase,
             py::arg("leg_index"), py::arg("phase"))
        .def("reset_leg", &SwingTrajectoryPlanner::resetLeg, py::arg("leg_index"))
        .def("get_leg_state", &SwingTrajectoryPlanner::getLegState,
             py::arg("leg_index"), py::return_value_policy::reference_internal);
}
