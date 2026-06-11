#include "srbd_model.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

namespace py = pybind11;
using namespace quadruped;

PYBIND11_MODULE(quadruped_srbd, m) {
    m.doc() = "Quadruped Single Rigid Body Dynamics (SRBD) library";

    m.attr("STATE_DIM") = STATE_DIM;
    m.attr("INPUT_DIM") = INPUT_DIM;
    m.attr("NUM_LEGS") = NUM_LEGS;
    m.attr("SPACE_DIM") = SPACE_DIM;

    py::class_<SRBDModel>(m, "SRBDModel")
        .def(py::init<>())
        .def("set_mass", &SRBDModel::setMass,
             py::arg("mass"))
        .def("set_inertia", &SRBDModel::setInertia,
             py::arg("inertia"))
        .def("set_gravity", &SRBDModel::setGravity,
             py::arg("gravity"))
        .def("get_mass", &SRBDModel::getMass)
        .def("get_inertia", &SRBDModel::getInertia)
        .def("get_gravity", &SRBDModel::getGravity)
        .def("get_com_position", &SRBDModel::getComPosition,
             py::arg("state"))
        .def("get_orientation", [](const SRBDModel& self, const StateVector& state) {
            Eigen::Quaterniond q = self.getOrientation(state);
            return Eigen::Vector4d(q.w(), q.x(), q.y(), q.z());
        }, py::arg("state"))
        .def("get_linear_velocity", &SRBDModel::getLinearVelocity,
             py::arg("state"))
        .def("get_angular_velocity", &SRBDModel::getAngularVelocity,
             py::arg("state"))
        .def("build_state", &SRBDModel::buildState,
             py::arg("pos"), py::arg("quat"), 
             py::arg("lin_vel"), py::arg("ang_vel"))
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
        .def("skew_symmetric", &SRBDModel::skewSymmetric,
             py::arg("v"))
        .def("rotation_matrix", &SRBDModel::rotationMatrix,
             py::arg("state"))
        .def("quaternion_jacobian", &SRBDModel::quaternionJacobian,
             py::arg("q"))
        .def("integrate_rk4", &SRBDModel::integrateRK4,
             py::arg("state"), py::arg("input"), 
             py::arg("foot_positions"), py::arg("dt"))
        .def("discrete_A_matrix", &SRBDModel::discreteAMatrix,
             py::arg("state"), py::arg("foot_positions"), 
             py::arg("contact"), py::arg("dt"))
        .def("discrete_B_matrix", &SRBDModel::discreteBMatrix,
             py::arg("state"), py::arg("foot_positions"), 
             py::arg("contact"), py::arg("dt"));

    m.def("skew_symmetric", [](const Eigen::Vector3d& v) {
        Eigen::Matrix3d skew;
        skew <<  0.0, -v.z(),  v.y(),
                 v.z(),  0.0, -v.x(),
                -v.y(),  v.x(),  0.0;
        return skew;
    }, py::arg("v"));
}
