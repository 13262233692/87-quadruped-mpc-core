#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <array>
#include <vector>

namespace quadruped {

constexpr int STATE_DIM = 13;      
constexpr int INPUT_DIM = 12;      
constexpr int NUM_LEGS = 4;        
constexpr int SPACE_DIM = 3;       

using StateVector = Eigen::Matrix<double, STATE_DIM, 1>;
using InputVector = Eigen::Matrix<double, INPUT_DIM, 1>;
using StateMatrix = Eigen::Matrix<double, STATE_DIM, STATE_DIM>;
using InputMatrix = Eigen::Matrix<double, STATE_DIM, INPUT_DIM>;
using FootPosArray = std::array<Eigen::Vector3d, NUM_LEGS>;
using ContactArray = std::array<bool, NUM_LEGS>;

enum StateIndex {
    POS_X = 0, POS_Y = 1, POS_Z = 2,
    QUAT_W = 3, QUAT_X = 4, QUAT_Y = 5, QUAT_Z = 6,
    VEL_X = 7, VEL_Y = 8, VEL_Z = 9,
    ANG_VEL_X = 10, ANG_VEL_Y = 11, ANG_VEL_Z = 12
};

class SRBDModel {
public:
    SRBDModel();

    void setMass(double mass);
    void setInertia(const Eigen::Matrix3d& inertia);
    void setGravity(const Eigen::Vector3d& gravity);

    double getMass() const { return mass_; }
    Eigen::Matrix3d getInertia() const { return inertia_; }
    Eigen::Vector3d getGravity() const { return gravity_; }

    Eigen::Vector3d getComPosition(const StateVector& state) const;
    Eigen::Quaterniond getOrientation(const StateVector& state) const;
    Eigen::Vector3d getLinearVelocity(const StateVector& state) const;
    Eigen::Vector3d getAngularVelocity(const StateVector& state) const;

    StateVector buildState(
        const Eigen::Vector3d& pos,
        const Eigen::Quaterniond& quat,
        const Eigen::Vector3d& lin_vel,
        const Eigen::Vector3d& ang_vel
    ) const;

    StateVector continuousDynamics(
        const StateVector& state,
        const InputVector& input,
        const FootPosArray& foot_positions
    ) const;

    StateMatrix continuousAMatrix(
        const StateVector& state,
        const FootPosArray& foot_positions,
        const ContactArray& contact
    ) const;

    InputMatrix continuousBMatrix(
        const StateVector& state,
        const FootPosArray& foot_positions,
        const ContactArray& contact
    ) const;

    void continuousABMatrices(
        const StateVector& state,
        const FootPosArray& foot_positions,
        const ContactArray& contact,
        StateMatrix& A,
        InputMatrix& B
    ) const;

    Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& v) const;

    Eigen::Matrix3d rotationMatrix(const StateVector& state) const;

    Eigen::Matrix<double, 3, 4> quaternionJacobian(const Eigen::Quaterniond& q) const;

    StateVector integrateRK4(
        const StateVector& state,
        const InputVector& input,
        const FootPosArray& foot_positions,
        double dt
    ) const;

    StateMatrix discreteAMatrix(
        const StateVector& state,
        const FootPosArray& foot_positions,
        const ContactArray& contact,
        double dt
    ) const;

    InputMatrix discreteBMatrix(
        const StateVector& state,
        const FootPosArray& foot_positions,
        const ContactArray& contact,
        double dt
    ) const;

private:
    double mass_;
    Eigen::Matrix3d inertia_;
    Eigen::Matrix3d inertia_inv_;
    Eigen::Vector3d gravity_;

    void updateInertiaInverse();
};

} 
