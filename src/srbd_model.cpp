#include "srbd_model.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <iostream>

namespace quadruped {

SRBDModel::SRBDModel()
    : mass_(12.0)
    , gravity_(0.0, 0.0, -9.81)
    , default_method_(DiscretizationMethod::MATRIX_EXPONENTIAL)
{
    inertia_ = Eigen::Matrix3d::Identity();
    inertia_ *= 0.2;
    updateInertiaInverse();
}

void SRBDModel::setMass(double mass) {
    if (mass <= 0.0) {
        throw std::invalid_argument("Mass must be positive");
    }
    mass_ = mass;
}

void SRBDModel::setInertia(const Eigen::Matrix3d& inertia) {
    if ((inertia - inertia.transpose()).norm() > 1e-6) {
        throw std::invalid_argument("Inertia matrix must be symmetric");
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(inertia);
    if (es.eigenvalues().minCoeff() <= 0.0) {
        throw std::invalid_argument("Inertia matrix must be positive definite");
    }
    inertia_ = inertia;
    updateInertiaInverse();
}

void SRBDModel::setGravity(const Eigen::Vector3d& gravity) {
    gravity_ = gravity;
}

void SRBDModel::updateInertiaInverse() {
    inertia_inv_ = inertia_.inverse();
}

Eigen::Vector3d SRBDModel::getComPosition(const StateVector& state) const {
    return state.segment<3>(POS_X);
}

Eigen::Quaterniond SRBDModel::getOrientation(const StateVector& state) const {
    return Eigen::Quaterniond(
        state(QUAT_W), state(QUAT_X), state(QUAT_Y), state(QUAT_Z)
    );
}

Eigen::Vector3d SRBDModel::getLinearVelocity(const StateVector& state) const {
    return state.segment<3>(VEL_X);
}

Eigen::Vector3d SRBDModel::getAngularVelocity(const StateVector& state) const {
    return state.segment<3>(ANG_VEL_X);
}

StateVector SRBDModel::buildState(
    const Eigen::Vector3d& pos,
    const Eigen::Quaterniond& quat,
    const Eigen::Vector3d& lin_vel,
    const Eigen::Vector3d& ang_vel
) const {
    StateVector state;
    state.segment<3>(POS_X) = pos;
    state(QUAT_W) = quat.w();
    state(QUAT_X) = quat.x();
    state(QUAT_Y) = quat.y();
    state(QUAT_Z) = quat.z();
    state.segment<3>(VEL_X) = lin_vel;
    state.segment<3>(ANG_VEL_X) = ang_vel;
    return state;
}

Eigen::Matrix3d SRBDModel::skewSymmetric(const Eigen::Vector3d& v) const {
    Eigen::Matrix3d skew;
    skew <<  0.0, -v.z(),  v.y(),
             v.z(),  0.0, -v.x(),
            -v.y(),  v.x(),  0.0;
    return skew;
}

Eigen::Matrix3d SRBDModel::rotationMatrix(const StateVector& state) const {
    Eigen::Quaterniond q = getOrientation(state);
    return q.toRotationMatrix();
}

Eigen::Matrix<double, 3, 4> SRBDModel::quaternionJacobian(
    const Eigen::Quaterniond& q
) const {
    Eigen::Matrix<double, 3, 4> Jq;
    Jq << -q.x(),  q.w(), -q.z(),  q.y(),
          -q.y(),  q.z(),  q.w(), -q.x(),
          -q.z(), -q.y(),  q.x(),  q.w();
    Jq *= 0.5;
    return Jq;
}

StateVector SRBDModel::continuousDynamics(
    const StateVector& state,
    const InputVector& input,
    const FootPosArray& foot_positions
) const {
    StateVector state_deriv;

    Eigen::Vector3d pos = getComPosition(state);
    Eigen::Quaterniond quat = getOrientation(state);
    Eigen::Vector3d lin_vel = getLinearVelocity(state);
    Eigen::Vector3d ang_vel = getAngularVelocity(state);

    state_deriv.segment<3>(POS_X) = lin_vel;

    Eigen::Vector4d q_vec(quat.w(), quat.x(), quat.y(), quat.z());
    Eigen::Matrix4d Omega;
    Omega << 0.0,   -ang_vel.x(), -ang_vel.y(), -ang_vel.z(),
             ang_vel.x(),  0.0,    ang_vel.z(), -ang_vel.y(),
             ang_vel.y(), -ang_vel.z(),  0.0,    ang_vel.x(),
             ang_vel.z(),  ang_vel.y(), -ang_vel.x(),  0.0;
    Eigen::Vector4d q_dot = 0.5 * Omega * q_vec;
    state_deriv(QUAT_W) = q_dot(0);
    state_deriv(QUAT_X) = q_dot(1);
    state_deriv(QUAT_Y) = q_dot(2);
    state_deriv(QUAT_Z) = q_dot(3);

    Eigen::Vector3d total_force(0.0, 0.0, 0.0);
    Eigen::Vector3d total_torque(0.0, 0.0, 0.0);

    Eigen::Matrix3d R = quat.toRotationMatrix();

    for (int i = 0; i < NUM_LEGS; ++i) {
        Eigen::Vector3d f_body = input.segment<3>(i * 3);
        total_force += f_body;

        Eigen::Vector3d r_body = R.transpose() * foot_positions[i] - pos;
        total_torque += r_body.cross(f_body);
    }

    total_force += mass_ * gravity_;

    Eigen::Vector3d lin_acc = total_force / mass_;
    state_deriv.segment<3>(VEL_X) = lin_acc;

    Eigen::Vector3d ang_acc = inertia_inv_ * (
        total_torque - ang_vel.cross(inertia_ * ang_vel)
    );
    state_deriv.segment<3>(ANG_VEL_X) = ang_acc;

    return state_deriv;
}

StateMatrix SRBDModel::continuousAMatrix(
    const StateVector& state,
    const FootPosArray& foot_positions,
    const ContactArray& contact
) const {
    StateMatrix A = StateMatrix::Zero();

    Eigen::Quaterniond quat = getOrientation(state);
    Eigen::Vector3d ang_vel = getAngularVelocity(state);

    A.block<3, 3>(POS_X, VEL_X) = Eigen::Matrix3d::Identity();

    double qw = quat.w(), qx = quat.x(), qy = quat.y(), qz = quat.z();
    double wx = ang_vel.x(), wy = ang_vel.y(), wz = ang_vel.z();

    Eigen::Matrix<double, 4, 4> dq_dot_dq;
    dq_dot_dq << 0.0, -wx, -wy, -wz,
                 wx,  0.0,  wz, -wy,
                 wy, -wz,  0.0,  wx,
                 wz,  wy, -wx,  0.0;
    dq_dot_dq *= 0.5;
    A.block<4, 4>(QUAT_W, QUAT_W) = dq_dot_dq;

    Eigen::Matrix<double, 4, 3> dq_dot_domega;
    dq_dot_domega << -qx, -qy, -qz,
                      qw, -qz,  qy,
                      qz,  qw, -qx,
                     -qy,  qx,  qw;
    dq_dot_domega *= 0.5;
    A.block<4, 3>(QUAT_W, ANG_VEL_X) = dq_dot_domega;

    Eigen::Matrix3d skew_omega = skewSymmetric(ang_vel);
    Eigen::Matrix3d skew_Iomega = skewSymmetric(inertia_ * ang_vel);
    A.block<3, 3>(ANG_VEL_X, ANG_VEL_X) = -inertia_inv_ * (
        skew_omega * inertia_ - skew_Iomega
    );

    return A;
}

InputMatrix SRBDModel::continuousBMatrix(
    const StateVector& state,
    const FootPosArray& foot_positions,
    const ContactArray& contact
) const {
    InputMatrix B = InputMatrix::Zero();

    Eigen::Quaterniond quat = getOrientation(state);
    Eigen::Matrix3d R = quat.toRotationMatrix();
    Eigen::Matrix3d R_transpose = R.transpose();

    for (int i = 0; i < NUM_LEGS; ++i) {
        if (!contact[i]) continue;

        int col_start = i * 3;

        B.block<3, 3>(VEL_X, col_start) = Eigen::Matrix3d::Identity() / mass_;

        Eigen::Vector3d r_foot = R_transpose * foot_positions[i] 
                                 - getComPosition(state);
        Eigen::Matrix3d r_skew = skewSymmetric(r_foot);
        B.block<3, 3>(ANG_VEL_X, col_start) = inertia_inv_ * r_skew;
    }

    return B;
}

void SRBDModel::continuousABMatrices(
    const StateVector& state,
    const FootPosArray& foot_positions,
    const ContactArray& contact,
    StateMatrix& A,
    InputMatrix& B
) const {
    A = continuousAMatrix(state, foot_positions, contact);
    B = continuousBMatrix(state, foot_positions, contact);
}

StateVector SRBDModel::integrateRK4(
    const StateVector& state,
    const InputVector& input,
    const FootPosArray& foot_positions,
    double dt
) const {
    StateVector k1 = continuousDynamics(state, input, foot_positions);
    StateVector k2 = continuousDynamics(
        state + 0.5 * dt * k1, input, foot_positions
    );
    StateVector k3 = continuousDynamics(
        state + 0.5 * dt * k2, input, foot_positions
    );
    StateVector k4 = continuousDynamics(
        state + dt * k3, input, foot_positions
    );

    StateVector new_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4);

    Eigen::Quaterniond q(
        new_state(QUAT_W), new_state(QUAT_X), 
        new_state(QUAT_Y), new_state(QUAT_Z)
    );
    q.normalize();
    new_state(QUAT_W) = q.w();
    new_state(QUAT_X) = q.x();
    new_state(QUAT_Y) = q.y();
    new_state(QUAT_Z) = q.z();

    return new_state;
}

StateMatrix SRBDModel::discreteAMatrix(
    const StateVector& state,
    const FootPosArray& foot_positions,
    const ContactArray& contact,
    double dt
) const {
    StateMatrix A_cont;
    InputMatrix B_cont;
    continuousABMatrices(state, foot_positions, contact, A_cont, B_cont);
    auto result = discretize(A_cont, B_cont, dt, default_method_);
    return result.A_d;
}

InputMatrix SRBDModel::discreteBMatrix(
    const StateVector& state,
    const FootPosArray& foot_positions,
    const ContactArray& contact,
    double dt
) const {
    StateMatrix A_cont;
    InputMatrix B_cont;
    continuousABMatrices(state, foot_positions, contact, A_cont, B_cont);
    auto result = discretize(A_cont, B_cont, dt, default_method_);
    return result.B_d;
}

DiscretizationResult SRBDModel::discretize(
    const StateVector& state,
    const FootPosArray& foot_positions,
    const ContactArray& contact,
    double dt,
    DiscretizationMethod method
) const {
    StateMatrix A_cont;
    InputMatrix B_cont;
    continuousABMatrices(state, foot_positions, contact, A_cont, B_cont);
    return discretize(A_cont, B_cont, dt, method);
}

DiscretizationResult SRBDModel::discretize(
    const StateMatrix& A_cont,
    const InputMatrix& B_cont,
    double dt,
    DiscretizationMethod method
) const {
    switch (method) {
        case DiscretizationMethod::FORWARD_EULER:
            return discretizeForwardEuler(A_cont, B_cont, dt);
        case DiscretizationMethod::MATRIX_EXPONENTIAL:
        case DiscretizationMethod::ZOH:
            return discretizeMatrixExponential(A_cont, B_cont, dt);
        case DiscretizationMethod::TUSTIN:
            return discretizeTustin(A_cont, B_cont, dt);
        default:
            return discretizeMatrixExponential(A_cont, B_cont, dt);
    }
}

StateMatrix SRBDModel::matrixExponential(
    const StateMatrix& A,
    int& num_terms_used
) const {
    return matrixExponentialScalingSquaring(A, num_terms_used);
}

StateMatrix SRBDModel::matrixExponentialTaylor(
    const StateMatrix& A,
    int& num_terms_used,
    double tolerance,
    int max_terms
) const {
    StateMatrix exp_A = StateMatrix::Identity();
    StateMatrix term = StateMatrix::Identity();
    
    num_terms_used = 1;
    
    for (int k = 1; k <= max_terms; ++k) {
        term = term * A / k;
        exp_A += term;
        num_terms_used = k + 1;
        
        if (term.norm() < tolerance * exp_A.norm()) {
            break;
        }
    }
    
    return exp_A;
}

StateMatrix SRBDModel::matrixExponentialScalingSquaring(
    const StateMatrix& A,
    int& num_terms_used,
    double tolerance
) const {
    double norm_A = A.norm();
    
    int s = 0;
    if (norm_A > 1.0) {
        s = static_cast<int>(std::ceil(std::log2(norm_A)));
    }
    
    StateMatrix A_scaled = A / std::pow(2.0, s);
    
    StateMatrix exp_scaled = matrixExponentialTaylor(
        A_scaled, num_terms_used, tolerance, 50
    );
    
    StateMatrix result = exp_scaled;
    for (int i = 0; i < s; ++i) {
        result = result * result;
    }
    
    return result;
}

Eigen::MatrixXd SRBDModel::matrixExponentialDynamic(
    const Eigen::MatrixXd& A,
    int& num_terms_used,
    double tolerance
) const {
    double norm_A = A.norm();
    
    int s = 0;
    if (norm_A > 1.0) {
        s = static_cast<int>(std::ceil(std::log2(norm_A)));
    }
    
    Eigen::MatrixXd A_scaled = A / std::pow(2.0, s);
    
    int n = A.rows();
    Eigen::MatrixXd exp_A = Eigen::MatrixXd::Identity(n, n);
    Eigen::MatrixXd term = Eigen::MatrixXd::Identity(n, n);
    
    num_terms_used = 1;
    
    for (int k = 1; k <= 50; ++k) {
        term = term * A_scaled / k;
        exp_A += term;
        num_terms_used = k + 1;
        
        if (term.norm() < tolerance * exp_A.norm()) {
            break;
        }
    }
    
    Eigen::MatrixXd result = exp_A;
    for (int i = 0; i < s; ++i) {
        result = result * result;
    }
    
    return result;
}

DiscretizationResult SRBDModel::discretizeMatrixExponential(
    const StateMatrix& A_cont,
    const InputMatrix& B_cont,
    double dt
) const {
    DiscretizationResult result;
    
    constexpr int aug_dim = STATE_DIM + INPUT_DIM;
    Eigen::MatrixXd M(aug_dim, aug_dim);
    M.setZero();
    
    M.block<STATE_DIM, STATE_DIM>(0, 0) = dt * A_cont;
    M.block<STATE_DIM, INPUT_DIM>(0, STATE_DIM) = dt * B_cont;
    
    int terms_used;
    Eigen::MatrixXd exp_M = matrixExponentialDynamic(M, terms_used);
    result.matrix_exp_series_terms = terms_used;
    
    result.A_d = exp_M.block<STATE_DIM, STATE_DIM>(0, 0);
    result.B_d = exp_M.block<STATE_DIM, INPUT_DIM>(0, STATE_DIM);
    
    result.spectral_radius = spectralRadius(result.A_d);
    result.is_stable = checkStability(result.A_d);
    
    return result;
}

DiscretizationResult SRBDModel::discretizeTustin(
    const StateMatrix& A_cont,
    const InputMatrix& B_cont,
    double dt
) const {
    DiscretizationResult result;
    result.matrix_exp_series_terms = 0;
    
    StateMatrix I = StateMatrix::Identity();
    StateMatrix half_dt_A = 0.5 * dt * A_cont;
    
    StateMatrix lhs = I - half_dt_A;
    StateMatrix rhs = I + half_dt_A;
    
    result.A_d = lhs.colPivHouseholderQr().solve(rhs);
    
    InputMatrix B_d_part1 = 0.5 * dt * B_cont;
    InputMatrix B_d_part2 = lhs.colPivHouseholderQr().solve(B_d_part1);
    
    result.B_d = B_d_part1 + 0.5 * dt * A_cont * B_d_part2;
    result.B_d = lhs.colPivHouseholderQr().solve(result.B_d);
    
    result.spectral_radius = spectralRadius(result.A_d);
    result.is_stable = checkStability(result.A_d);
    
    return result;
}

DiscretizationResult SRBDModel::discretizeForwardEuler(
    const StateMatrix& A_cont,
    const InputMatrix& B_cont,
    double dt
) const {
    DiscretizationResult result;
    result.matrix_exp_series_terms = 2;
    
    result.A_d = StateMatrix::Identity() + dt * A_cont;
    result.B_d = dt * B_cont;
    
    result.spectral_radius = spectralRadius(result.A_d);
    result.is_stable = checkStability(result.A_d);
    
    return result;
}

double SRBDModel::spectralRadius(const StateMatrix& A) const {
    Eigen::EigenSolver<StateMatrix> es(A);
    auto eigenvalues = es.eigenvalues();
    
    double max_magnitude = 0.0;
    for (int i = 0; i < STATE_DIM; ++i) {
        double mag = std::abs(eigenvalues(i));
        if (mag > max_magnitude) {
            max_magnitude = mag;
        }
    }
    
    return max_magnitude;
}

bool SRBDModel::checkStability(const StateMatrix& A_disc) const {
    return spectralRadius(A_disc) <= 1.0 + 1e-8;
}

} 
