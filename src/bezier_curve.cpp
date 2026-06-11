#include "bezier_curve.h"
#include <numeric>

namespace quadruped {

BezierCurve::BezierCurve(const std::vector<Eigen::Vector3d>& control_points)
    : control_points_(control_points) {
    if (control_points.empty()) {
        throw std::invalid_argument("BezierCurve requires at least 1 control point");
    }
}

BezierCurve BezierCurve::cubic(
    const Eigen::Vector3d& p0,
    const Eigen::Vector3d& p1,
    const Eigen::Vector3d& p2,
    const Eigen::Vector3d& p3
) {
    return BezierCurve({p0, p1, p2, p3});
}

BezierCurve BezierCurve::quintic(
    const Eigen::Vector3d& p0,
    const Eigen::Vector3d& p1,
    const Eigen::Vector3d& p2,
    const Eigen::Vector3d& p3,
    const Eigen::Vector3d& p4,
    const Eigen::Vector3d& p5
) {
    return BezierCurve({p0, p1, p2, p3, p4, p5});
}

Eigen::Vector3d BezierCurve::deCasteljau(
    const std::vector<Eigen::Vector3d>& points, double t) const {
    if (points.size() == 1) {
        return points[0];
    }
    std::vector<Eigen::Vector3d> next_level(points.size() - 1);
    for (size_t i = 0; i < points.size() - 1; ++i) {
        next_level[i] = (1.0 - t) * points[i] + t * points[i + 1];
    }
    return deCasteljau(next_level, t);
}

Eigen::Vector3d BezierCurve::evaluate(double t) const {
    t = std::clamp(t, 0.0, 1.0);
    return deCasteljau(control_points_, t);
}

Eigen::Vector3d BezierCurve::derivative(double t) const {
    t = std::clamp(t, 0.0, 1.0);
    int n = order();
    if (n == 0) {
        return Eigen::Vector3d::Zero();
    }
    std::vector<Eigen::Vector3d> deriv_points(n);
    for (int i = 0; i < n; ++i) {
        deriv_points[i] = static_cast<double>(n) * (control_points_[i + 1] - control_points_[i]);
    }
    if (deriv_points.size() == 1) {
        return deriv_points[0];
    }
    return deCasteljau(deriv_points, t);
}

Eigen::Vector3d BezierCurve::secondDerivative(double t) const {
    t = std::clamp(t, 0.0, 1.0);
    int n = order();
    if (n < 2) {
        return Eigen::Vector3d::Zero();
    }
    std::vector<Eigen::Vector3d> first_deriv(n);
    for (int i = 0; i < n; ++i) {
        first_deriv[i] = static_cast<double>(n) * (control_points_[i + 1] - control_points_[i]);
    }
    int n2 = n - 1;
    std::vector<Eigen::Vector3d> second_deriv(n2);
    for (int i = 0; i < n2; ++i) {
        second_deriv[i] = static_cast<double>(n2) * (first_deriv[i + 1] - first_deriv[i]);
    }
    if (second_deriv.size() == 1) {
        return second_deriv[0];
    }
    return deCasteljau(second_deriv, t);
}

std::vector<Eigen::Vector3d> BezierCurve::sample(int num_samples) const {
    std::vector<Eigen::Vector3d> samples;
    samples.reserve(num_samples + 1);
    for (int i = 0; i <= num_samples; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(num_samples);
        samples.push_back(evaluate(t));
    }
    return samples;
}

double BezierCurve::arcLength(int num_segments) const {
    double length = 0.0;
    Eigen::Vector3d prev = evaluate(0.0);
    for (int i = 1; i <= num_segments; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(num_segments);
        Eigen::Vector3d curr = evaluate(t);
        length += (curr - prev).norm();
        prev = curr;
    }
    return length;
}

double BezierCurve::findParameterByArcLength(double target_length, int num_segments) const {
    double total = arcLength(num_segments);
    if (target_length <= 0.0) return 0.0;
    if (target_length >= total) return 1.0;

    double accumulated = 0.0;
    Eigen::Vector3d prev = evaluate(0.0);
    for (int i = 1; i <= num_segments; ++i) {
        double t = static_cast<double>(i) / static_cast<double>(num_segments);
        Eigen::Vector3d curr = evaluate(t);
        double seg_len = (curr - prev).norm();
        if (accumulated + seg_len >= target_length) {
            double ratio = (target_length - accumulated) / seg_len;
            double t_prev = static_cast<double>(i - 1) / static_cast<double>(num_segments);
            return t_prev + ratio / static_cast<double>(num_segments);
        }
        accumulated += seg_len;
        prev = curr;
    }
    return 1.0;
}

Eigen::Vector3d BezierCurve::startPoint() const {
    return control_points_.front();
}

Eigen::Vector3d BezierCurve::endPoint() const {
    return control_points_.back();
}

BezierCurve BezierCurve::elevateDegree() const {
    int n = order();
    std::vector<Eigen::Vector3d> new_points(n + 2);
    new_points[0] = control_points_[0];
    new_points[n + 1] = control_points_[n];
    for (int i = 1; i <= n; ++i) {
        double alpha = static_cast<double>(i) / static_cast<double>(n + 1);
        new_points[i] = (1.0 - alpha) * control_points_[i] + alpha * control_points_[i - 1];
    }
    return BezierCurve(new_points);
}

void BezierCurve::setControlPoint(int index, const Eigen::Vector3d& point) {
    if (index < 0 || index >= static_cast<int>(control_points_.size())) {
        throw std::out_of_range("Control point index out of range");
    }
    control_points_[index] = point;
}

double BezierCurve::minZ() const {
    double min_z = control_points_[0].z();
    for (int i = 1; i < numControlPoints(); ++i) {
        min_z = std::min(min_z, control_points_[i].z());
    }
    return min_z;
}

double BezierCurve::maxZ() const {
    double max_z = control_points_[0].z();
    for (int i = 1; i < numControlPoints(); ++i) {
        max_z = std::max(max_z, control_points_[i].z());
    }
    return max_z;
}

bool BezierCurve::hasGroundPenetration(double ground_height) const {
    auto samples = sample(50);
    for (const auto& p : samples) {
        if (p.z() < ground_height) {
            return true;
        }
    }
    return false;
}

}
