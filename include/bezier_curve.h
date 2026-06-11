#pragma once

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace quadruped {

class BezierCurve {
public:
    BezierCurve() = default;

    explicit BezierCurve(const std::vector<Eigen::Vector3d>& control_points);

    static BezierCurve cubic(
        const Eigen::Vector3d& p0,
        const Eigen::Vector3d& p1,
        const Eigen::Vector3d& p2,
        const Eigen::Vector3d& p3
    );

    static BezierCurve quintic(
        const Eigen::Vector3d& p0,
        const Eigen::Vector3d& p1,
        const Eigen::Vector3d& p2,
        const Eigen::Vector3d& p3,
        const Eigen::Vector3d& p4,
        const Eigen::Vector3d& p5
    );

    Eigen::Vector3d evaluate(double t) const;

    Eigen::Vector3d derivative(double t) const;

    Eigen::Vector3d secondDerivative(double t) const;

    std::vector<Eigen::Vector3d> sample(int num_samples) const;

    double arcLength(int num_segments = 100) const;

    double findParameterByArcLength(double target_length, int num_segments = 200) const;

    const std::vector<Eigen::Vector3d>& controlPoints() const { return control_points_; }

    int order() const { return static_cast<int>(control_points_.size()) - 1; }

    int numControlPoints() const { return static_cast<int>(control_points_.size()); }

    Eigen::Vector3d startPoint() const;

    Eigen::Vector3d endPoint() const;

    BezierCurve elevateDegree() const;

    void setControlPoint(int index, const Eigen::Vector3d& point);

    double minZ() const;

    double maxZ() const;

    bool hasGroundPenetration(double ground_height = 0.0) const;

private:
    std::vector<Eigen::Vector3d> control_points_;

    Eigen::Vector3d deCasteljau(const std::vector<Eigen::Vector3d>& points, double t) const;

    std::vector<Eigen::Vector3d> deCasteljauDerivativePoints(
        const std::vector<Eigen::Vector3d>& points, double t) const;
};

}
