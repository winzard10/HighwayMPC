/// ------------------------------------------ ///
/// Centerline Map Header. (centerline_map.hpp) ///
/// ------------------------------------------ ///

#pragma once
#include <string>
#include <vector>
#include <cstddef>

/**
 * CenterlineMap parses and interpolates a highway two-lane CSV:
 *   s,x_right,y_right,x_left,y_left,psi_center,kappa_center,v_ref,
 *   x_center,y_center,lane_width,
 *   x_left_border,y_left_border,x_right_border,y_right_border
 *
 * Coordinates: x forward (≈ arclength s in this synthetic map), y left.
 * Units: meters, radians, 1/m, m/s.
 *
 * It provides:
 *  - Linear interpolation at arbitrary s
 *  - Lane center poses (Right / Left)
 *  - Road center (midline) pose
 *  - Projection of a world point (x,y) onto a chosen reference curve
 *    to compute Frenet-style errors (ey, epsi via psi_ref).
 */
class CenterlineMap {
public:
    // Which curve to use as the reference for projection / errors
    enum class LaneRef { Right, Left, Center };

    // Raw CSV row (interpolated or indexed)
    struct Row {
        double s{};
        double x_right{}, y_right{};
        double x_left{},  y_left{};
        double psi_center{};
        double kappa_center{};
        double v_ref{};
        double x_center{}, y_center{};
        double lane_width{};
        double x_left_border{},  y_left_border{};
        double x_right_border{}, y_right_border{};
    };

    // Pose sampled on a lane centerline
    struct LanePose {
        double x{};
        double y{};
        double psi{};   // tangent heading
        double kappa{}; // curvature
        double v_ref{}; // reference speed
        double lane_width{};
    };

    // Pose sampled on the road midline (between lanes)
    struct CenterPose {
        double x{};
        double y{};
        double psi{};
        double kappa{};
        double v_ref{};
        double lane_width{};
    };

    // Result of projecting a world point onto a chosen reference curve
    struct ProjectResult {
        double s_proj{};   // arclength at closest point on the chosen curve
        double ey{};       // signed lateral error (+ left of tangent)
        double psi_ref{};  // tangent heading at s_proj (from road centerline)
        double x_ref{};    // projection point (world)
        double y_ref{};
    };

    // ---- I/O ----
    // Load CSV file. Returns true on success.
    bool load_csv(const std::string& path);

    // ---- Introspection ----
    std::size_t size() const { return s_.size(); }
    bool ok() const { return !s_.empty(); }
    double s_min() const { return s_.empty() ? 0.0 : s_.front(); }
    double s_max() const { return s_.empty() ? 0.0 : s_.back(); }

    // ---- Sampling (by arclength s) ----
    Row      row(std::size_t i) const;     // direct indexed access (no bounds check)
    Row      sample(double s) const;       // linear interpolation (clamped to [s_min, s_max])
    LanePose right_lane_at(double s) const;
    LanePose left_lane_at(double s)  const;
    CenterPose center_at(double s)   const;

    // ---- Projection & Frenet-style errors ----
    // Project (x,y) onto the chosen reference curve (Right/Left/Center).
    // psi_ref is taken from the road midline at s_proj (smooth, shared tangent).
    ProjectResult project(double x, double y, LaneRef which = LaneRef::Center) const;

private:
    // CSV columns
    std::vector<double> s_;       // arclength / station
    std::vector<double> xr_, yr_; // right-lane centerline
    std::vector<double> xl_, yl_; // left-lane centerline
    std::vector<double> psi_;     // road tangent (shared across lanes)
    std::vector<double> kappa_;   // curvature at centerline
    std::vector<double> vref_;    // reference speed profile

    std::vector<double> xc_;      // midline x
    std::vector<double> yc_;      // midline y
    std::vector<double> lane_width_; // lane width (per lane or combined, depending on map)
    std::vector<double> xl_border_, yl_border_; // left road border
    std::vector<double> xr_border_, yr_border_; // right road border

    // utilities
    // Find j with s_[j-1] <= s < s_[j] (using upper_bound)
    std::size_t upper_index(double s) const;
};
