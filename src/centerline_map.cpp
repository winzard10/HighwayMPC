/// -------------------------------------------------- ///
/// Centerline Map Implementation (centerline_map.cpp) ///
/// -------------------------------------------------- ///

#include "centerline_map.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <iostream>

// Clamp x into [a, b]
static inline double clamp(double x, double a, double b) {
    return std::min(std::max(x, a), b);
}

// -----------------------------------------------------------------------------
// CenterlineMap::load_csv
//
// Load a centerline / lane geometry CSV with columns:
//
//   s,
//   x_right, y_right,
//   x_left,  y_left,
//   psi_center, kappa_center, v_ref,
//   x_center, y_center, lane_width,
//   x_left_border, y_left_border,
//   x_right_border, y_right_border
//
// The file must have at least one header line (ignored) and then
// at least two data rows. s must be strictly increasing.
//
// On success:
//   - fills internal vectors s_, xr_, yr_, xl_, yl_, psi_, kappa_, vref_,
//     xc_, yc_, lane_width_, xl_border_, yl_border_, xr_border_, yr_border_
//   - returns true
// On failure (I/O error, malformed data, or non-monotone s), returns false.
// -----------------------------------------------------------------------------
bool CenterlineMap::load_csv(const std::string& path) {
    s_.clear(); xr_.clear(); yr_.clear(); xl_.clear(); yl_.clear();
    psi_.clear(); kappa_.clear(); vref_.clear(); xc_.clear(); yc_.clear();
    lane_width_.clear();
    xl_border_.clear(); yl_border_.clear(); xr_border_.clear(); yr_border_.clear();

    std::ifstream fin(path);
    if (!fin) return false;

    std::string line;
    if (!std::getline(fin, line)) return false; // header

    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string cell; std::vector<double> v;

        // Parse comma-separated columns; empty cells become 0.0
        while (std::getline(ss, cell, ',')) v.push_back(cell.empty()? 0.0 : std::stod(cell));
        if (v.size() < 15) continue;

        // Map CSV columns into internal vectors
        s_.push_back(v[0]);  xr_.push_back(v[1]);  yr_.push_back(v[2]);
        xl_.push_back(v[3]); yl_.push_back(v[4]);
        psi_.push_back(v[5]); kappa_.push_back(v[6]); vref_.push_back(v[7]);
        xc_.push_back(v[8]); yc_.push_back(v[9]); lane_width_.push_back(v[10]);
        xl_border_.push_back(v[11]); yl_border_.push_back(v[12]);
        xr_border_.push_back(v[13]); yr_border_.push_back(v[14]);
    }
    if (s_.size() < 2) return false;

    // s must be strictly increasing (monotone)
    for (std::size_t i = 1; i < s_.size(); ++i)
        if (!(s_[i] > s_[i-1])) return false;

    return true;
}

// -----------------------------------------------------------------------------
// CenterlineMap::row
//
// Direct (index-based) access to stored row i, without interpolation.
// Assumes i is in range [0, size()).
// -----------------------------------------------------------------------------
CenterlineMap::Row CenterlineMap::row(std::size_t i) const {
    return {
        s_[i],
        xr_[i], yr_[i],
        xl_[i], yl_[i],
        psi_[i], kappa_[i], vref_[i],
        xc_[i], yc_[i], lane_width_[i],
        xl_border_[i], yl_border_[i],
        xr_border_[i], yr_border_[i]
    };
}

// -----------------------------------------------------------------------------
// CenterlineMap::upper_index
//
// Find index j such that:
//   - if s <= s_[0], return 1
//   - if s >= s_.back(), return size() - 1
//   - otherwise, upper_bound(s) over s_ and return its index
//
// This is used to locate the segment [s_[j-1], s_[j]] that contains s.
// -----------------------------------------------------------------------------
std::size_t CenterlineMap::upper_index(double s) const {
    if (s <= s_.front()) return 1;
    if (s >= s_.back())  return s_.size() - 1;
    auto it = std::upper_bound(s_.begin(), s_.end(), s);
    return std::distance(s_.begin(), it);
}

// -----------------------------------------------------------------------------
// CenterlineMap::sample
//
// Linear interpolation of map data at arclength s, clamped to [s_min, s_max].
//
// Steps:
//   1) Find segment [i, j] with s in [s_[i], s_[j]].
//   2) Compute t in [0,1].
//   3) Linearly interpolate each column between i and j.
// -----------------------------------------------------------------------------
CenterlineMap::Row CenterlineMap::sample(double s) const {
    std::size_t j = upper_index(s), i = j - 1;
    double s0 = s_[i], s1 = s_[j];
    double t = (clamp(s, s0, s1) - s0) / (s1 - s0);
    auto L = [t](double a, double b){ return a + t*(b-a); };

    return {
        clamp(s, s_.front(), s_.back()),      // interpolated s (clamped)
        L(xr_[i], xr_[j]),                    // x_right
        L(yr_[i], yr_[j]),                    // y_right
        L(xl_[i], xl_[j]),                    // x_left
        L(yl_[i], yl_[j]),                    // y_left
        L(psi_[i], psi_[j]),                  // psi_center
        L(kappa_[i], kappa_[j]),              // kappa_center
        L(vref_[i], vref_[j]),                // v_ref
        L(xc_[i], xc_[j]),                    // x_center
        L(yc_[i], yc_[j]),                    // y_center
        L(lane_width_[i], lane_width_[j]),    // lane_width
        L(xl_border_[i], xl_border_[j]),      // x_left_border
        L(yl_border_[i], yl_border_[j]),      // y_left_border
        L(xr_border_[i], xr_border_[j]),      // x_right_border
        L(yr_border_[i], yr_border_[j])       // y_right_border
    };
}

// -----------------------------------------------------------------------------
// Lane sampling helpers
//
// right_lane_at(s): pose on right lane centerline
// left_lane_at(s):  pose on left lane centerline
// center_at(s):     pose on midline between lanes
// -----------------------------------------------------------------------------
CenterlineMap::LanePose CenterlineMap::right_lane_at(double s) const {
    auto r = sample(s);
    return {r.x_right, r.y_right, r.psi_center, r.kappa_center, r.v_ref, r.lane_width};
}

CenterlineMap::LanePose CenterlineMap::left_lane_at(double s) const {
    auto r = sample(s);
    return {r.x_left, r.y_left, r.psi_center, r.kappa_center, r.v_ref, r.lane_width};
}

CenterlineMap::CenterPose CenterlineMap::center_at(double s) const {
    auto r = sample(s);
    return {r.x_center, r.y_center, r.psi_center, r.kappa_center, r.v_ref, r.lane_width};
}

// pick which left, right, center curve to project onto
using XYRef = std::pair<const std::vector<double>&, const std::vector<double>&>;

// Choose the (x,y) curve vectors to use for projection based on LaneRef
static inline XYRef pick_xy_curve(
    const std::vector<double>& xr, const std::vector<double>& xl, const std::vector<double>& xc,
    const std::vector<double>& yr, const std::vector<double>& yl, const std::vector<double>& yc,
    CenterlineMap::LaneRef which)
{
    switch (which) {
        case CenterlineMap::LaneRef::Right: return {xr, yr};
        case CenterlineMap::LaneRef::Left:  return {xl, yl};
        default:                            return {xc, yc};
    }
}

// -----------------------------------------------------------------------------
// CenterlineMap::project
//
// Project a world point (x, y) onto a chosen reference curve:
//
//   - which = Right  : use (x_right, y_right)
//   - which = Left   : use (x_left,  y_left)
//   - which = Center : use (x_center,y_center)
//
// Returns:
//   ProjectResult {
//     s_proj : arclength coordinate of closest point
//     ey     : signed lateral error (+ left of tangent)
//     psi_ref: tangent from road centerline at s_proj
//     x_ref  : x of closest point on curve
//     y_ref  : y of closest point on curve
//   }
//
// Algorithm:
//   1) Choose curve arrays xcurve, ycurve according to which.
//   2) If there are < 2 points, snap to first point as fallback.
//   3) Otherwise, search segments [i,i+1] in a local window around s:
//      - for each segment, orthogonally project (x,y) onto it,
//        clamp t ∈ [0,1], and keep the closest point in Euclidean distance.
//   4) Compute s_proj as linear interpolation of s_[i] and s_[i+1].
//   5) Use center_at(s_proj) to get psi_ref (smooth shared tangent).
//   6) Compute ey using normal (nx,ny) = (-sin(psi_ref), cos(psi_ref)).
// -----------------------------------------------------------------------------
CenterlineMap::ProjectResult CenterlineMap::project(double x, double y, LaneRef which) const {
    ProjectResult out;

    const auto& curve  = pick_xy_curve(xr_, xl_, xc_, yr_, yl_, yc_, which);
    const auto& xcurve = curve.first;
    const auto& ycurve = curve.second;

    const std::size_t n = s_.size();
    if (n < 2) {
        // Fallback: no segments to project onto; snap to closest vertex if present.
        out.s_proj  = n ? s_.front() : 0.0;
        out.x_ref   = n ? xcurve.front() : x;
        out.y_ref   = n ? ycurve.front() : y;
        out.psi_ref = center_at(out.s_proj).psi;
        double nx = -std::sin(out.psi_ref), ny = std::cos(out.psi_ref);
        out.ey = (x - out.x_ref)*nx + (y - out.y_ref)*ny;
        return out;
    }

    // initial window around x (using index j into s_)
    // NOTE: This uses upper_index(x) as a rough proxy, assuming s_ and x
    // are roughly aligned in this synthetic map.
    std::size_t j  = upper_index(x);                       // assumes this is consistent with s_/curve ordering
    std::size_t i0 = (j > 50) ? (j - 50) : 0;              // include segment [0,1]
    std::size_t i1 = std::min(n - 1, j + 50);              // last vertex index; last segment is [i1-1, i1]

    double best_d2 = 1e300;
    double best_s  = s_.front();
    double best_x  = xcurve.front();
    double best_y  = ycurve.front();

    // Search for closest point on curve within window [i0, i1-1]
    for (std::size_t i = i0; i < i1; ++i) {
        // segment endpoints (x from selected curve, y from selected curve)
        const double x0 = xcurve[i],     y0 = ycurve[i];
        const double x1 = xcurve[i + 1], y1 = ycurve[i + 1];

        const double dx = x1 - x0, dy = y1 - y0;
        const double seg2 = dx*dx + dy*dy;
        if (seg2 <= 1e-12) continue; // skip degenerate segment

        // projection parameter clamped to [0,1]
        double t = ((x - x0)*dx + (y - y0)*dy) / seg2;
        t = clamp(t, 0.0, 1.0);

        const double px = x0 + t*dx;
        const double py = y0 + t*dy;
        const double d2 = (x - px)*(x - px) + (y - py)*(y - py);

        if (d2 < best_d2) {
            best_d2 = d2;
            best_s  = s_[i] + t*(s_[i + 1] - s_[i]);
            best_x  = px;
            best_y  = py;
        }
    }

    // tangent & v_ref from road midline (shared smooth tangent)
    const auto c = center_at(best_s);

    out.s_proj  = best_s;
    out.x_ref   = best_x;
    out.y_ref   = best_y;
    out.psi_ref = c.psi;

    // signed lateral error (+ left of tangent)
    const double nx = -std::sin(out.psi_ref);
    const double ny =  std::cos(out.psi_ref);
    out.ey = (x - out.x_ref)*nx + (y - out.y_ref)*ny;

    return out;
}
