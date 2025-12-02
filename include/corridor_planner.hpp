/// ---------------------------------------------- ///
/// Corridor Planner Module (corridor_planner.hpp) ///
/// ---------------------------------------------- ///

#pragma once
#include <vector>
#include <Eigen/Dense>
#include "centerline_map.hpp"
#include "obstacles.hpp"

namespace corridor {

// Result of corridor planning:
//   - lo, up : smoothed lateral corridor bounds
//   - ey_ref : chosen center path (reference lateral offset) within these bounds
struct Output {
  std::vector<double> lo, up;    // corridor bounds per step
  std::vector<double> ey_ref;    // chosen center/path per step
};

// -----------------------------------------------------------------------------
// buildRawBounds
//
// Construct raw lateral bounds [lo, up] along the reference path by taking
// lane boundaries and subtracting obstacle footprints (disks) inflated by:
//
//   - margin: extra safety buffer
//   - track_w: half-vehicle width for collision-free corridor
//   - L_look: along-track look distance
//
// `lo` and `up` are passed by reference and modified in-place.
// -----------------------------------------------------------------------------
void buildRawBounds(
  const std::vector<Eigen::Vector2d>& p_ref_h,                // reference (x,y)
  const std::vector<double>&          psi_ref_h,              // heading along ref
  const std::vector<double>&          lane_w_h,               // lane width per step
  const std::vector<CenterlineMap::LaneRef>& lane_ref_h,      // lane type per step
  double t0, double dt,                                       // current time, step
  const Obstacles& obstacles,                                 // obstacle set
  double track_w, double margin, double L_look,               // corridor padding
  std::vector<double>& lo, std::vector<double>& up);

// -----------------------------------------------------------------------------
// planGraph
//
// Given raw bounds [lo_raw, up_raw] and an s-grid (centerline arc-length),
// plan a smooth ey_ref within those bounds using a layered graph DP built over
// three candidates per step: {lo, mid, up}.
//
// slope_per_step controls temporal smoothing of lo_raw/up_raw before planning.
// -----------------------------------------------------------------------------
Output planGraph(const std::vector<double>& s_grid,
  const std::vector<double>& lo_raw,
  const std::vector<double>& up_raw,
  double slope_per_step);

} // namespace corridor
