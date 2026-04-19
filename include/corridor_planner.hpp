/// ---------------------------------------------- ///
/// Corridor Planner Module. (corridor_planner.hpp) ///
/// ---------------------------------------------- ///

#pragma once
#include <vector>
#include <Eigen/Dense>
#include "centerline_map.hpp"
#include "obstacles.hpp"

namespace corridor {

// -----------------------------------------------------------------------------
// Parameters used by the corridor (lane) planning module.
// This struct configures how the planner interprets obstacle geometry and how
// aggressively it smooths the raw lateral bounds.
// -----------------------------------------------------------------------------
struct CorridorParams {
  double margin = 0.1;   // Safety buffer added around obstacles [m].
                         // A larger margin produces more conservative corridors.

  double L_look = 70.0;  // Forward look-ahead distance for detecting obstacles [m].
                         // Only obstacles within this range influence the bounds.

  double slope  = 0.05;  // Temporal smoothing factor applied to raw bounds
                         // (units: [m] per step). Higher values enforce smoother,
                         // slower changes in the corridor shape.
};

// -----------------------------------------------------------------------------
// Result container for the corridor-planning step.
//  - lo, up  : Smoothed lower/upper lateral limits of the driving corridor.
//  - ey_ref  : Selected centerline (lateral reference) used by downstream MPC.
// Each vector has size equal to the prediction horizon N.
// -----------------------------------------------------------------------------
struct Output {
  std::vector<double> lo;      // Lower lateral corridor boundary at each step.
  std::vector<double> up;      // Upper lateral corridor boundary at each step.
  std::vector<double> ey_ref;  // Chosen lateral reference e_y for each step.
};

// -----------------------------------------------------------------------------
// adaptEyRefPWA()
// Applies a piecewise-affine (PWA) adaptation rule to shift the reference
// lateral offset toward the road centerline while respecting corridor limits.
//
// Inputs:
//   ey_road  - Desired road center e_y profile (lane centerline).
//   ey_hat_k - Current estimated lateral position of the vehicle at step k.
//
// Output:
//   A modified lateral reference sequence ey_ref that blends road geometry
//   with a stabilizing PWA policy around the current vehicle offset.
// -----------------------------------------------------------------------------
std::vector<double> adaptEyRefPWA(
    const std::vector<double>& ey_road,
    double ey_hat_k);


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
  double track_w,                                             // track width
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
  const std::vector<double>& up_raw);

} // namespace corridor
