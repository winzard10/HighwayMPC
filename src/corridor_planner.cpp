/// ------------------------------------------------ ///
/// Obstacle Avoidance Module. (corridor_planner.cpp) ///
/// ------------------------------------------------ ///

#include "corridor_planner.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace corridor {

// -----------------------------------------------------------------------------
// buildRawBounds
//
// Build raw lateral bounds [lo, up] in Frenet ey-space by sweeping obstacles
// along the reference path. Each obstacle is modeled as a disk (radius +
// margin + half vehicle track). Bounds are lane-aware:
//
//   - p_ref_h[k]  : reference (x,y) at step k
//   - psi_ref_h[k]: reference heading at step k
//   - lane_w_h[k] : lane width at step k
//   - lane_ref_h[k]: which lane this segment belongs to (Left/Right/Center)
//   - t0, dt      : current time and horizon step
//   - obstacles   : obstacle set (with ey_obs already filled externally)
//   - track_w     : vehicle track width
//   - margin      : extra lateral safety margin
//   - L_look      : along-track look-ahead distance for obstacles
//
// Inputs:
//   lo, up must be pre-initialized with lane-based bounds before calling.
//   This function *tightens* lo[k]/up[k] based on obstacles.
// -----------------------------------------------------------------------------
void buildRawBounds(
    const std::vector<Eigen::Vector2d>& p_ref_h,
    const std::vector<double>&          psi_ref_h,
    const std::vector<double>&          lane_w_h,
    const std::vector<CenterlineMap::LaneRef>& lane_ref_h,
    double t0, double dt,
    const Obstacles& obstacles,
    double track_w,
    std::vector<double>& lo, std::vector<double>& up) {

  const int N = static_cast<int>(p_ref_h.size());
  CorridorParams cor_params;
  double margin = cor_params.margin;
  double L_look = cor_params.L_look;

  for (int k = 0; k < N; ++k) {
    const double tk  = t0 + (k + 1) * dt;  // look at time at end of step k
    const double psi = psi_ref_h[k];
    const double tx  = std::cos(psi), ty = std::sin(psi); // tangential direction

    // Store original bounds to use as references for adjustments
    const double lo_k = lo[k];
    const double up_k = up[k];

    // Loop over all obstacles active at time tk
    for (const auto& obs : obstacles.active_at(tk)) {
      // Vector from reference point to obstacle in world frame
      const double dx = obs.x - p_ref_h[k].x();
      const double dy = obs.y - p_ref_h[k].y();
      const double ds = dx * tx + dy * ty;               // projection along lane

      // Only consider obstacles within look-ahead window
      if (ds < -L_look/3.0 || ds > L_look) continue;

      // Lateral position of obstacle relative to reference centerline, precomputed
      double ey_obs = obstacles.items[obs.idx].ey_obs;

      // Lane-aware corridor clipping logic:
      //   - Right lane: constrain around lane center at positive ey
      //   - Left lane : constrain around lane center at negative ey
      //   - Center    : symmetric around ey=0
      if (lane_ref_h[k] == CenterlineMap::LaneRef::Right) {
        // Right lane: centerline is shifted to +lane_w/2
        if (ey_obs <= 0.0) {
          // Obstacle lies "inside" or near centerline: tighten lower bound
          lo[k] = std::max(lo[k],
                           lo_k + lane_w_h[k]/2.0 +
                           (ey_obs + obs.radius + track_w/2.0 + margin));
        } else {
          // Obstacle lies on the right side of centerline
          if (ey_obs >= lane_w_h[k]/2.0) {
            // Very far right: clip upper bound near it
            up[k] = std::min(up[k],
                             lo_k + lane_w_h[k]/2.0 +
                             (ey_obs - obs.radius - track_w/2.0 - margin));
          } else if (lane_w_h[k]/2.0 >
                     -(ey_obs - obs.radius - track_w - margin)) {
            // Obstacle intersects lane area more centrally: use a different cut
            up[k] = std::min(up[k],
                             up_k - 3.0*lane_w_h[k]/2.0 +
                             (ey_obs - obs.radius - track_w/2.0 - margin));
          } else {
            // Fallback: treat as inner obstacle and adjust lower bound
            lo[k] = std::max(lo[k],
                             lo_k + lane_w_h[k]/2.0 +
                             (ey_obs + obs.radius + track_w/2.0 + margin));
          }
        }

      } else if (lane_ref_h[k] == CenterlineMap::LaneRef::Left) {
        // Left lane: centerline is shifted to -lane_w/2
        if (ey_obs >= 0.0)  {
          // Obstacle on the right side, tighten upper bound
          up[k] = std::min(up[k],
                           up_k - lane_w_h[k]/2 +
                           (ey_obs - obs.radius - track_w/2.0 - margin));
        } else { 
          // Obstacle on left side of origin
          if (ey_obs <= -lane_w_h[k]/2.0) {
            // Far left: tighten lower bound
            lo[k] = std::max(lo[k],
                             up_k - lane_w_h[k]/2.0 +
                             (ey_obs + obs.radius + track_w/2.0 + margin));
          }
          else if (lane_w_h[k]/2.0 > ey_obs + obs.radius + track_w + margin) {
            // More central intersection with the lane
            lo[k] = std::max(lo[k],
                             lo_k + 3.0*lane_w_h[k]/2.0 +
                             (ey_obs + obs.radius + track_w/2.0 + margin));
          }
          else {
            // Fallback: tighten upper bound
            up[k] = std::min(up[k],
                             up_k - lane_w_h[k]/2.0 +
                             (ey_obs - obs.radius - track_w/2.0 - margin));
          }
        }
      } else { 
        // Center lane: symmetric around ey = 0
        if (ey_obs >= 0.0)
          up[k] = std::min(up[k],
                           up_k + (ey_obs - obs.radius - track_w/2.0 - margin));
        else
          lo[k] = std::max(lo[k],
                           lo_k + (ey_obs + obs.radius + track_w/2.0 + margin));
      }
    }

    // Safety clamp: if lo exceeds up, collapse to midpoint
    if (lo[k] > up[k]) {
      const double mid = 0.5 * (lo[k] + up[k]);
      lo[k] = up[k] = mid;
    }
  }
}

// -----------------------------------------------------------------------------
// smoothBounds
//
// Simple temporal smoothing on lo/up via a slope (slew-rate) limit.
//
// We perform a forward and backward pass enforcing:
//
//   lo[k] >= lo[k-1] - slope_per_step
//   up[k] <= up[k-1] + slope_per_step
//
// This prevents extremely sharp corridor steps across the horizon.
// -----------------------------------------------------------------------------
static inline void smoothBounds(std::vector<double>& lo,
                                std::vector<double>& up,
                                double slope_per_step) {
  const int N = static_cast<int>(lo.size());

  // forward pass
  for (int k = 1; k < N; ++k) {
    lo[k] = std::max(lo[k], lo[k - 1] - slope_per_step);
    up[k] = std::min(up[k], up[k - 1] + slope_per_step);
    if (lo[k] > up[k]) {
      const double m = 0.5 * (lo[k] + up[k]);
      lo[k] = up[k] = m;
    }
  }
  // backward pass
  for (int k = N - 2; k >= 0; --k) {
    lo[k] = std::max(lo[k], lo[k + 1] - slope_per_step);
    up[k] = std::min(up[k], up[k + 1] + slope_per_step);
    if (lo[k] > up[k]) {
      const double m = 0.5 * (lo[k] + up[k]);
      lo[k] = up[k] = m;
    }
  }
}

// -----------------------------------------------------------------------------
// segmentInside
//
// Check if the line segment in ey-space between (i, ey_i) and (j, ey_j)
// stays entirely within the [lo[m], up[m]] interval for all m ∈ [i, j].
//
// We linearly interpolate ey(m) and test against lo/up + small epsilon.
// Used as feasibility test for edges in the DP graph.
// -----------------------------------------------------------------------------
static inline bool segmentInside(const std::vector<double>& lo,
                                 const std::vector<double>& up,
                                 int i, int j, double ey_i, double ey_j) {
  if (j <= i) return false;
  const int len = j - i;
  for (int m = i; m <= j; ++m) {
    const double tau = static_cast<double>(m - i) / static_cast<double>(len);
    const double ey  = (1.0 - tau) * ey_i + tau * ey_j;
    if (ey < lo[m] - 1e-6 || ey > up[m] + 1e-6) return false;
  }
  return true;
}

// -----------------------------------------------------------------------------
// adaptEyRefPWA
//
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
    double ey_hat_k)
{
    const int N = static_cast<int>(ey_road.size()) - 1; // j = 0..N
    std::vector<double> ey_ref_pwa(N + 1);

    if (N <= 3) {
        // Degenerate short-horizon case: just linear interp from ey_hat_k to ey_road[j]
        for (int j = 0; j <= N; ++j) {
            double alpha = (N == 0) ? 0.0 : static_cast<double>(j) / N;
            ey_ref_pwa[j] = (1.0 - alpha) * ey_hat_k + alpha * ey_road[j];
        }
        return ey_ref_pwa;
    }

    for (int j = 0; j <= N; ++j) {
        const double e_road_j = ey_road[j];

        if (j <= 3) {
            // First segment:  hat{e}_y,k + ((e_road - hat{e}_y,k)/2) * j/3
            double alpha = static_cast<double>(j) / 3.0;
            ey_ref_pwa[j] =
                ey_hat_k + 0.5 * (e_road_j - ey_hat_k) * alpha;
        } else {
            // Second segment:
            // (hat{e}_y,k + e_road)/2 + ((e_road - hat{e}_y,k)/2) * (j-3)/(N-3)
            double mid   = 0.5 * (ey_hat_k + e_road_j);
            double alpha = static_cast<double>(j - 3) / static_cast<double>(N - 3);
            ey_ref_pwa[j] =
                mid + 0.5 * (e_road_j - ey_hat_k) * alpha;
        }
    }

    return ey_ref_pwa;
}

// -----------------------------------------------------------------------------
// planGraph
//
// Plan a smooth ey_ref path within lateral bounds using a layered-graph DP.
//
// Inputs:
//   s_grid      : centerline s at each step (monotone, ~arc length)
//   lo_raw,up_raw: raw lateral bounds (from buildRawBounds or lane edges)
//   slope_per_step: max per-step change of lo/up during smoothing
//
// Steps:
//   1) Smooth bounds in time (lo, up).
//   2) Define 3 candidate ey values at each step: {lo, mid, up}.
//   3) Run DP over (k,c) nodes:
//        - edges connect (k,c) → (kp,cp) if straight segment stays inside [lo,up]
//        - cost = accumulated |dtheta|, where dtheta ~ atan((ey_{kp}-ey_k)/ds)
//   4) Backtrack best terminal node to get ey_ref[k] path.
//
// Output:
//   Output{ lo, up, ey_ref } with smoothed bounds and chosen center path.
// -----------------------------------------------------------------------------
Output planGraph(
    const std::vector<double>& s_grid,     // centerline s per step (monotone)
    const std::vector<double>& lo_raw,
    const std::vector<double>& up_raw) {

  const int N = static_cast<int>(s_grid.size());
  CorridorParams cor_params;
  double slope_per_step = cor_params.slope;

  // 1) smooth the bounds
  std::vector<double> lo = lo_raw, up = up_raw;
  smoothBounds(lo, up, slope_per_step);

  // 2) layered graph over {lo, mid, up}
  auto mid = [&](int k) { return 0.5 * (lo[k] + up[k]); };
  constexpr int C = 3; // candidates: 0=lo, 1=mid, 2=up

  // eyOf(k,c) gives candidate ey at step k and candidate index c
  auto eyOf = [&](int k, int c) -> double {
    return (c == 0) ? lo[k] : ((c == 1) ? mid(k) : up[k]);
  };

  struct Node { double cost; int prev_k; int prev_c; };
  std::vector<std::array<Node, C>> best(N);

  // Initialize first step with zero cost at all candidates
  for (int c = 0; c < C; ++c) best[0][c] = {0.0, -1, -1};

  // DP over horizon
  for (int k = 0; k < N - 1; ++k) {
    for (int c = 0; c < C; ++c) {
      const double ey_k = eyOf(k, c);
      const double base = best[k][c].cost;
      if (!std::isfinite(base)) continue;

      // Try jumping to any future step kp > k (not only k+1)
      for (int kp = k + 1; kp < N; ++kp) {
        const double ds = std::max(1e-3, s_grid[kp] - s_grid[k]);
        for (int cp = 0; cp < C; ++cp) {
          const double ey_kp = eyOf(kp, cp);
          // Ensure the segment stays inside bounds
          if (!segmentInside(lo, up, k, kp, ey_k, ey_kp)) continue;

          // Approximate heading change as atan(dy/ds) and accumulate abs
          const double dtheta = std::atan2(ey_kp - ey_k, ds);  // heading change proxy
          const double cost   = base + std::abs(dtheta);

          // Update best candidate for (kp,cp) if cheaper
          if (best[kp][cp].prev_k < 0 || cost < best[kp][cp].cost) {
            best[kp][cp] = {cost, k, c};
          }
        }
      }
    }
  }

  // 3) pick best terminal & backtrack ey_ref
  int kf = N - 1, cf = 0;
  for (int c = 1; c < C; ++c)
    if (best[kf][c].prev_k >= 0 && best[kf][c].cost < best[kf][cf].cost) cf = c;

  std::vector<double> ey_ref(N);
  int k = kf, c = cf;
  while (k >= 0) {
    ey_ref[k] = eyOf(k, c);
    const int pk = best[k][c].prev_k, pc = best[k][c].prev_c;
    if (pk < 0) break;
    k = pk; c = pc;
  }
  // If backtracking doesn't reach index 0 (shouldn't normally happen),
  // fill leading entries with first valid ey_ref as a safety fallback.
  for (int i = 0; i < k; ++i) ey_ref[i] = ey_ref[k];

  return {std::move(lo), std::move(up), std::move(ey_ref)};
}

} // namespace corridor
