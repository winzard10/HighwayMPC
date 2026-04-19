/// ----------------------------------- /// 
/// Obstacle Parameters. (obstacles.hpp) ///
/// ----------------------------------- ///

#pragma once
#include <vector>
#include <string>
#include <optional>

// -----------------------------
// Obstacle CSV model & queries
// -----------------------------
struct Obstacle {
    int    id = -1;

    // Obstacle type:
    //   Static: position fixed at (x0, y0)
    //   Moving: position evolves linearly with (vx, vy)
    enum class Kind { Static, Moving } kind = Kind::Static;

    double x0 = 0.0, y0 = 0.0;          // initial position [m]
    double vx = 0.0, vy = 0.0;          // velocity if moving [m/s]
    double radius = 0.5;                // safety radius [m]
    double t_start = 0.0, t_end = 1e9;  // active time window [s]

public:
    // Lateral offset from centerline at current time [m].
    // This field is intended to be populated by higher-level code
    // that projects (x,y) onto the reference path / Frenet frame.
    double ey_obs = 0.0;
};

struct Obstacles {
    // Load from CSV with header:
    //   id,kind,x0,y0,vx,vy,radius,t_start,t_end   (kind ∈ {static,moving})
    //
    // Clears existing items, then fills from file.
    bool load_csv(const std::string& path);

    // Position at time t (for moving obstacles).
    // Returns std::nullopt if t is outside [t_start, t_end].
    std::optional<std::pair<double,double>> position_of(const Obstacle& ob, double t) const;

    // Get list of active obstacles at time t with their positions.
    // Active entry exposes:
    //   idx     : index into `items`
    //   id      : obstacle ID
    //   x, y    : world-frame position at time t
    //   radius  : obstacle radius
    //   ey_obs  : lateral offset from centerline (if set)
    struct Active { size_t idx; int id; double x; double y; double radius; double ey_obs; };
    std::vector<Active> active_at(double t) const;

    // Raw obstacle list loaded from CSV
    std::vector<Obstacle> items;
};

// ---------------------------------------------
// MPC-facing half-spaces and ey bound utility
// ---------------------------------------------
// Each inequality encodes a half-space in lateral error:
//   a * ey >= b
// These can be combined to form polygonal forbidden regions.
struct ObsIneq { double a; double b; };

// Set of obstacle-induced constraints over the MPC horizon.
// obs[k] holds the list of half-spaces to apply at stage k.
struct MPCObsSet {
    // obs[k] = list of half-spaces to apply at step k (k = 0..N-1)
    std::vector<std::vector<ObsIneq>> obs;
};
