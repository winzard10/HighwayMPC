/// --------------------------------------- /// 
/// Vehicle Parameters. (vehicle_model.hpp)  ///
/// --------------------------------------- ///

#pragma once
#include "tire_model.hpp"

namespace dynamics::vehicle {

// -----------------------------------------------------------------------------
// Vehicle physical parameters
// -----------------------------------------------------------------------------
struct Params {
  double m{660.0};      // sprung mass [kg]
  double L{3.4};        // wheelbase [m]
  double d{1.6};        // CG distance from rear axle [m]
  double JG{2500.0};    // yaw moment of inertia about CG [kg m^2]
  double track_w{1.7};  // track width [m]

  double dt{0.05};      // integration time step [s]
};

// -----------------------------------------------------------------------------
// Actuation and force limits
// -----------------------------------------------------------------------------
struct Limits {
  double delta_max{0.5};    // max steering angle [rad]
  double ddelta_max{0.7};   // max steering rate [rad/s]

  double R_min{-5000.0};    // min rear-axle longitudinal "force" (drive/brake) [N]
  double R_max{ 5000.0};    // max rear-axle longitudinal "force" [N]

  // Optional individual wheel force limits (front-left / rear-left)
  double Ffl_max{5000.0};
  double Frl_max{5500.0};
};

// -----------------------------------------------------------------------------
// Vehicle state in both world and body frames
// -----------------------------------------------------------------------------
struct State {
  double s{0.0};                    // path coordinate along reference [m]
  double x{0.0}, y{0.0}, psi{0.0};  // global pose (x,y, heading psi)

  double vx{0.0}, vy{0.0}, dpsi{0.0};  // body-frame velocities and yaw rate
  double delta{0.0};                   // front steering angle [rad]

  // Derived accelerations (body frame + yaw)
  double ax{0.0}, ay{0.0}, ddpsi{0.0};

  // Logging utilities for jerk calculation
  double ax_prev{0.0};  // previous longitudinal acceleration
  double jerk{0.0};     // estimated longitudinal jerk
};

// -----------------------------------------------------------------------------
// Control inputs
// -----------------------------------------------------------------------------
struct Control {
  double R{0.0};       // rear-axle longitudinal "force" command [N]
  double ddelta{0.0};  // steering rate command [rad/s]
};

// -----------------------------------------------------------------------------
// step()
// Advance one simulation step using the dynamic bicycle model and tire forces.
// -----------------------------------------------------------------------------
State step(const State& s, const Control& u,
           const Params& vp, const Limits& lim,
           const dynamics::tire::TireParams& tp);

} // namespace dynamics::vehicle
