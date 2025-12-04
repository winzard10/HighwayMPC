/// ------------------------------------- /// 
/// Vehicle Dynamics (vehicle_model.cpp)  ///
/// ------------------------------------- ///

#include "vehicle_model.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace dynamics::vehicle {

// Clamp helper: saturate u to [lo, hi]
static inline double clamp(double u,double lo,double hi){
  return std::min(std::max(u,lo),hi);
}

// Wrap angle into (-pi, pi]
static inline double wrapAngle(double a){
  while(a>M_PI)a-=2*M_PI; while(a<=-M_PI)a+=2*M_PI; return a;
}

// -----------------------------------------------------------------------------
// step()
// One integration step of the dynamic bicycle model with tire forces.
//
// Inputs:
//   s   : current vehicle state
//   u   : control input (rear-longitudinal "force" R, steering rate ddelta)
//   vp  : vehicle parameters (mass, wheelbase, CG offset, inertia, dt, etc.)
//   lim : actuator limits (R_min/max, delta_max, ddelta_max, etc.)
//   tp  : tire parameters (Magic Formula + unsprung masses)
//
// Returns:
//   n   : next state at t + dt
//
// Model:
//   - Steering angle is updated with rate input ddelta and clamped.
//   - Tire forces are computed in the body frame using computeForcesBody().
//   - Longitudinal/lateral accelerations and yaw acceleration are derived
//     from forces and moments.
//   - Linear Euler integration is used for vx, vy, yaw rate, and position.
//   - Heading psi is wrapped to (-pi, pi].
//   - Path coordinate s is progressed by forward speed (vx >= 0).
// -----------------------------------------------------------------------------
State step(const State& s, const Control& u,
           const Params& vp, const Limits& lim,
           const dynamics::tire::TireParams& tp)
{
  State n = s;
  const double dt = vp.dt;

  // Clamp commanded rear force and steering rate to actuator limits
  double R_cmd  = clamp(u.R,      lim.R_min, lim.R_max);
  double ddel   = clamp(u.ddelta, -lim.ddelta_max, lim.ddelta_max);

  // Integrate steering angle and clamp to steering limits
  n.delta = clamp(s.delta + ddel*dt, -lim.delta_max, lim.delta_max);

  // Geometry + inertial properties for tire model
  dynamics::tire::VehicleGeom vg{vp.m, vp.L, vp.d, vp.JG};
  const double g = 9.81;

  // Compute tire forces and yaw moment in the body frame
  const auto fr = dynamics::tire::computeForcesBody(
      s.vx, s.vy, s.dpsi, n.delta, R_cmd, vg, tp, g);

  // Total mass accounts for sprung + unsprung masses
  double m_tot = vp.m + tp.m_unsprung_front + tp.m_unsprung_rear;

  // Rigid-body accelerations in body frame:
  //   ax = Fx/m + r*vy
  //   ay = Fy/m - r*vx
  //   ddpsi = Mz / JG
  n.ax   = fr.Fx_sum / m_tot + s.dpsi * s.vy;
  n.ay   = fr.Fy_sum / m_tot - s.dpsi * s.vx;
  n.ddpsi= fr.Mz     / vp.JG;

  // Integrate velocities and yaw rate (Euler)
  n.vx   = std::max(0.0, s.vx + n.ax*dt); // prevent negative forward speed
  n.vy   =              s.vy + n.ay*dt;
  n.dpsi =              s.dpsi + n.ddpsi*dt;

  // World-frame velocity from body-frame (vx,vy) and heading psi
  const double c = std::cos(n.psi), d = std::sin(n.psi);
  const double xdot = n.vx*c - n.vy*d;
  const double ydot = n.vx*d + n.vy*c;

  // Integrate pose
  n.x   = s.x + xdot*dt;
  n.y   = s.y + ydot*dt;
  n.psi = wrapAngle(s.psi + n.dpsi*dt);

  // Integrate path coordinate s along forward motion (vx >= 0)
  n.s   = s.s + std::max(0.0, n.vx)*dt;

  return n;
}

} // namespace dynamics::vehicle
