#include "vehicle_model.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace dynamics::vehicle {

static inline double clamp(double u,double lo,double hi){
  return std::min(std::max(u,lo),hi);
}
static inline double wrapAngle(double a){
  while(a>M_PI)a-=2*M_PI; while(a<=-M_PI)a+=2*M_PI; return a;
}

State step(const State& s, const Control& u,
           const Params& vp, const Limits& lim)
{
  State n = s;
  const double dt = vp.dt;
  
  // clamp inputs
  const double R_cmd   = clamp(u.R,      lim.R_min, lim.R_max);
  const double ddelta  = clamp(u.ddelta, -lim.ddelta_max, lim.ddelta_max);
  
  // advance steering
  n.delta = clamp(s.delta + ddelta * dt, -lim.delta_max, lim.delta_max);
  
  // Equation (6): R = F_wind + m₀(tan γ / cos² γ)σ̇ + (m + m₀ tan² γ)u
  // Solve for u (longitudinal acceleration):
  // u = (R - F_wind - m₀(tan γ / cos² γ)σ̇) / (m + m₀ tan² γ)
  
  const double c  = std::cos(s.delta);
  const double s2 = 1.0 / (c * c);            // sec²(δ)
  const double t  = std::tan(s.delta);
  
  // For numerical stability, limit tan(delta)
  // const double t_safe = clamp(t, -3.0, 3.0);
  
  const double denom = (vp.m + vp.m0 * t * t);
  
  // Coupling term: m₀ * tan(δ) * sec²(δ) * δ̇ * σ
  const double coupling = (vp.m0 * t * s2) * ddelta * s.v;
  
  // F_wind = 0 in your case
  const double F_wind = 0.0;
  
  // Solve for longitudinal acceleration from Eq. (6)
  const double a_eff = (R_cmd - F_wind - coupling) / denom;
  
  // Safety check for extreme accelerations
  // const double a_clamped = clamp(a_eff, -10.0, 10.0);  // reasonable limits
  
  if (std::abs(a_eff) > 15.0) {
    std::cerr << "WARNING: Extreme acceleration " << a_eff 
              << " m/s² (R=" << R_cmd << " N, delta=" << s.delta 
              << " rad, v=" << s.v << " m/s)\n";
  }
  
  // integrate longitudinal velocity (Eq. 5: σ̇ = u)
  n.v = std::max(0.0, s.v + a_eff * dt);
  
  // Kinematic bicycle model for pose (Eq. 5)
  // Use CURRENT state values for consistent integration (forward Euler)
  // ẋ = σ cos ψ
  // ẏ = σ sin ψ  
  // ψ̇ = (tan δ / L) σ
  
  // Update pose using OLD velocity and steering (consistent with working version)
  n.x   = s.x + s.v * std::cos(s.psi) * dt;
  n.y   = s.y + s.v * std::sin(s.psi) * dt;
  n.psi = wrapAngle(s.psi + (s.v / vp.L) * std::tan(s.delta) * dt);
  
  // Update dpsi for logging (using new state)
  n.dpsi = (n.v / vp.L) * std::tan(n.delta);
  
  // Longitudinal distance
  n.s = s.s + n.v * dt;
  
  return n;
}

} // namespace dynamics::vehicle