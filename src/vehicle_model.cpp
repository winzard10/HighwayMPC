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
           const Params& vp, const Limits& lim,
           const dynamics::tire::TireParams& tp)
{
  State n = s;
  const double dt = vp.dt;

  const double R_cmd  = clamp(u.R,      lim.R_min, lim.R_max);
  const double ddel   = clamp(u.ddelta, -lim.ddelta_max, lim.ddelta_max);
  n.delta = clamp(s.delta + ddel*dt, -lim.delta_max, lim.delta_max);

  dynamics::tire::VehicleGeom vg{vp.m, vp.L, vp.d, vp.JG};
  const auto fr = dynamics::tire::computeForcesBody(
      s.vx, s.vy, s.dpsi, n.delta, R_cmd, vg, tp, 9.81);

  printf("R_rear: %.2f N, out.Fx_sum: %.2f N, out.Fy_sum: %.2f N\n",
      R_cmd, fr.Fx_sum, fr.Fy_sum);

  double m_tot = vp.m + tp.m_unsprung_front + tp.m_unsprung_rear;

  const double ax   = fr.Fx_sum / m_tot + s.dpsi * s.vy;
  const double ay   = fr.Fy_sum / m_tot - s.dpsi * s.vx;
  const double rdot = fr.Mz     / vp.JG;

  n.vx   = std::max(0.0, s.vx + ax*dt);
  n.vy   =              s.vy + ay*dt;
  n.dpsi =              s.dpsi + rdot*dt;

  const double c = std::cos(n.psi), d = std::sin(n.psi);
  const double xdot = n.vx*c - n.vy*d;
  const double ydot = n.vx*d + n.vy*c;

  n.x   = s.x + xdot*dt;
  n.y   = s.y + ydot*dt;
  n.psi = wrapAngle(s.psi + n.dpsi*dt);
  n.s   = s.s + std::max(0.0, n.vx)*dt;
  return n;
}

} // namespace dynamics::vehicle
