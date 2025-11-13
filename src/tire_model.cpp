#include "tire_model.hpp"
#include <cmath>

namespace dynamics::tire {

// ---- central params store ----
static TireParams g_tp{};
const TireParams& current() { return g_tp; }
void set(const TireParams& tp) { g_tp = tp; }

// ---- small internals ----
static inline double clampAlpha(double a, double amax=0.6){
  return std::clamp(a, -amax, amax);
}

// double slipAngleFront(double vx,double vy,double r,double delta,double Lf){
//   const double vx_eff = (std::abs(vx)<0.1) ? (vx>=0.0?0.1:-0.1) : vx;
//   const double beta_f = std::atan2(vy + Lf*r, vx_eff);
//   return clampAlpha(beta_f - delta);
// }
// double slipAngleRear(double vx,double vy,double r,double Lr){
//   const double vx_eff = (std::abs(vx)<0.1) ? (vx>=0.0?0.1:-0.1) : vx;
//   const double beta_r = std::atan2(vy - Lr*r, vx_eff);
//   return clampAlpha(beta_r);
// }

// double pacejkaFy(double B,double C,double D,double E,double a){
//   const double x = B*a;
//   return D * std::sin( C * std::atan( x - E*(x - std::atan(x)) ) );
// }

static inline void clampEllipse(double& Fx, double& Fy, double mu, double Fz){
  const double Fmax = std::max(1e-6, mu*Fz);
  const double n = std::hypot(Fx, Fy);
  if (n > Fmax){ const double s = Fmax/n; Fx*=s; Fy*=s; }
}

static inline void staticAxleLoads(double m,double g,double L,double d,
                                   double& Fzf,double& Fzr){
  Fzf = m*g*(L-d)/L;  // front axle
  Fzr = m*g*(d)/L;    // rear axle
}

// ---- main API ----
ForceResult computeForcesBody(double v,
                              double delta, double R_rear, double ddelta,
                              const VehicleGeom& vg,
                              double g)
{
  ForceResult out;
  const double Lf = vg.L - vg.d, Lr = vg.d;

  const double c  = std::cos(delta);
  const double s2 = 1.0 / (c * c);      // sec^2(delta)
  const double t  = std::tan(delta);
  const double denom  = (vg.m + vg.m0 * t * t);
  const double common = (R_rear * t) + (vg.m * v * ddelta * s2); // Fwind = 0

  // From the figure (Eq. 7), with Fwind = 0
  const double Ffl = (vg.m / vg.L) * t * (1.0 - vg.d / vg.L) * v * v
                    - ((vg.m0 - vg.m * vg.d / vg.L) / denom) * common;

  const double Frl = (1.0 / c) * ( (vg.m / (vg.L * vg.L)) * vg.d * v * v * t
                                  + (vg.m0 / denom) * common );

  out.Fy_f_body = Ffl;
  out.Fy_r_body = Frl;

  return out;
}

} // namespace dynamics::tire
