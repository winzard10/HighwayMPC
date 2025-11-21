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
  const double Frl = (vg.m / vg.L) * t * (1.0 - vg.d / vg.L) * v * v
                    - ((vg.m0 - vg.m * vg.d / vg.L) / denom) * common;

  const double Ffl = (1.0 / c) * ( (vg.m / (vg.L * vg.L)) * vg.d * v * v * t
                                  + (vg.m0 / denom) * common );

  out.Fy_f_body = Ffl;
  out.Fy_r_body = Frl;

  return out;
}

} // namespace dynamics::tire
