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

double slipAngleFront(double vx,double vy,double r,double delta,double Lf){
  const double vx_eff = (std::abs(vx)<0.1) ? (vx>=0.0?0.1:-0.1) : vx;
  const double beta_f = std::atan2(vy + Lf*r, vx_eff);
  return clampAlpha(beta_f - delta);
}
double slipAngleRear(double vx,double vy,double r,double Lr){
  const double vx_eff = (std::abs(vx)<0.1) ? (vx>=0.0?0.1:-0.1) : vx;
  const double beta_r = std::atan2(vy - Lr*r, vx_eff);
  return clampAlpha(beta_r);
}

double pacejkaFy(double B,double C,double D,double E,double a){
  const double x = B*a;
  return D * std::sin( C * std::atan( x - E*(x - std::atan(x)) ) );
}

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
ForceResult computeForcesBody(double vx, double vy, double r,
                              double delta, double R_rear,
                              const VehicleGeom& vg,
                              const TireParams& tp,
                              double g)
{
  ForceResult out;
  const double Lf = vg.L - vg.d, Lr = vg.d;

  double Fzf_ax, Fzr_ax;
  staticAxleLoads(vg.m, g, vg.L, vg.d, Fzf_ax, Fzr_ax);

  const double a_f = slipAngleFront(vx,vy,r,delta,Lf);
  const double a_r = slipAngleRear (vx,vy,r,       Lr);

  const double Df_w = tp.muf * (Fzf_ax * 0.5);  // per wheel
  const double Dr_w = tp.mur * (Fzr_ax * 0.5);

  double Fy_f_tire = -2.0 * pacejkaFy(tp.Bf,tp.Cf,Df_w,tp.Ef,a_f);
  double Fy_r_tire = -2.0 * pacejkaFy(tp.Br,tp.Cr,Dr_w,tp.Er,a_r);

  double Fx_f_tire = 0.0;       // RWD
  double Fx_r_tire = R_rear;

  clampEllipse(Fx_f_tire, Fy_f_tire, tp.muf, Fzf_ax);
  clampEllipse(Fx_r_tire, Fy_r_tire, tp.mur, Fzr_ax);

  const double c = std::cos(delta), s = std::sin(delta);
  out.Fx_f_body =  Fx_f_tire * c - Fy_f_tire * s;
  out.Fy_f_body =  Fx_f_tire * s + Fy_f_tire * c;

  out.Fx_r_body = Fx_r_tire;
  out.Fy_r_body = Fy_r_tire;

  out.Fx_sum = out.Fx_f_body + out.Fx_r_body;
  out.Fy_sum = out.Fy_f_body + out.Fy_r_body;
  out.Mz     =  Lf * out.Fy_f_body - Lr * out.Fy_r_body;
  return out;
}

} // namespace dynamics::tire
