#include "tire_model.hpp"
#include <cmath>
#include <iostream>

namespace dynamics::tire {

// ---- central params store ----
static TireParams g_tp{};
const TireParams& current() { return g_tp; }
void set(const TireParams& tp) { g_tp = tp; }

// ---- small internals ----
static inline double clampAlpha(double a, double amax=0.6){
  return std::clamp(a, -amax, amax);
}

double slipAngleFront(double vx,double vy,double dpsi,double delta,double Lf){
  const double vx_eff = (std::abs(vx)<0.1) ? (vx>=0.0?0.1:-0.1) : vx;
  const double beta_f = std::atan2(vy + Lf*dpsi, vx_eff);
  return clampAlpha(beta_f - delta);
}
double slipAngleRear(double vx,double vy,double dpsi,double Lr){
  const double vx_eff = (std::abs(vx)<0.1) ? (vx>=0.0?0.1:-0.1) : vx;
  const double beta_r = std::atan2(vy - Lr*dpsi, vx_eff);
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

// static inline void staticAxleLoads(double m,double g,double L,double d, double m_unsprung_f, double m_unsprung_r,
//                                    double& Fzf,double& Fzr){
//   Fzf = m*g*(L-d)/L + m_unsprung_f*g;  // front axle
//   Fzr = m*g*(d)/L + m_unsprung_r*g;    // rear axle
// }

// static inline void dynamicLoads(double m,double g,double L,double d, double Kzx, double m_unsprung_f, double m_unsprung_r,
//                                 double vx, double vy, double dpsi,
//                                    double& Fzf,double& Fzr){
//   Fzf = m*g*(L-d)/L + m_unsprung_f*g;  // front axle
//   Fzr = m*g*(d)/L + m_unsprung_r*g;    // rear axle
// }

// // ---- main API ----
// ForceResult computeForcesBody(double vx, double vy, double dpsi,
//                               double delta, double R_rear,
//                               const VehicleGeom& vg,
//                               const TireParams& tp,
//                               double g)
// {
//   ForceResult out;
//   const double Lf = vg.L - vg.d, Lr = vg.d;
//   const double Kzx = tp.Kzx;

//   double Fzf_ax, Fzr_ax;
//   staticAxleLoads(vg.m, g, vg.L, vg.d, tp.m_unsprung_front, tp.m_unsprung_rear, Fzf_ax, Fzr_ax);

//   const double a_f = slipAngleFront(vx,vy,dpsi,delta,Lf);
//   const double a_r = slipAngleRear (vx,vy,dpsi,      Lr);

//   const double Df_w = tp.muf * (Fzf_ax * 0.5);  // per wheel
//   const double Dr_w = tp.mur * (Fzr_ax * 0.5);

//   double Fy_f_tire = -2.0 * pacejkaFy(tp.Bf,tp.Cf,Df_w,tp.Ef,a_f);
//   double Fy_r_tire = -2.0 * pacejkaFy(tp.Br,tp.Cr,Dr_w,tp.Er,a_r);

//   double Fx_f_tire = 0.0;       // RWD
//   double Fx_r_tire = R_rear;

//   clampEllipse(Fx_f_tire, Fy_f_tire, tp.muf, Fzf_ax);
//   clampEllipse(Fx_r_tire, Fy_r_tire, tp.mur, Fzr_ax);

//   const double c = std::cos(delta), s = std::sin(delta);
//   out.Fx_f_body =  Fx_f_tire * c - Fy_f_tire * s;
//   out.Fy_f_body =  Fx_f_tire * s + Fy_f_tire * c;

//   out.Fx_r_body = Fx_r_tire;
//   out.Fy_r_body = Fy_r_tire;

//   out.Fx_sum = out.Fx_f_body + out.Fx_r_body;
//   out.Fy_sum = out.Fy_f_body + out.Fy_r_body;
//   out.Mz     =  Lf * out.Fy_f_body - Lr * out.Fy_r_body;
//   return out;
// }


static inline void staticAxleLoads(double m_s, double g,
                                   double L, double d,
                                   double m_unsprung_f, double m_unsprung_r,
                                   double& Fzf0, double& Fzr0)
{
    const double Fz_front_sprung = m_s * g * (L - d) / L;
    const double Fz_rear_sprung  = m_s * g * d / L;
    Fzf0 = Fz_front_sprung + m_unsprung_f * g;
    Fzr0 = Fz_rear_sprung  + m_unsprung_r * g;
}

static inline void dynamicLoads(double Fzf0, double Fzr0,
                                double Kzx, double ax_body,
                                double& Fzf, double& Fzr)
{
    const double dF = Kzx * ax_body;  // load shift front<->rear
    Fzf = Fzf0 - dF;
    Fzr = Fzr0 + dF;
}

ForceResult computeForcesBody(double vx, double vy, double dpsi,
                              double delta, double R_rear,
                              const VehicleGeom& vg,
                              const TireParams& tp,
                              double g)
{
    ForceResult out;
    const double Lf = vg.L - vg.d;
    const double Lr = vg.d;

    // total mass (sprung + unsprung)
    const double m_tot = vg.m + tp.m_unsprung_front + tp.m_unsprung_rear;

    // --- 1) static axle loads ---
    double Fzf0, Fzr0;
    staticAxleLoads(vg.m, g, vg.L, vg.d,
                    tp.m_unsprung_front, tp.m_unsprung_rear,
                    Fzf0, Fzr0);

    // --- 2) approximate body-longitudinal acceleration from command ---
    // RWD: front Fx = 0, rear Fx ≈ R_rear (before saturation)
    const double Fx_cmd_body = R_rear;
    const double ax_body = Fx_cmd_body / m_tot;   // ≈ (Udot - V*omega_z)

    // --- 3) dynamic loads using this approximate ax_body ---
    double Fzf_dyn, Fzr_dyn;
    dynamicLoads(Fzf0, Fzr0, tp.Kzx, ax_body, Fzf_dyn, Fzr_dyn);
    if (Fzf_dyn < 0.0 || Fzr_dyn < 0.0) {
        std::cout << "Warning: Negative dynamic axle load computed! "
                  << "Fzf_dyn: " << Fzf_dyn << ", Fzr_dyn: " << Fzr_dyn << "\n";
        std::cout << "  (Fzf0: " << Fzf0 << ", Fzr0: " << Fzr0
                  << ", ax_body: " << ax_body << ", Kzx: " << tp.Kzx << ")\n";
    }

    // optional safety: avoid negative normal loads
    Fzf_dyn = std::max(0.0, Fzf_dyn);
    Fzr_dyn = std::max(0.0, Fzr_dyn);

    // --- 4) single-pass tire forces with dynamic loads ---
    const double a_f = slipAngleFront(vx,vy,dpsi,delta,Lf);
    const double a_r = slipAngleRear (vx,vy,dpsi,      Lr);

    const double Df_w = tp.muf * (Fzf_dyn * 0.5);    // per wheel
    const double Dr_w = tp.mur * (Fzr_dyn * 0.5);

    double Fy_f_tire = -2.0 * pacejkaFy(tp.Bf,tp.Cf,Df_w,tp.Ef,a_f);
    double Fy_r_tire = -2.0 * pacejkaFy(tp.Br,tp.Cr,Dr_w,tp.Er,a_r);

    double Fx_f_tire = 0.0;      // RWD
    double Fx_r_tire = R_rear;   // commanded traction/brake

    // combined-slip clamp
    clampEllipse(Fx_f_tire, Fy_f_tire, tp.muf, Fzf_dyn);
    clampEllipse(Fx_r_tire, Fy_r_tire, tp.mur, Fzr_dyn);

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
