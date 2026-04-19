/// ------------------------------- /// 
/// Tire Dynamics. (tire_model.cpp)  ///
/// ------------------------------- ///

#include "tire_model.hpp"
#include <cmath>
#include <iostream>

namespace dynamics::tire {

// ---- central params store ----
// Global (process-wide) tire parameter instance used as default.
static TireParams g_tp{};

// Return current global tire parameters
const TireParams& current() { return g_tp; }

// Set global tire parameters (e.g., from configuration)
void set(const TireParams& tp) { g_tp = tp; }

// ---- small internals ----

// Clamp slip angle magnitude to avoid ridiculous angles in the MF
static inline double clampAlpha(double a, double amax=0.6){
  return std::clamp(a, -amax, amax);
}

// Front axle slip angle:
//   alpha_f = beta_f - delta
// where beta_f is front-axle sideslip, computed from (vx, vy, yaw rate)
double slipAngleFront(double vx,double vy,double dpsi,double delta,double Lf){
  // Avoid division by very small longitudinal speeds
  const double vx_eff = (std::abs(vx)<0.1) ? (vx>=0.0?0.1:-0.1) : vx;
  // front-axle sideslip angle
  const double beta_f = std::atan2(vy + Lf*dpsi, vx_eff);
  return clampAlpha(beta_f - delta);
}

// Rear axle slip angle (no steering at rear):
//   alpha_r = beta_r
double slipAngleRear(double vx,double vy,double dpsi,double Lr){
  // Avoid division by very small longitudinal speeds
  const double vx_eff = (std::abs(vx)<0.1) ? (vx>=0.0?0.1:-0.1) : vx;
  const double beta_r = std::atan2(vy - Lr*dpsi, vx_eff);
  return clampAlpha(beta_r);
}

// Lateral force from Pacejka "Magic Formula":
//   Fy = D * sin( C * atan( B*alpha - E( B*alpha - atan(B*alpha) ) ) )
double pacejkaFy(double B,double C,double D,double E,double a){
  const double x = B*a;
  return D * std::sin( C * std::atan( x - E*(x - std::atan(x)) ) );
}

// Clamp combined longitudinal/lateral forces to friction ellipse:
//   sqrt(Fx^2 + Fy^2) <= mu * Fz
static inline void clampEllipse(double& Fx, double& Fy, double mu, double Fz){
  const double Fmax = std::max(1e-6, mu*Fz);
  const double n = std::hypot(Fx, Fy);
  if (n > Fmax){ const double s = Fmax/n; Fx*=s; Fy*=s; }
}

// Compute static normal loads on each axle (front/rear) from CG position and
// unsprung masses.
static inline void staticAxleLoads(double m_s, double g,
                                   double L, double d,
                                   double m_unsprung_f, double m_unsprung_r,
                                   double& Fzf0, double& Fzr0)
{
    // Sprung mass distribution based on CG location
    const double Fz_front_sprung = m_s * g * (L - d) / L;
    const double Fz_rear_sprung  = m_s * g * d / L;
    // Add unsprung mass contribution at each axle
    Fzf0 = Fz_front_sprung + m_unsprung_f * g;
    Fzr0 = Fz_rear_sprung  + m_unsprung_r * g;
}

// Approximate longitudinal load transfer using:
//   Fzf = Fzf0 - Kzx * ax_body
//   Fzr = Fzr0 + Kzx * ax_body
// where Kzx is a tuning coefficient.
static inline void dynamicLoads(double Fzf0, double Fzr0,
                                double Kzx, double ax_body,
                                double& Fzf, double& Fzr)
{
    const double dF = Kzx * ax_body;  // load shift front<->rear
    Fzf = Fzf0 - dF;
    Fzr = Fzr0 + dF;
}

// -----------------------------------------------------------------------------
// computeForcesBody
//
// Computes tire forces at front and rear axles in the **body frame** given:
//
//   - longitudinal & lateral velocities (vx, vy)
//   - yaw rate (dpsi)
//   - steering angle (delta)
//   - rear longitudinal command (R_rear)
//   - vehicle geometry / mass (vg)
//   - tire parameters (tp)
//   - gravity (g)
//
// Model summary:
//   1. Compute static axle loads from CG and unsprung masses.
//   2. Approximate body longitudinal acceleration from R_rear to model
//      longitudinal load transfer.
//   3. Compute dynamic axle loads using Kzx and ax_body.
//   4. Compute front/rear slip angles and lateral forces via Pacejka.
//   5. Combine longitudinal & lateral forces and clamp to friction ellipse.
//   6. Rotate front forces to body frame, sum forces, and compute yaw moment.
// -----------------------------------------------------------------------------
ForceResult computeForcesBody(double vx, double vy, double dpsi,
                              double delta, double R_rear,
                              const VehicleGeom& vg,
                              const TireParams& tp,
                              double g)
{
    ForceResult out;
    const double Lf = vg.L - vg.d;  // distance CG->front axle
    const double Lr = vg.d;        // distance CG->rear axle

    // total mass (sprung + unsprung)
    const double m_tot = vg.m + tp.m_unsprung_front + tp.m_unsprung_rear;

    // --- 1) static axle loads ---
    double Fzf0, Fzr0;
    staticAxleLoads(vg.m, g, vg.L, vg.d,
                    tp.m_unsprung_front, tp.m_unsprung_rear,
                    Fzf0, Fzr0);

    // --- 2) approximate body-longitudinal acceleration from command ---
    // RWD assumption: front Fx = 0, rear Fx ≈ R_rear (before saturation)
    const double Fx_cmd_body = R_rear;
    const double ax_body = Fx_cmd_body / m_tot;   // ≈ longitudinal accel

    // --- 3) dynamic loads using this approximate ax_body ---
    double Fzf_dyn, Fzr_dyn;
    dynamicLoads(Fzf0, Fzr0, tp.Kzx, ax_body, Fzf_dyn, Fzr_dyn);
    if (Fzf_dyn < 0.0 || Fzr_dyn < 0.0) {
        std::cout << "Warning: Negative dynamic axle load computed! "
                  << "Fzf_dyn: " << Fzf_dyn << ", Fzr_dyn: " << Fzr_dyn << "\n";
        std::cout << "  (Fzf0: " << Fzf0 << ", Fzr0: " << Fzr0
                  << ", ax_body: " << ax_body << ", Kzx: " << tp.Kzx << ")\n";
    }

    // Optional safety: clamp floor of normal loads to zero
    Fzf_dyn = std::max(0.0, Fzf_dyn);
    Fzr_dyn = std::max(0.0, Fzr_dyn);

    // --- 4) single-pass tire forces with dynamic loads ---
    // Slip angles (front/rear)
    const double a_f = slipAngleFront(vx,vy,dpsi,delta,Lf);
    const double a_r = slipAngleRear (vx,vy,dpsi,      Lr);

    // Per-wheel force capacity scaling: assume two wheels per axle
    const double Df_w = tp.muf * (Fzf_dyn * 0.5);    // front
    const double Dr_w = tp.mur * (Fzr_dyn * 0.5);    // rear

    // Lateral forces from Pacejka for each axle (multiply by 2 wheels)
    double Fy_f_tire = -2.0 * pacejkaFy(tp.Bf,tp.Cf,Df_w,tp.Ef,a_f);
    double Fy_r_tire = -2.0 * pacejkaFy(tp.Br,tp.Cr,Dr_w,tp.Er,a_r);

    // Longitudinal forces: RWD (front Fx=0, rear Fx=R_rear)
    double Fx_f_tire;      // front: no drive/brake
    double Fx_r_tire;   // rear: commanded traction/brake

    if (R_rear >= 0.0) {
        // --- GAS (RWD) ---
        // 100% torque to rear axle
        Fx_f_tire = 0.0;
        Fx_r_tire = R_rear;
    } 
    else {
        // --- BRAKE (All-Wheel Braking) ---
        // Standard Bias: 60% Front, 40% Rear
        // Note: R_rear is negative here, so forces will be negative
        const double brake_bias = 0.60; 
        
        Fx_f_tire = R_rear * brake_bias;       // e.g. -1000 * 0.6 = -600 N
        Fx_r_tire = R_rear * (1.0 - brake_bias); // e.g. -1000 * 0.4 = -400 N
    }

    // combined-slip clamp at each axle using friction ellipse
    clampEllipse(Fx_f_tire, Fy_f_tire, tp.muf, Fzf_dyn);
    clampEllipse(Fx_r_tire, Fy_r_tire, tp.mur, Fzr_dyn);

    // Rotate front axle forces from tire frame to body frame via steering delta
    const double c = std::cos(delta), s = std::sin(delta);

    out.Fx_f_body =  Fx_f_tire * c - Fy_f_tire * s;
    out.Fy_f_body =  Fx_f_tire * s + Fy_f_tire * c;

    out.Fx_r_body = Fx_r_tire;
    out.Fy_r_body = Fy_r_tire;

    // Sum forces and compute yaw moment about CG
    out.Fx_sum = out.Fx_f_body + out.Fx_r_body;
    out.Fy_sum = out.Fy_f_body + out.Fy_r_body;
    out.Mz     =  Lf * out.Fy_f_body - Lr * out.Fy_r_body;

    // Store axle normal loads and slip angles for logging/debug
    out.Fz_f_body = Fzf_dyn;
    out.Fz_r_body = Fzr_dyn;
    
    out.alpha_f   = a_f;
    out.alpha_r   = a_r;

    return out;
}

} // namespace dynamics::tire
