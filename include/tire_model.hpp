/// --------------------------------- /// 
/// Tire Parameters (tire_model.hpp)  ///
/// --------------------------------- ///

#pragma once
#include <algorithm>
#include <cmath>

namespace dynamics::tire {

// -------- Parameters --------
// Tire + load-transfer parameters used by the dynamic bicycle model.
struct TireParams {
  double muf{0.9};  // effective friction coefficient at front axle
  double mur{0.9};  // effective friction coefficient at rear axle

  // Magic Formula (Pacejka) lateral parameters for front axle
  double Bf{13.0}, Cf{1.285}, Ef{-0.9};
  // Magic Formula (Pacejka) lateral parameters for rear axle
  double Br{13.0}, Cr{1.285}, Er{-0.9};

  // Unsprung masses (lumped per axle)
  double m_unsprung_front{120.0}; // per front axle [kg]
  double m_unsprung_rear{80.0};   // per rear axle [kg]

  // Longitudinal load transfer coefficient (front <-> rear)
  // Fzf = Fzf0 - Kzx*ax_body, Fzr = Fzr0 + Kzx*ax_body
  double Kzx{130.0}; // Longitudinal load transfer coefficient
};

// Active / global tire set
// These provide a simple global storage for default tire parameters.
const TireParams& current();
void set(const TireParams&);

// -------- Vehicle geometry used by tire model --------
struct VehicleGeom {
  double m;   // sprung mass [kg]
  double L;   // wheelbase [m]
  double d;   // CG distance from rear axle [m]
  double JG;  // yaw moment of inertia [kg m^2]
};

// -------- Output forces --------
struct ForceResult {
  // Total forces / moment in body frame
  double Fx_sum{0.0};
  double Fy_sum{0.0};
  double Mz{0.0};

  // Per-axle body-frame forces
  double Fx_f_body{0.0};
  double Fy_f_body{0.0};
  double Fx_r_body{0.0};
  double Fy_r_body{0.0};

  // Axle normal loads
  double Fz_f_body{0.0};
  double Fz_r_body{0.0};

  // Slip angles (front/rear)
  double alpha_f{0.0};
  double alpha_r{0.0};
};

// API: compute body-frame forces from tire model + load transfer
ForceResult computeForcesBody(double vx, double vy, double dpsi,
                              double delta, double R_rear,
                              const VehicleGeom& vg,
                              const TireParams& tp,
                              double g = 9.81);

// Helpers
double pacejkaFy(double B,double C,double D,double E,double alpha);
double slipAngleFront(double vx,double vy,double dpsi,double delta,double Lf);
double slipAngleRear (double vx,double vy,double dpsi,double Lr);

} // namespace dynamics::tire
