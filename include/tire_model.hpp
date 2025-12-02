// tire_model.hpp
#pragma once
#include <algorithm>
#include <cmath>

namespace dynamics::tire {

// -------- Parameters --------
struct TireParams {
  double muf{0.9};  // front axle friction
  double mur{0.9};  // rear axle friction
  // double Bf{13.0}, Cf{1.3}, Ef{-0.9};  // front MF
  // double Br{13.0}, Cr{1.3}, Er{-0.9};  // rear  MF
  double Bf{13.0}, Cf{1.285}, Ef{-0.9};  // front MF
  double Br{13.0}, Cr{1.285}, Er{-0.9};  // rear  MF


  double m_unsprung_front{120.0}; // per axle
  double m_unsprung_rear{80.0};  // per axle

  double Kzx{130.0}; // Longitudinal load transfer coefficient
};

// Active / global tire set
const TireParams& current();
void set(const TireParams&);

// -------- Vehicle geometry used by tire model --------
struct VehicleGeom {
  double m;
  double L;
  double d;
  double JG;
};

// -------- Output forces --------
struct ForceResult {
  // totals (body frame)
  double Fx_sum{0.0};
  double Fy_sum{0.0};
  double Mz{0.0};
  // per-axle body-frame forces
  double Fx_f_body{0.0};
  double Fy_f_body{0.0};
  double Fx_r_body{0.0};
  double Fy_r_body{0.0};
};

// API
ForceResult computeForcesBody(double vx, double vy, double dpsi,
                              double delta, double R_rear,
                              const VehicleGeom& vg,
                              const TireParams& tp,
                              double g = 9.81);

// (exposed helpers if you need them elsewhere)
double pacejkaFy(double B,double C,double D,double E,double alpha);
double slipAngleFront(double vx,double vy,double dpsi,double delta,double Lf);
double slipAngleRear (double vx,double vy,double dpsi,double Lr);

} // namespace dynamics::tire
