// tire_model.hpp
#pragma once
#include <algorithm>
#include <cmath>

namespace dynamics::tire {

// -------- Parameters --------
struct TireParams {
  double muf{1.00};  // front axle friction
  double mur{1.00};  // rear axle friction
  double Bf{7.8727}, Cf{2.5296}, Ef{1.3059};  // front MF
  double Br{7.8727}, Cr{2.5296}, Er{1.3059};  // rear  MF
};

// Active / global tire set
const TireParams& current();
void set(const TireParams&);

// -------- Vehicle geometry used by tire model --------
struct VehicleGeom {
  double m{660.0};
  double L{3.4};
  double d{1.6};
  double JG{2500.0};
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
ForceResult computeForcesBody(double vx, double vy, double r,
                              double delta, double R_rear,
                              const VehicleGeom& vg,
                              const TireParams& tp,
                              double g = 9.81);

// (exposed helpers if you need them elsewhere)
double pacejkaFy(double B,double C,double D,double E,double alpha);
double slipAngleFront(double vx,double vy,double r,double delta,double Lf);
double slipAngleRear (double vx,double vy,double r,double Lr);

} // namespace dynamics::tire
