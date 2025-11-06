// vehicle_model.hpp
#pragma once
#include "tire_model.hpp"

namespace dynamics::vehicle {

struct Params {
  double m{660.0};
  double L{3.4};
  double d{1.6};
  double JG{2500.0};
  double m0{185.09};

  double dt{0.05};
  double track_w{1.7};
};

struct Limits {
  double delta_max{0.5};
  double ddelta_max{0.7};
  double R_min{-5000.0};
  double R_max{ 5000.0};
  double Ffl_max{5000.0};
  double Frl_max{5500.0};
};

struct State {
  double s{0.0};
  double x{0.0}, y{0.0}, psi{0.0};
  double vx{0.0}, vy{0.0}, dpsi{0.0};
  double delta{0.0};
};

struct Control {
  double R{0.0};
  double ddelta{0.0};
};

// advance one step (uses tire model)
State step(const State& s, const Control& u,
           const Params& vp, const Limits& lim,
           const dynamics::tire::TireParams& tp);

} // namespace dynamics::vehicle
