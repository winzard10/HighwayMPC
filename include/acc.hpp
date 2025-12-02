/// ------------------------ /// 
/// ACC Parameters (acc.hpp) ///
/// ------------------------ ///

#pragma once
#include <vector>
#include <functional>
#include <Eigen/Sparse>

struct MPCParams;
struct MPCRef;

namespace acc {

// ---------- Parameters ----------
// ACC configuration parameters
struct Params {
    bool   enable{true};    // ACC on/off flag (used by MPC)
    double tau{1.4};        // desired time headway [s]
    double dmin{5.0};       // minimum standstill distance [m]
    double d_init{150.0};   // initial gap to lead vehicle [m]
};

// ---------- Lead profile ----------
// Simple piecewise lead-vehicle speed profile used for testing ACC.
struct PiecewiseLead {
    double v1{33.0}, v2{20.0}, v3{28.0};  // segment speeds [m/s]
    double t1{10.0}, t2{20.0}, t3{40.0};  // break times [s]
};

// Module initialization: reset internal globals (lead profile, gap, etc.)
void reset_defaults();

// Configure lead profile / callback
void set_piecewise_profile(const PiecewiseLead& pw);
void set_lead_speed_callback(const std::function<double(double)>& cb);
void set_lead_present(bool present);

// Query lead presence / speed
bool lead_present();
double lead_speed(double t);

// ---------- GAP STATE MANAGEMENT ----------
// (Stored fully inside acc.cpp)
//
// Global gap dynamics are kept here to be shared between the simulator
// and any ACC-related logic without duplicating state elsewhere.

void reset_gap(const Params& p);  // resets global gap to p.d_init
double gap();                     // read current internal gap value
double update_gap(double v_lead, double v_ego, double dt); // updates & stores gap

// ---------- MPC preview ----------
// Fill MPCRef with lead-speed and presence flags over horizon
void fill_preview(MPCRef& ref, double t0, double dt, int N);

} // namespace acc
