#pragma once
#include <vector>
#include <functional>
#include <Eigen/Sparse>

struct MPCParams;
struct MPCRef;

namespace acc {

// ---------- Parameters ----------
struct Params {
    bool   enable{true};
    double tau{1.4};
    double dmin{5.0};
    double d_init{300.0}; 
};

// ---------- Lead profile ----------
struct PiecewiseLead {
    double v1{33.0}, v2{20.0}, v3{28.0};
    double t1{10.0}, t2{20.0}, t3{30.0};
};

// Module initialization
void reset_defaults();

// Configure lead
void set_piecewise_profile(const PiecewiseLead& pw);
void set_lead_speed_callback(const std::function<double(double)>& cb);
void set_lead_present(bool present);

// Query lead
bool lead_present();
double lead_speed(double t);

// ---------- GAP STATE MANAGEMENT ----------
// (Stored fully inside acc.cpp)

void reset_gap(const Params& p);  // resets global gap to p.d_init
double gap();                     // read current internal gap value
double update_gap(double v_lead, double v_ego, double dt); // updates & stores gap

// ---------- MPC preview ----------
void fill_preview(MPCRef& ref, double t0, double dt, int N);

// ---------- Constraint helper ----------
void append_headway_constraints(
    const Params& accp,
    int N, int id_d, int id_vx,
    std::vector<Eigen::Triplet<double>>& Aint,
    std::vector<double>& lin,
    std::vector<double>& uin,
    const std::function<bool(int)>& has_lead,
    int& row);

} // namespace acc
