/// -------------------- /// 
/// ACC module (acc.cpp) ///
/// -------------------- ///

#include "acc.hpp"
#include "mpc_ltv.hpp" 
#include <algorithm>
#include <limits>

namespace acc {

namespace {

// -------- global module state --------
// These globals represent the current lead-vehicle model and gap state,
// used by ACC preview and gap integration.

PiecewiseLead                  g_pw{};     // piecewise-constant/linear lead profile
std::function<double(double)>  g_cb{};     // optional callback for lead speed v_lead(t)
bool                           g_present{true}; // whether lead vehicle is present

double                         g_gap{150.0};   // internal gap state [m]

// Simple helper: evaluate piecewise lead speed profile at time t
double pw_speed(const PiecewiseLead& pw, double t) {
    if (t < pw.t1) return pw.v1;
    if (t <= pw.t2){
        double u = (t - pw.t1) / (pw.t2 - pw.t1);      // blend 1 -> 2
        return pw.v1 + (pw.v2 - pw.v1) * u;
    }
    if (t <= pw.t3){
        double u = (t - pw.t2) / (pw.t3 - pw.t2);      // blend 2 -> 3
        return pw.v2 + (pw.v3 - pw.v2) * u;
    }
    return pw.v3;
}

} // anon

// ---------- param helpers ----------
// Build ACC parameters from generic MPC parameters (if needed).
// Currently just returns a default ACC configuration.
Params from_mpc_params(const MPCParams&) {
    Params a;
    a.enable = true;   // or false, your default
    a.tau    = 1.4;
    a.dmin   = 5.0;
    a.d_init = 150.0;
    return a;
}

// ---------- module reset ----------
// Reset internal ACC module state to defaults:
//   - zeroed piecewise profile
//   - default callback using pw_speed(g_pw, t)
//   - lead present
//   - gap reset to 150 m
void reset_defaults(){
    g_pw = PiecewiseLead{};
    g_cb = [=](double t){ return pw_speed(g_pw, t); };
    g_present = true;
    g_gap = 150.0;
}

// Set piecewise lead profile and hook its speed function as callback
void set_piecewise_profile(const PiecewiseLead& pw){
    g_pw = pw;
    g_cb = [=](double t){ return pw_speed(g_pw, t); };
}

// Set arbitrary lead-speed callback v_lead(t). If empty, lead_speed() returns 0.
void set_lead_speed_callback(const std::function<double(double)>& cb){
    g_cb = cb; // may be empty
}

// Mark whether a lead vehicle is present
void set_lead_present(bool present){ g_present = present; }

// Query presence of lead vehicle
bool lead_present(){ return g_present; }

// Query lead speed at time t (either from callback or default 0)
double lead_speed(double t){
    return g_cb ? g_cb(t) : 0.0;
}

// ---------- GAP management ----------
// Reset internal gap state to initial distance from Params
void reset_gap(const Params& p){
    g_gap = p.d_init;
}

// Read current internal gap state
double gap(){
    return g_gap;
}

// Integrate gap state:
//   d_{k+1} = d_k + dt * (v_lead - v_ego)
// and clamp to d >= 0
double update_gap(double v_lead, double v_ego, double dt){
    g_gap += dt * (v_lead - v_ego);
    if (g_gap < 0) g_gap = 0;
    return g_gap;
}

// ---------- Preview ----------
// Fill MPCRef with lead-vehicle preview:
//   ref.v_obj[k]   = v_lead(t0 + k*dt)
//   ref.has_obj[k] = 1 if a lead is present (global flag), else 0
void fill_preview(MPCRef& ref, double t0, double dt, int N){
    ref.v_obj.resize(N);
    ref.has_obj.resize(N);
    for (int k = 0; k < N; ++k){
        const double tk = t0 + k * dt;
        ref.v_obj[k]   = lead_speed(tk);
        ref.has_obj[k] = static_cast<uint8_t>(g_present ? 1 : 0);
    }
}

} // namespace acc
