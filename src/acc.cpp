#include "acc.hpp"
#include "mpc_ltv.hpp" 
#include <algorithm>
#include <limits>

namespace acc {

namespace {

// -------- global module state --------
PiecewiseLead                  g_pw{};
std::function<double(double)>  g_cb{};
bool                           g_present{true};

double                         g_gap{150.0};   

double pw_speed(const PiecewiseLead& pw, double t) {
    if (t < pw.t1) return pw.v1;
    if (t <= pw.t2){
        double u = (t - pw.t1) / (pw.t2 - pw.t1);
        return pw.v1 + (pw.v2 - pw.v1) * u;
    }
    if (t <= pw.t3){
        double u = (t - pw.t2) / (pw.t3 - pw.t2);
        return pw.v2 + (pw.v3 - pw.v2) * u;
    }
    return pw.v3;
}

} // anon

// ---------- param helpers ----------
Params from_mpc_params(const MPCParams&) {
    Params a;
    a.enable = true;   // or false, your default
    a.tau    = 1.4;
    a.dmin   = 5.0;
    a.d_init = 150.0;
    return a;
}

// ---------- module reset ----------
void reset_defaults(){
    g_pw = PiecewiseLead{};
    g_cb = [=](double t){ return pw_speed(g_pw, t); };
    g_present = true;
    g_gap = 150.0;
}

void set_piecewise_profile(const PiecewiseLead& pw){
    g_pw = pw;
    g_cb = [=](double t){ return pw_speed(g_pw, t); };
}

void set_lead_speed_callback(const std::function<double(double)>& cb){
    g_cb = cb; // may be empty
}

void set_lead_present(bool present){ g_present = present; }
bool lead_present(){ return g_present; }

double lead_speed(double t){
    return g_cb ? g_cb(t) : 0.0;
}

// ---------- GAP management ----------
void reset_gap(const Params& p){
    g_gap = p.d_init;
}

double gap(){
    return g_gap;
}

double update_gap(double v_lead, double v_ego, double dt){
    g_gap += dt * (v_lead - v_ego);
    if (g_gap < 0) g_gap = 0;
    return g_gap;
}

// ---------- Preview ----------
void fill_preview(MPCRef& ref, double t0, double dt, int N){
    ref.v_obj.resize(N);
    ref.has_obj.resize(N);
    for (int k = 0; k < N; ++k){
        const double tk = t0 + k * dt;
        ref.v_obj[k]   = lead_speed(tk);
        ref.has_obj[k] = static_cast<uint8_t>(g_present ? 1 : 0);
    }
}

// ---------- Constraint wrapper ----------
void append_headway_constraints(
    const Params& accp,
    int N, int id_d, int id_vx,
    std::vector<Eigen::Triplet<double>>&,
    std::vector<double>&,
    std::vector<double>&,
    const std::function<bool(int)>&,
    int&)
{
    // NOTE:
    // Your solveQP() must still place actual triplets using idx_x(k,...)
    // This helper remains optional.
    (void)accp;
    (void)N;
    (void)id_d;
    (void)id_vx;
}

} // namespace acc
