#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <cstddef>
#include <optional>

#include "obstacles.hpp"      // MPCObsSet + compute_lateral_bounds
#include "vehicle_model.hpp"  // dynamics::vehicle::{Params,Limits}
#include "tire_model.hpp"     // dynamics::tire::{TireParams,current}
#include "acc.hpp"            // acc::Params

namespace acc { struct Params; }

// ---------------------------------
// MPC configuration
// ---------------------------------
struct MPCParams {
    int    N   = 200;
    double dt;

    // weights
    double wy    = 2.5;
    double wpsi  = 0.10;
    double wv    = 0.25;
    double wR    = 1e-4;
    double wdR   = 1e-5;
    double wddR  = 2e-5;
    double wdd   = 8.0;
    double wddd  = 10.0;
    double wyf   = 3.0;
    double wpsif = 8.0;

    // bounds
    double ddelta_max = 0.25;
    double delta_max  = 0.40;
    double v_min = 0.0,  v_max = 40.0;

    double ey_up_max = 3.0;
    double ey_lo_max = -3.0;
    double ey_max    = 1.8;   // fallback if no corridor provided
};

// Preview point used by sim
struct PreviewPoint {
    double kappa = 0.0;  // road curvature
    double v_ref = 0.0;  // target speed
};

// Horizon preview (sim fills hp[i].{kappa,v_ref})
struct MPCRef {
    std::vector<PreviewPoint> hp;     // length >= N
    std::vector<double> ey_ref;       // size N+1
    double ey_ref_N{0.0};             // terminal

    // Lead object profile
    std::vector<double>  v_obj;       // size N
    std::vector<uint8_t> has_obj;     // size N

    std::vector<double> epsi_nom;
};

// State / Input / Output
struct MPCState {
    double ey    = 0.0;
    double epsi  = 0.0;
    double vx    = 0.0;
    double vy    = 0.0;
    double dpsi  = 0.0;
    double delta = 0.0;
    double d     = 1e6;   // gap (used iff acc_enable)
};

struct MPCControl {
    double R      = 0.0;  // rear-axle longitudinal force [N]
    double ddelta = 0.0;  // steering rate [rad/s]
    bool   ok     = false;
};

// ---------------------------------
// LTV MPC Controller
// ---------------------------------
class LTV_MPC {
public:
    explicit LTV_MPC(const MPCParams& p) : P(p) {
        nom_.x.resize(P.N + 1);
        nom_.u.resize(P.N);
        tp_ = dynamics::tire::current();   // default from tire_model
    }

    // Physics / limits
    void setVehicleParams(const dynamics::vehicle::Params& vp);  // defined in .cpp
    void setLimits(const dynamics::vehicle::Limits& L);          // defined in .cpp
    void setTireParams(const dynamics::tire::TireParams& tp) { tp_ = tp; }

    // Obstacles / corridor
    void setObstacleConstraints(std::optional<MPCObsSet> s) { obs_ = std::move(s); }
    void setCorridorBounds(const std::vector<double>& lo,
                           const std::vector<double>& up);

    // ACC 
    void setACCParams(const acc::Params& p) { accp_ = p; }
    const acc::Params& accParams() const { return accp_; }

    // Nominal warm-start
    void setNominal(const std::vector<MPCState>& x_nom,
                    const std::vector<Eigen::Vector2d>& u_nom) {
        nom_.x = x_nom; nom_.u = u_nom;
    }

    // Model + QP
    void        buildLinearization(const MPCRef& ref);
    MPCControl  solveQP(const MPCState& x0, const MPCRef& ref);

    // Convenience wrapper
    MPCControl solve(const MPCState& x0, const MPCRef& ref) {
        buildLinearization(ref);
        return solveQP(x0, ref);
    }

    // helpers
    static void angleWrap(double& a);

private:
    MPCParams P;

    // Shared vehicle + tire model parameters
    dynamics::vehicle::Params  vp_{};   // m, L, d, JG, dt
    dynamics::vehicle::Limits  lim_{};  // steering & R bounds
    dynamics::tire::TireParams tp_{};   // B,C,E,mu (front/rear)
    // acc::Params accp_{ /*enable=*/false, /*tau=*/1.4, /*dmin=*/5.0, /*d_init=*/150.0 };
    acc::Params accp_{};

    struct LinModel {
        std::vector<Eigen::MatrixXd> A;
        std::vector<Eigen::MatrixXd> B;
        std::vector<Eigen::VectorXd> c;
    } lm_;

    struct Nominal {
        std::vector<MPCState>        x;  // N+1
        std::vector<Eigen::Vector2d> u;  // N
    } nom_;

    std::vector<double> ey_lo_provided_;
    std::vector<double> ey_up_provided_;
    bool have_corridor_bounds_{false};

    std::optional<MPCObsSet> obs_;
};
