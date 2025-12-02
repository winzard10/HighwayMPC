/// ------------------------------------- /// 
/// Highway LTV-MPC header (mpc_ltv.hpp)  ///
/// ------------------------------------- ///

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

using VControl = dynamics::vehicle::Control;

namespace acc { struct Params; }

// ---------------------------------
// MPC configuration (weights, horizon, bounds)
// ---------------------------------
struct MPCParams {
    int    N   = 100;   // prediction horizon length
    double dt;          // sampling time [s]

    // stage + terminal weights
    double wy    = 2.0;   // lateral error ey tracking weight
    double wpsi  = 0.1;   // heading error epsi weight
    double wv    = 2.5;   // longitudinal speed tracking weight
    double wR    = 1e-4;  // input R (longitudinal) magnitude weight
    double wdR   = 1e-5;  // input R rate (R_k - R_{k-1}) weight
    double wddR  = 2e-5;  // input R jerk (R_k - 2R_{k-1} + R_{k-2}) weight
    double wdd   = 12.0;  // steering rate |ddelta| weight
    double wddd  = 5.0;   // steering rate change weight
    double wyf   = 3.0;   // terminal ey_N weight
    double wpsif = 8.0;   // terminal epsi_N weight

    // ACC slack penalty – allows small violation of headway
    double w_acc_slack = 10.0; // slack variable penalty for ACC -> allowing to violate headway approx 14 cm.

    // hard bounds
    double ddelta_max = 0.25;   // max steering rate [rad/s]
    double delta_max  = 0.40;   // steering angle limit [rad]
    double v_min = 0.0,  v_max = 40.0;  // speed limits [m/s]

    // corridor bounds (used when obstacle constraints provide lateral limits)
    double ey_max    = 1.8;   // fallback lateral bound if no corridor provided
};

// Preview point along reference centerline
// (used for each stage in the horizon)
struct PreviewPoint {
    double kappa = 0.0;  // road curvature at preview point
    double v_ref = 0.0;  // target speed at preview point
};

// Polynomial coefficients for speed-dependent accel limits:
//   a_max(vx) = sum c_i * vx^i
//   a_min(vx) = sum c_i * vx^i
struct ACCBoundCoeffs {
    std::vector<double> a_max_coeffs{
        2.0,          // c0
        -0.0425,      // c1
        0.000125,     // c2
        0.0           // c3
        };  // size N
    std::vector<double> a_min_coeffs{
        -2.5,         // c0
        0.0,          // c1
        0.0,          // c2
        0.0           // c3
        };  // size N
};

// Horizon reference data populated by the simulator
// and path planner.
struct MPCRef {
    std::vector<PreviewPoint> hp;     // curvature + v_ref, length >= N
    std::vector<double> ey_ref;       // lateral reference ey[k], size N+1
    double ey_ref_N{0.0};             // terminal lateral reference

    // Lead object (for ACC) – per-step profiles
    std::vector<double>  v_obj;       // lead vehicle speed, size N
    std::vector<uint8_t> has_obj;     // 1 if object present at step, size N

    std::vector<double> epsi_nom;     // nominal heading error (optional)
};

// State of vehicle in the Frenet + body frame used by MPC
struct MPCState {
    double ey    = 0.0;  // lateral error to centerline
    double epsi  = 0.0;  // heading error to centerline
    double vx    = 0.0;  // longitudinal speed (body frame)
    double vy    = 0.0;  // lateral speed (body frame)
    double dpsi  = 0.0;  // yaw rate r
    double delta = 0.0;  // steering angle
    double d     = 1e6;  // distance gap to lead vehicle (if ACC enabled)
};

// Control output from the MPC
struct MPCControl {
    double R      = 0.0;  // rear-axle longitudinal generalized force [N]
    double ddelta = 0.0;  // steering rate [rad/s]
    bool   ok     = false; // solver success flag
};

// ---------------------------------
// LTV MPC Controller
// ---------------------------------
class LTV_MPC {
public:
    explicit LTV_MPC(const MPCParams& p) : P(p) {
        // allocate nominal trajectory buffers
        nom_.x.resize(P.N + 1);
        nom_.u.resize(P.N);
        // pull default tire parameters from global tire_model
        tp_ = dynamics::tire::current();   // default from tire_model
    }

    // Physics / limits setters
    void setVehicleParams(const dynamics::vehicle::Params& vp);  // defined in .cpp
    void setLimits(const dynamics::vehicle::Limits& L);          // defined in .cpp
    void setTireParams(const dynamics::tire::TireParams& tp) { tp_ = tp; }

    // Obstacles / corridor setters
    void setObstacleConstraints(std::optional<MPCObsSet> s) { obs_ = std::move(s); }
    void setCorridorBounds(const std::vector<double>& lo,
                           const std::vector<double>& up);

    // ACC configuration
    void setACCParams(const acc::Params& p) { accp_ = p; }
    const acc::Params& accParams() const { return accp_; }

    // Nominal warm-start for x and u (used by buildLinearization / solveQP if desired)
    void setNominal(const std::vector<MPCState>& x_nom,
                    const std::vector<Eigen::Vector2d>& u_nom) {
        nom_.x = x_nom; nom_.u = u_nom;
    }

    // Build stage-wise LTV model from nonlinear dynamics and reference
    void        buildLinearization(const MPCRef& ref);
    // Assemble and solve QP, returning first control input
    MPCControl  solveQP(const MPCState& x0, const MPCRef& ref);

    // Convenience wrapper: rebuild linearization and solve in one call
    MPCControl solve(const MPCState& x0, const MPCRef& ref) {
        buildLinearization(ref);
        return solveQP(x0, ref);
    }

    // Helpers
    static void angleWrap(double& a);          // wrap angle to (-pi, pi]
    double ax_max(double vx, const ACCBoundCoeffs& coeffs); // speed-dependent accel upper bound
    double ax_min(double vx, const ACCBoundCoeffs& coeffs); // speed-dependent accel lower bound
    double acclimit(const std::vector<double>& coeffs, double vx); // polynomial evaluation

private:
    MPCParams P;  // configuration and weights

    // Shared vehicle + tire model parameters
    dynamics::vehicle::Params  vp_{};   // mass, wheelbase, CG offset, yaw inertia, etc.
    dynamics::vehicle::Limits  lim_{};  // steering & longitudinal actuation limits
    dynamics::tire::TireParams tp_{};   // tire Magic Formula params + friction + unsprung masses
    acc::Params accp_{};                // ACC timing / headway / min distance params

    // Linear time-varying system model over horizon
    struct LinModel {
        std::vector<Eigen::MatrixXd> A; // Ad_k
        std::vector<Eigen::MatrixXd> B; // Bd_k
        std::vector<Eigen::VectorXd> c; // cd_k
    } lm_;

    // Nominal state & input trajectories used e.g. for warm-start or logging
    struct Nominal {
        std::vector<MPCState>        x;  // size N+1
        std::vector<Eigen::Vector2d> u;  // size N
    } nom_;

    // Corridor bounds provided externally, per-step
    std::vector<double> ey_lo_provided_;
    std::vector<double> ey_up_provided_;
    bool have_corridor_bounds_{false};

    // Obstacle set
    std::optional<MPCObsSet> obs_;
};
