#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <cstddef>
#include <optional>

#include "obstacles.hpp"   // MPCObsSet + compute_lateral_bounds

// ---------------------------------
// MPC configuration (kept as-is)
// ---------------------------------
struct MPCParams {
    int    N   = 200;
    double dt  = 0.1;
    double L   = 2.7;

    // weights
    double wy    = 0.25;
    double wpsi  = 0.10;    // heading error (rad^2)  << much larger than before
    double wv    = 0.25;
    double wR    = 1e-5;
    double wdR   = 1e-5;   // slew R    (u_k - u_{k-1})
    double wddR  = 2e-5;    // jerk R    (u_k - 2u_{k-1} + u_{k-2})
    double wdd   = 8.0;   // effort on ddelta
    double wddd  = 1.0;   // slew ddelta
    double wyf   = 3.0;    // terminal ey
    double wpsif = 8.0;    // terminal epsi

    // bounds
    // double a_min = -5.0, a_max = 3.0;
    double ddelta_max = 0.25;     // |steer rate| [rad/s]
    double delta_max  = 0.40;     // steering angle cap [rad]
    double v_min = 0.0,  v_max = 40.0;

    double ey_up_max = 3.0;
    double ey_lo_max = -3.0;

    double ey_max = 1.8;  // fallback if no corridor provided

    // ===== ACC options =====
    bool   acc_enable{true};   // add 5th state (gap) and constraints
    double acc_tau{1.4};       // [s] time headway
    double acc_dmin{5.0};      // [m] standstill gap (a.k.a. jam distance)
};

// Preview point used by your sim
struct PreviewPoint {
    double kappa = 0.0;   // road curvature at s_k
    double v_ref = 0.0;   // desired speed at s_k
};

// Horizon preview (your sim fills ref.hp[i].{kappa,v_ref})
struct MPCRef {
    std::vector<PreviewPoint> hp;  // length N (or >= N)
    std::vector<double> ey_ref;      // size N+1 (state at k=0..N)
    double ey_ref_N{0.0};            // terminal, if you prefer separate

    // Lead object profile (size N). If has_obj[k]==false we skip the ACC constraint at k.
    std::vector<double> v_obj;
    std::vector<uint8_t> has_obj;

    std::vector<double> epsi_nom;
};

// State / Input / Output
struct MPCState {
    double ey    = 0.0;   // lateral error [m]
    double epsi  = 0.0;   // heading error [rad]
    double vx     = 0.0;   // longitudinal speed [m/s]
    double vy    = 0.0;   // lateral speed [m/s]
    double dpsi     = 0.0;   // yaw rate [rad/s]
    double delta = 0.0;   // steering angle [rad]

    double d{1e6};   // longitudinal gap [m] (only used when acc_enable=true)
};

struct MPCControl {
    double R      = 0.0;  // Propulsional force [N]
    double ddelta = 0.0;  // steering rate [rad/s]
    bool   ok     = false;
};

// ---------------------------------
// LTV MPC Controller
// ---------------------------------
class LTV_MPC {
public:
    // explicit LTV_MPC(const MPCParams& p): P(p) {
    //     nom_.x.resize(P.N + 1);
    //     nom_.u.resize(P.N);
    // }

    explicit LTV_MPC(const MPCParams& p) : P(p) {nom_.x.resize(P.N + 1); nom_.u.resize(P.N);}

    // Physics from sim.VehicleParams
    void setVehicleParams(double m, double L, double d, double JG, double m0);

    // Actuator/tire limits from sim.Limits
    void setLimits(double delta_max, double ddelta_max,
                    double R_min, double R_max,
                    double Ffl_max, double Frl_max);

    void setObstacleConstraints(std::optional<MPCObsSet> s) { obs_ = std::move(s); }

    // Optional warm-start
    void setNominal(const std::vector<MPCState>& x_nom,
                    const std::vector<Eigen::Vector2d>& u_nom) {
        nom_.x = x_nom; nom_.u = u_nom;
    }

    // Build linearized discrete model x_{k+1} = A_k x_k + B_k u_k + c_k
    void buildLinearization(const MPCRef& ref);

    // Assemble and solve QP; returns first input to apply
    MPCControl solveQP(const MPCState& x0, const MPCRef& ref);

    // Backward-compatible wrapper (your sim calls mpc.solve(...))
    MPCControl solve(const MPCState& x0, const MPCRef& ref) {
        buildLinearization(ref);
        return solveQP(x0, ref);
    }

    // tire model

    struct TireParams {
        // Simple “pure lateral” Magic Formula form where D = mu * Fz.
        // If you later want full D(Fz), B(Fz), E(Fz), keep these as functions.
        // double Bf{10.0}, Cf{1.3}, Ef{0.97}, muf{1.0};  // front
        // double Br{12.0}, Cr{1.3}, Er{1.00}, mur{1.0};  // rear
        double Bf{7.8727}, Cf{2.5296}, Ef{1.3059}, muf{1.0};  // front
        double Br{7.8727}, Cr{2.5296}, Er{1.3059}, mur{1.0};  // rear
    };

    // in class LTV_MPC public:
    void setTireParams(const TireParams& tp) { tires_ = tp; }

    // helpers
    static void angleWrap(double& a);

    void setCorridorBounds(const std::vector<double>& lo,
        const std::vector<double>& up);

private:
    MPCParams P;
    TireParams tires_;

    // struct LinModel {
    //     std::vector<Eigen::Matrix<double,4,4>> A;  // ey, epsi, v, delta
    //     std::vector<Eigen::Matrix<double,4,2>> B;  // a, ddelta
    //     std::vector<Eigen::Matrix<double,4,1>> c;  // affine
    // } lm_;

    // --- vehicle numbers (from VehicleParams) ---
    double m_{660.0}, L_{3.4}, d_{1.6}, JG_{450.0}, m0_{185.09}; double Lf_, Lr_;

    // --- limits (from Limits) ---
    double delta_max_{0.5}, ddelta_max_{0.7};
    double Rmin_{-10000.0}, Rmax_{5500.0};
    double Ffl_max_{5000.0}, Frl_max_{5500.0};

    struct LinModel { std::vector<Eigen::MatrixXd> A;
                    std::vector<Eigen::MatrixXd> B;
                    std::vector<Eigen::VectorXd> c; } lm_;

    struct Nominal {
        std::vector<MPCState>        x;   // N+1
        std::vector<Eigen::Vector2d> u;   // N
    } nom_;

    std::vector<double> ey_lo_provided_;
    std::vector<double> ey_up_provided_;
    bool have_corridor_bounds_{false};

    std::optional<MPCObsSet> obs_;        // per-step ey half-spaces
};
