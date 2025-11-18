#include "mpc_ltv.hpp"
#include "obstacles.hpp"
#include "tire_model.hpp"     
#include "vehicle_model.hpp"  
#include "acc.hpp"

#include <OsqpEigen/OsqpEigen.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <unsupported/Eigen/MatrixFunctions>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::SparseMatrix;
using Eigen::Triplet;

#ifndef OSQP_INFTY
#define OSQP_INFTY 1e20
#endif

// -----------------------
// small helpers
// -----------------------
static inline double clamp(double x, double a, double b){
    return std::min(std::max(x,a),b);
}

void LTV_MPC::angleWrap(double& a) {
    while (a <= -M_PI) a += 2.0 * M_PI;
    while (a >    M_PI) a -= 2.0 * M_PI;
}

void LTV_MPC::setVehicleParams(const dynamics::vehicle::Params& vp) {
    vp_ = vp;
  }
  
void LTV_MPC::setLimits(const dynamics::vehicle::Limits& L) {
    lim_ = L;
  }

void discretizeZOH(const Eigen::MatrixXd& A,
    const Eigen::MatrixXd& B,         // may be (n×0) if no inputs
    const Eigen::VectorXd& d,         // may be size-0 if no affine term
    double dt,
    Eigen::MatrixXd& Ad,
    Eigen::MatrixXd& Bd,
    Eigen::VectorXd& cd)
{
    using namespace Eigen;
    const int n = static_cast<int>(A.rows());
    const int m = static_cast<int>(B.cols());

    // --- Ad = exp(A*dt)
    Ad = (A * dt).exp();

    // --- Bd via Van Loan block exp: exp([A  B; 0 0] dt) = [Ad  Bd; 0  I]
    if (m > 0) {
    MatrixXd M(n + m, n + m);
    M.setZero();
    M.topLeftCorner(n, n) = A * dt;
    M.topRightCorner(n, m) = B * dt;
    // bottom-right is 0 (whose exp is I)
    MatrixXd expM = M.exp();
    Bd = expM.topRightCorner(n, m);
    } else {
    Bd.resize(n, 0);
    }

    // --- cd via same trick: exp([A  d; 0 0] dt) = [Ad  cd; 0  1]
    if (d.size() == n) {
    MatrixXd Md(n + 1, n + 1);
    Md.setZero();
    Md.topLeftCorner(n, n) = A * dt;
    Md.topRightCorner(n, 1) = d * dt;
    MatrixXd expMd = Md.exp();
    cd = expMd.topRightCorner(n, 1);
    } else {
    cd = VectorXd::Zero(n);
    }
}

// ------------------------------------------------------------------
// buildLinearization: Dynamic bicycle model with lateral tire slip
// States: [ey, epsi, vx, vy, r, delta, (d)]
// Inputs: [R_cmd, ddelta]
//  ey_dot   ≈ vx*sin(epsi) + vy*cos(epsi)
//  epsi_dot ≈ r - kappa_ref*vx
//  vx_dot   = ax
//  vy_dot   = ay
//  r_dot    = rdot
//  v_dot    = a
//  delta_dot= ddelta
//  d_dot    = v_obj - vx   (if ACC enabled)
// Discretization: Zero-Order Hold (ZOH)
// ------------------------------------------------------------------
void LTV_MPC::buildLinearization(const MPCRef& ref) {
    using Eigen::MatrixXd;
    using Eigen::VectorXd;

    const int N  = P.N;
    const bool ACC_ENABLE = accp_.enable;
    const int nx = ACC_ENABLE ? 7 : 6;   // [ey, epsi, vx, vy, r, delta, (d)]
    const int nu = 2;                      // [R, ddelta]

    constexpr int id_ey    = 0;
    constexpr int id_epsi  = 1;
    constexpr int id_vx    = 2;
    constexpr int id_vy    = 3;
    constexpr int id_r     = 4;
    constexpr int id_delta = 5;
    const int     id_d     = ACC_ENABLE ? 6 : -1;

    const int id_R      = 0;;
    const int id_ddelta = 1;

    // allocate horizon containers
    lm_.A.assign(N, MatrixXd::Identity(nx, nx));
    lm_.B.assign(N, MatrixXd::Zero(nx, nu));
    lm_.c.assign(N, VectorXd::Zero(nx));

    // -------- helper: continuous-time f(x,u; ref_k) ----------
    auto f_dyn = [&](const VectorXd& x,
                     const VectorXd& u,
                     const MPCRef& ref_k) -> VectorXd
    {
        // unpack state
        const double ey    = x(id_ey);
        const double epsi  = x(id_epsi);
        double       vx    = x(id_vx);
        const double vy    = x(id_vy);
        const double r     = x(id_r);       // yaw-rate = dpsi
        const double delta = x(id_delta);
        const double d_gap = ACC_ENABLE ? x(id_d) : 0.0; (void)d_gap;

        // guard near-zero forward speed for slip formulas
        vx = std::max(0.1, vx);

        // unpack inputs
        const double R_cmd  = u(id_R);   // rear-axle longitudinal force request
        const double ddelta = u(id_ddelta);   // steering rate

        // tire/body forces + yaw moment using shared physics
        dynamics::tire::VehicleGeom vg{ vp_.m, vp_.L, vp_.d, vp_.JG };
        const auto fr = dynamics::tire::computeForcesBody(
            vx, vy, r, delta, R_cmd, vg, tp_, /*g=*/9.81);

        // rigid-body accelerations (body frame)
        const double ax   = fr.Fx_sum / vp_.m + vy * r;
        const double ay   = fr.Fy_sum / vp_.m - vx * r;
        const double rdot = fr.Mz     / vp_.JG;

        // reference curvature/speed for Frenet kinematics (stage-local)
        const double kappa_ref = ref_k.hp[0].kappa;
        const double v_obj_ref =
            (!ref_k.v_obj.empty() ? ref_k.v_obj[0] : ref_k.hp[0].v_ref);

        VectorXd dx = VectorXd::Zero(ACC_ENABLE ? 7 : 6);

        // Frenet geometry
        dx(id_ey) = vx * std::sin(epsi) + vy * std::cos(epsi);   // e_y_dot
        dx(id_epsi) = r  - kappa_ref * vx;                         // e_psi_dot

        // body-frame RB dynamics
        dx(id_vx) = ax;                                          // v_x_dot
        dx(id_vy) = ay;                                          // v_y_dot
        dx(id_r) = rdot;                                        // yaw-rate dot

        // steering actuator kinematics
        dx(id_delta) = ddelta;                                      // delta dot

        // ACC gap dynamics (simple relative-speed integrator)
        if (ACC_ENABLE) {
            dx(id_d) = v_obj_ref - vx;                          // d_dot
        }

        (void)ey; // used in kinematics above; silence warnings in some builds
        return dx;
    };

    // -------- linearize each stage, then ZOH-discretize --------
    for (int k = 0; k < N; ++k) {
        const int idx = std::min<int>(k, static_cast<int>(ref.hp.size()) - 1);

        // nominal expansion point (x0,u0)
        VectorXd x0 = VectorXd::Zero(nx);
        VectorXd u0 = VectorXd::Zero(nu);

        x0(id_vx) = std::max(0.0, ref.hp[idx].v_ref);  // vx nominal
        x0(id_vy) = 0.0;                               // vy
        x0(id_r) = 0.0;                               // yaw rate r
        x0(id_delta) = 0.0;                               // delta
        if (ACC_ENABLE) x0(id_d) = 0.0;            // gap

        // one-step view for reference fields at stage k
        MPCRef ref_k;
        ref_k.hp.resize(1);
        ref_k.hp[0] = ref.hp[idx];
        if (!ref.v_obj.empty())
            ref_k.v_obj = { ref.v_obj[ std::min<int>(k, (int)ref.v_obj.size()-1) ] };
        if (!ref.has_obj.empty())
            ref_k.has_obj = { ref.has_obj[ std::min<int>(k, (int)ref.has_obj.size()-1) ] };

        // base vector field and finite-diff Jacobians
        const VectorXd f0 = f_dyn(x0, u0, ref_k);

        MatrixXd A = MatrixXd::Zero(nx, nx);
        MatrixXd B = MatrixXd::Zero(nx, nu);
        VectorXd d = VectorXd::Zero(nx);

        const double eps = 1e-6;

        // df/dx
        for (int i = 0; i < nx; ++i) {
            VectorXd x_eps = x0; x_eps(i) += eps;
            const VectorXd f_eps = f_dyn(x_eps, u0, ref_k);
            A.col(i) = (f_eps - f0) / eps;
        }

        // df/du
        for (int j = 0; j < nu; ++j) {
            VectorXd u_eps = u0; u_eps(j) += eps;
            const VectorXd f_eps = f_dyn(x0, u_eps, ref_k);
            B.col(j) = (f_eps - f0) / eps;
        }

        // affine term c: f ≈ A(x-x0) + B(u-u0) + c  =>  c = f0 - A x0 - B u0
        d = f0 - A * x0 - B * u0;

        // ZOH discretization
        MatrixXd Ad, Bd;
        VectorXd cd;
        discretizeZOH(A, B, d, P.dt, Ad, Bd, cd);

        lm_.A[k] = Ad;
        lm_.B[k] = Bd;
        lm_.c[k] = cd;
    }

    // ---- (optional) set nominal rollouts used elsewhere ----
    nom_.x.resize(N + 1);
    nom_.u.resize(N);
    for (int k = 0; k <= N; ++k) {
        nom_.x[k].ey    = 0.0;
        nom_.x[k].epsi  = 0.0;
        nom_.x[k].vx    = (k < (int)ref.hp.size()) ? ref.hp[k].v_ref : 0.0;
        nom_.x[k].vy    = 0.0;
        nom_.x[k].dpsi  = 0.0;
        nom_.x[k].delta = 0.0;
        if (ACC_ENABLE) nom_.x[k].d = 0.0;
        if (k < N) nom_.u[k].setZero();
    }
}

void LTV_MPC::setCorridorBounds(const std::vector<double>& lo,
    const std::vector<double>& up) {
    const int N = P.N;
    ey_lo_provided_.assign(N, -OSQP_INFTY);
    ey_up_provided_.assign(N, +OSQP_INFTY);
    const int steps = std::min<int>(N, std::min(lo.size(), up.size()));
    for (int i=0; i<steps; ++i) { ey_lo_provided_[i] = lo[i]; ey_up_provided_[i] = up[i]; }
    have_corridor_bounds_ = true;
}


// ------------------------------------------------------------------
// solveQP: assemble constraints/cost and solve (first input returned)
// Uses triplet assembly (no sparse block assignment).
// ------------------------------------------------------------------
MPCControl LTV_MPC::solveQP(const MPCState& x0, const MPCRef& ref) {
    using Eigen::VectorXd;
    using Eigen::SparseMatrix;
    using Eigen::Triplet;

    const bool ACC_ENABLE = accp_.enable;
    const int nx = ACC_ENABLE ? 7 : 6;   // [ey, epsi, vx, vy, r, delta, (d)]
    const int nu = 2;                      // [R, ddelta]
    const int N  = P.N;

    // canonical indices in the state vector
    constexpr int id_ey    = 0;
    constexpr int id_epsi  = 1;
    constexpr int id_vx    = 2;
    constexpr int id_vy    = 3;
    constexpr int id_r     = 4;
    constexpr int id_delta = 5;
    const int     id_d     = ACC_ENABLE ? 6 : -1;

    const int id_R      = 0;;
    const int id_ddelta = 1;

    // stacked decision sizes
    const int NX = (N + 1) * nx;
    const int NU = N * nu;
    const int NZ = NX + NU;

    auto idx_x = [&](int k, int i) { return k * nx + i; };        // 0..NX-1
    auto idx_u = [&](int k, int j) { return NX + k * nu + j; };   // NX..NX+NU-1

    // -----------------------------
    // Corridor bounds for ey
    // -----------------------------
    std::vector<double> ey_upper(N, +OSQP_INFTY);
    std::vector<double> ey_lower(N, -OSQP_INFTY);

    if (have_corridor_bounds_) {
        for (int k = 0; k < N; ++k) {
            double lo = (k < (int)ey_lo_provided_.size()) ? ey_lo_provided_[k] : -OSQP_INFTY;
            double up = (k < (int)ey_up_provided_.size()) ? ey_up_provided_[k] : +OSQP_INFTY;
            if (!std::isfinite(lo)) lo = -OSQP_INFTY;
            if (!std::isfinite(up)) up = +OSQP_INFTY;
            ey_lower[k] = lo;
            ey_upper[k] = up;
        }
    } else {
        for (int k = 0; k < N; ++k) {
            ey_lower[k] = -P.ey_max;
            ey_upper[k] = +P.ey_max;
        }
    }

    // ============================
    // COST: H (symmetric) and g
    // ============================
    std::vector<Triplet<double>> Ht;
    Ht.reserve((NX + NU) * 6);
    VectorXd g = VectorXd::Zero(NZ);

    // stage costs
    for (int k = 0; k < N; ++k) {
        const int idx = std::min<int>(k, (int)ref.hp.size() - 1);
        const double vref  = (idx >= 0) ? ref.hp[idx].v_ref : 0.0;
        const double eyref = (k < (int)ref.ey_ref.size()) ? ref.ey_ref[k] : 0.0;

        // ey tracking
        Ht.emplace_back(idx_x(k, id_ey), idx_x(k, id_ey), 2.0 * P.wy);
        g(idx_x(k, id_ey)) += -2.0 * P.wy * eyref;

        // epsi^2
        Ht.emplace_back(idx_x(k, id_epsi), idx_x(k, id_epsi), 2.0 * P.wpsi);

        // (vx - vref)^2
        Ht.emplace_back(idx_x(k, id_vx), idx_x(k, id_vx), 2.0 * P.wv);
        g(idx_x(k, id_vx)) += -2.0 * P.wv * vref;

        // input penalties
        Ht.emplace_back(idx_u(k, id_R), idx_u(k, 0), 2.0 * P.wR);
        Ht.emplace_back(idx_u(k, id_ddelta), idx_u(k, 1), 2.0 * P.wdd);

        // input slew (u_k - u_{k-1})^2
        if (k > 0) {
            for (int j = 0; j < nu; ++j) {
                const int uk   = idx_u(k, j);
                const int ukm1 = idx_u(k - 1, j);
                const double w = (j == 0 ? P.wdR : P.wddd);
                Ht.emplace_back(uk,    uk,    2.0 * w);
                Ht.emplace_back(ukm1,  ukm1,  2.0 * w);
                Ht.emplace_back(uk,    ukm1, -2.0 * w);
                Ht.emplace_back(ukm1,  uk,   -2.0 * w);
            }
        }
    }

    // optional propulsion jerk: (R_k - 2R_{k-1} + R_{k-2})^2
    if (P.wddR > 0.0) {
        for (int kk = 2; kk < N; ++kk) {
            const int Rk   = idx_u(kk, 0);
            const int Rkm1 = idx_u(kk - 1, 0);
            const int Rkm2 = idx_u(kk - 2, 0);
            const double a = 1.0, b = -2.0, c = 1.0;
            const double s = 2.0 * P.wddR;
            Ht.emplace_back(Rk,   Rk,   s * a * a);
            Ht.emplace_back(Rkm1, Rkm1, s * b * b);
            Ht.emplace_back(Rkm2, Rkm2, s * c * c);

            Ht.emplace_back(Rk,   Rkm1, s * a * b);
            Ht.emplace_back(Rkm1, Rk,   s * a * b);
            Ht.emplace_back(Rk,   Rkm2, s * a * c);
            Ht.emplace_back(Rkm2, Rk,   s * a * c);
            Ht.emplace_back(Rkm1, Rkm2, s * b * c);
            Ht.emplace_back(Rkm2, Rkm1, s * b * c);
        }
    }

    // terminal costs
    Ht.emplace_back(idx_x(N, id_ey),   idx_x(N, id_ey),   2.0 * P.wyf);
    g(idx_x(N, id_ey))   += -2.0 * P.wyf * ref.ey_ref_N;
    Ht.emplace_back(idx_x(N, id_epsi), idx_x(N, id_epsi), 2.0 * P.wpsif);

    SparseMatrix<double> H(NZ, NZ);
    H.setFromTriplets(Ht.begin(), Ht.end());

    // ============================
    // EQUALITY: dynamics + x0
    // ============================
    const int meq_rows = N * nx + nx;
    std::vector<Triplet<double>> Aeqt; Aeqt.reserve(meq_rows * (nx + nu));
    VectorXd beq = VectorXd::Zero(meq_rows);

    auto row_dyn = [&](int k, int i) { return k * nx + i; };   // 0 .. N*nx-1
    auto row_x0  = [&](int i)       { return N * nx + i; };

    for (int k = 0; k < N; ++k) {
        const auto& Ad = lm_.A[k];
        const auto& Bd = lm_.B[k];
        const auto& cd = lm_.c[k];

        for (int i = 0; i < nx; ++i) {
            Aeqt.emplace_back(row_dyn(k, i), idx_x(k + 1, i), 1.0);
            for (int j = 0; j < nx; ++j) {
                const double v = -Ad(i, j);
                if (v != 0.0) Aeqt.emplace_back(row_dyn(k, i), idx_x(k, j), v);
            }
            for (int j = 0; j < nu; ++j) {
                const double v = -Bd(i, j);
                if (v != 0.0) Aeqt.emplace_back(row_dyn(k, i), idx_u(k, j), v);
            }
            beq(row_dyn(k, i)) = -cd(i);
        }
    }

    // initial condition
    beq(row_x0(id_ey))     = x0.ey;     Aeqt.emplace_back(row_x0(id_ey),     idx_x(0, id_ey),     1.0);
    beq(row_x0(id_epsi))   = x0.epsi;   Aeqt.emplace_back(row_x0(id_epsi),   idx_x(0, id_epsi),   1.0);
    beq(row_x0(id_vx))     = x0.vx;     Aeqt.emplace_back(row_x0(id_vx),     idx_x(0, id_vx),     1.0);
    beq(row_x0(id_vy))     = x0.vy;     Aeqt.emplace_back(row_x0(id_vy),     idx_x(0, id_vy),     1.0);
    beq(row_x0(id_r))      = x0.dpsi;   Aeqt.emplace_back(row_x0(id_r),      idx_x(0, id_r),      1.0);
    beq(row_x0(id_delta))  = x0.delta;  Aeqt.emplace_back(row_x0(id_delta),  idx_x(0, id_delta),  1.0);
    if (ACC_ENABLE) {
        beq(row_x0(id_d)) = x0.d;
        // std::cout << "x0.d: " << x0.d << std::endl;
        Aeqt.emplace_back(row_x0(id_d), idx_x(0, id_d), 1.0);
    }

    // ============================
    // INEQUALITIES
    // ============================
    std::vector<Triplet<double>> Aint;
    std::vector<double> lin_v, uin_v;
    auto push_row_le = [&](int row, int col, double coeff, double lo, double hi) {
        Aint.emplace_back(row, col, coeff);
        lin_v.push_back(lo);
        uin_v.push_back(hi);
    };

    int row = 0;

    // (1) ey corridor for k = 0..N-1
    for (int k = 0; k < N; ++k) {
        if (ey_upper[k] < +OSQP_INFTY) {            // ey_k <= up
            push_row_le(row, idx_x(k, id_ey), +1.0, -OSQP_INFTY, ey_upper[k]);
            ++row;
        }
        if (ey_lower[k] > -OSQP_INFTY) {            // -ey_k <= -lo
            push_row_le(row, idx_x(k, id_ey), -1.0, -OSQP_INFTY, -ey_lower[k]);
            ++row;
        }
    }

    // (2) input boxes
    for (int k = 0; k < N; ++k) {
        push_row_le(row, idx_u(k, id_R), 1.0, lim_.R_min,        lim_.R_max);        ++row;  // R
        push_row_le(row, idx_u(k, id_ddelta), 1.0, -lim_.ddelta_max,  lim_.ddelta_max);   ++row;  // ddelta
    }

    // (3) steering angle bounds
    for (int k = 0; k <= N; ++k) {
        push_row_le(row, idx_x(k, id_delta), 1.0, -lim_.delta_max, lim_.delta_max);
        ++row;
    }

    // (4) speed bounds (optional)
    for (int k = 0; k <= N; ++k) {
        push_row_le(row, idx_x(k, id_vx), 1.0, P.v_min, P.v_max);
        ++row;
    }

    // (5) simple lateral capacity guard using static axle loads + mu
    {
        const double g = 9.81;
        const double Fzf0 = vp_.m * g * (vp_.L - vp_.d) / vp_.L + tp_.m_unsprung_front * g;  // front axle
        const double Fzr0 = vp_.m * g * (vp_.d)        / vp_.L + tp_.m_unsprung_rear * g;    // rear axle
        const double Fy_front_peak = 2.0 * (tp_.muf * Fzf0);
        const double Fy_rear_peak  = 2.0 * (tp_.mur * Fzr0);
        const double Fy_cap = Fy_front_peak + Fy_rear_peak;

        for (int k = 0; k < N; ++k) {
            const int idx = std::min<int>(k, (int)ref.hp.size() - 1);
            const double vref = (idx >= 0) ? ref.hp[idx].v_ref : 0.0;
            const double coeff = vp_.m * (vref * vref) / std::max(1e-6, vp_.L);

            // +coeff * delta_k <= Fy_cap
            Aint.emplace_back(row, idx_x(k, id_delta), +coeff);
            lin_v.push_back(-OSQP_INFTY);
            uin_v.push_back(Fy_cap);
            ++row;

            // -coeff * delta_k <= Fy_cap
            Aint.emplace_back(row, idx_x(k, id_delta), -coeff);
            lin_v.push_back(-OSQP_INFTY);
            uin_v.push_back(Fy_cap);
            ++row;
        }
    }

    // (6) ACC: d >= 0 and time-headway constraint if lead present
    if (ACC_ENABLE) {
        for (int k = 0; k <= N; ++k) {
            push_row_le(row, idx_x(k, id_d), 1.0, 0.0, +OSQP_INFTY);
            ++row;
        }
        for (int k = 0; k < N; ++k) {
            const bool has = (k < (int)ref.has_obj.size()) ? (ref.has_obj[k] != 0) : false;
            if (!has) continue;
    
            Aint.emplace_back(row, idx_x(k, id_d),  +1.0);
            Aint.emplace_back(row, idx_x(k, id_vx), -accp_.tau);   // <-- see note 2
            lin_v.push_back(accp_.dmin);                           // <-- see note 2
            // std::cout << "accp_.dmin: " << accp_.dmin << std::endl;
            uin_v.push_back(+OSQP_INFTY);
            ++row;
        }
    }

    // Stack equalities + inequalities
    const int meq   = (int)beq.size();
    const int mineq = (int)lin_v.size();
    const int m     = meq + mineq;

    std::vector<Triplet<double>> Atrips;
    Atrips.reserve(Aeqt.size() + Aint.size());
    Atrips.insert(Atrips.end(), Aeqt.begin(), Aeqt.end());
    for (const auto& t : Aint) Atrips.emplace_back(t.row() + meq, t.col(), t.value());

    SparseMatrix<double> A(m, NZ);
    A.setFromTriplets(Atrips.begin(), Atrips.end());

    VectorXd l(m), u(m);
    l.head(meq) = beq;
    u.head(meq) = beq;
    for (int i = 0; i < mineq; ++i) {
        l(meq + i) = lin_v[i];
        u(meq + i) = uin_v[i];
    }

    // -----------------------------
    // Solve with OSQP
    // -----------------------------
    OsqpEigen::Solver solver;
    solver.settings()->setWarmStart(true);
    solver.settings()->setVerbosity(false);
    solver.settings()->setAlpha(1.6);
    solver.data()->setNumberOfVariables(NZ);
    solver.data()->setNumberOfConstraints(m);

    if (!solver.data()->setHessianMatrix(H))       return {};
    if (!solver.data()->setGradient(g))            return {};
    if (!solver.data()->setLinearConstraintsMatrix(A)) return {};
    if (!solver.data()->setLowerBound(l))          return {};
    if (!solver.data()->setUpperBound(u))          return {};
    if (!solver.initSolver())                      return {};

    if (solver.solveProblem() != OsqpEigen::ErrorExitFlag::NoError) return {};

    const VectorXd z = solver.getSolution();
    MPCControl out;
    out.R      = z(idx_u(0, id_R));
    out.ddelta = z(idx_u(0, id_ddelta));
    out.ok     = true;
    return out;
}
