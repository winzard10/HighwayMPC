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
// buildLinearization: simple kinematic bicycle linearization
// States: [ey, epsi, v, delta], Inputs: [a, ddelta]
//  ey_dot   ≈ v * epsi
//  epsi_dot ≈ (v/L) * delta - v * kappa
//  v_dot    = a
//  delta_dot= ddelta
// Discretization: Zero-Order Hold (ZOH)
// ------------------------------------------------------------------
// dynamics with no slip — cleaner version using generic f(x,u)

void LTV_MPC::buildLinearization(const MPCRef& ref) {
    using Eigen::MatrixXd;
    using Eigen::VectorXd;

    const int N  = P.N;
    const bool ACC_ENABLE = accp_.enable;
    const int nx = ACC_ENABLE ? 5 : 4;   // [ey, epsi, v, delta, (d)]
    const int nu = 2;                    // [R, ddelta]

    // allocate horizon containers
    lm_.A.assign(N, MatrixXd::Identity(nx, nx));
    lm_.B.assign(N, MatrixXd::Zero(nx, nu));
    lm_.c.assign(N, VectorXd::Zero(nx));

    // ------------ helper: longitudinal vdot (same as your current code) ------------
    auto vdot_scalar = [&](double v, double delta, double ddelta, double R) -> double {
        const double c  = std::cos(delta);
        const double s2 = 1.0 / (c * c);           // sec^2
        const double t  = std::tan(delta);
        const double denom    = (vp_.m + vp_.m0 * t * t);
        const double coupling = (vp_.m0 * t * s2) * ddelta * v; // Fwind = 0
        return (R - coupling) / denom;                           // Eq.(6)
    };

    // ------------ helper: continuous-time f(x,u; ref_k) for no-slip model ------------
    auto f_dyn = [&](const VectorXd& x,
                     const VectorXd& u,
                     const MPCRef& ref_k) -> VectorXd
    {
        // unpack state
        const double ey    = x(0);
        const double epsi  = x(1);
        const double v     = x(2);
        const double delta = x(3);
        const double d_gap = ACC_ENABLE ? x(4) : 0.0; (void)d_gap; // gap only used for ACC

        // unpack inputs
        const double R_cmd  = u(0);   // rear longitudinal force
        const double ddelta = u(1);   // steering rate

        // reference info for this stage
        const double kappa_ref = ref_k.hp[0].kappa;
        const double v_obj_ref =
            (!ref_k.v_obj.empty() ? ref_k.v_obj[0] : ref_k.hp[0].v_ref);

        VectorXd dx = VectorXd::Zero(nx);

        // Frenet kinematics (no-slip approximation)
        dx(0) = v * epsi;                      // e_y_dot
        dx(1) = (v / vp_.L) * delta - v * kappa_ref; // e_psi_dot

        // longitudinal dynamics
        dx(2) = vdot_scalar(v, delta, ddelta, R_cmd);   // v_dot

        // steering actuator
        dx(3) = ddelta;                      // delta_dot

        // ACC gap dynamics
        if (ACC_ENABLE) {
            dx(4) = v_obj_ref - v;          // d_dot
        }

        (void)ey; // used in dx(0), but this silences some compilers
        return dx;
    };

    // ------------ linearize each stage, then ZOH-discretize ------------
    for (int k = 0; k < N; ++k) {
        const int idx = std::min<int>(k, static_cast<int>(ref.hp.size()) - 1);

        // nominal expansion point (x0,u0) from reference
        VectorXd x0 = VectorXd::Zero(nx);
        VectorXd u0 = VectorXd::Zero(nu);

        // states around nominal straight-lane tracking
        x0(2) = std::max(0.0, ref.hp[idx].v_ref);  // v nominal
        x0(3) = 0.0;                               // delta
        if (ACC_ENABLE) x0(4) = 0.0;              // gap (relative distance offset)

        // inputs nominally zero
        // u0(0) = 0 (R_cmd); u0(1) = 0 (ddelta) already by Zero()

        // build a "single-step" reference for this stage
        MPCRef ref_k;
        ref_k.hp.resize(1);
        ref_k.hp[0] = ref.hp[idx];
        if (!ref.v_obj.empty()) {
            ref_k.v_obj = { ref.v_obj[ std::min<int>(k,
                                 static_cast<int>(ref.v_obj.size() - 1)) ] };
        }

        // base vector field and finite-diff Jacobians
        const VectorXd f0 = f_dyn(x0, u0, ref_k);

        MatrixXd A = MatrixXd::Zero(nx, nx);
        MatrixXd B = MatrixXd::Zero(nx, nu);
        VectorXd d = VectorXd::Zero(nx);

        const double eps = 1e-6;

        // df/dx
        for (int i = 0; i < nx; ++i) {
            VectorXd x_eps = x0;
            x_eps(i) += eps;
            const VectorXd f_eps = f_dyn(x_eps, u0, ref_k);
            A.col(i) = (f_eps - f0) / eps;
        }

        // df/du
        for (int j = 0; j < nu; ++j) {
            VectorXd u_eps = u0;
            u_eps(j) += eps;
            const VectorXd f_eps = f_dyn(x0, u_eps, ref_k);
            B.col(j) = (f_eps - f0) / eps;
        }

        // affine term: f ≈ A(x-x0) + B(u-u0) + d
        d = f0 - A * x0 - B * u0;

        // ZOH discretization
        MatrixXd Ad, Bd;
        VectorXd cd;
        discretizeZOH(A, B, d, P.dt, Ad, Bd, cd);

        lm_.A[k] = Ad;
        lm_.B[k] = Bd;
        lm_.c[k] = cd;
    }

    // ------------ nominal rollout (same idea as before) ------------
    nom_.x.resize(N + 1);
    nom_.u.resize(N);
    for (int k = 0; k <= N; ++k) {
        nom_.x[k].ey    = 0.0;
        nom_.x[k].epsi  = 0.0;
        nom_.x[k].v     = (k < (int)ref.hp.size())
                            ? ref.hp[k].v_ref
                            : (ref.hp.empty() ? 0.0 : ref.hp.back().v_ref);
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
    // const int nx = 4;   // [ey, epsi, v, delta]
    const bool ACC_ENABLE = accp_.enable;
    const int nx = ACC_ENABLE ? 5 : 4;   // [ey, epsi, v, delta, (d)]
    const int nu = 2;   // [a, ddelta]
    const int N  = P.N;
    const int id_ey = 0, id_epsi = 1, id_v = 2, id_delta = 3;
    const int id_d  = ACC_ENABLE ? 4 : -1;

    const int NX = (N+1)*nx;
    const int NU = N*nu;
    const int NZ = NX + NU;

    // indices
    auto idx_x = [&](int k,int i){ return k*nx + i; };       // 0..NX-1
    auto idx_u = [&](int k,int j){ return NX + k*nu + j; };  // NX..NX+NU-1

    // -----------------------------
    // Bounds for ey (from corridor)
    // -----------------------------
    std::vector<double> ey_upper(N, +OSQP_INFTY);
    std::vector<double> ey_lower(N, -OSQP_INFTY);

    if (have_corridor_bounds_) {
        for (int k = 0; k < N; ++k) {
            double lo = ey_lo_provided_.size() > k ? ey_lo_provided_[k] : -OSQP_INFTY;
            double up = ey_up_provided_.size() > k ? ey_up_provided_[k] : +OSQP_INFTY;

            // if NaN, treat as unbounded
            if (!std::isfinite(lo)) lo = -OSQP_INFTY;
            if (!std::isfinite(up)) up = +OSQP_INFTY;

            ey_lower[k] = lo;
            ey_upper[k] = up;
        }
    } else {
        // fallback to global road bounds if no corridor provided
        for (int k = 0; k < N; ++k) {
            ey_lower[k] = -P.ey_max;
            ey_upper[k] = +P.ey_max;
        }
    }

    // ============================
    // COST: H (symmetric) and g
    // ============================
    std::vector<Triplet<double>> Ht;
    Ht.reserve( (NX + NU) * 6 );
    VectorXd g = VectorXd::Zero(NZ);

    // stage costs (k = 0..N-1)
    for (int k=0; k<N; ++k){
        const int idx = std::min<int>(k, (int)ref.hp.size()-1);
        const double vref  = (idx >= 0) ? ref.hp[idx].v_ref : 0.0;
        const double eyref = (k < (int)ref.ey_ref.size()) ? ref.ey_ref[k] : 0.0;

        // (ey_k - eyref)^2
        Ht.emplace_back(idx_x(k,0), idx_x(k,0), 2.0 * P.wy);
        g(idx_x(k,0)) += -2.0 * P.wy * eyref;

        // epsi_k^2
        Ht.emplace_back(idx_x(k,1), idx_x(k,1), 2.0 * P.wpsi);

        // (v_k - vref)^2
        Ht.emplace_back(idx_x(k,2), idx_x(k,2), 2.0 * P.wv);
        g(idx_x(k,2)) += -2.0 * P.wv * vref;

        // input effort
        Ht.emplace_back(idx_u(k,0), idx_u(k,0), 2.0 * P.wR);
        Ht.emplace_back(idx_u(k,1), idx_u(k,1), 2.0 * P.wdd);

        // input slew (u_k - u_{k-1})^2
        if (k>0){
            for (int j=0;j<nu;++j){
                const int uk   = idx_u(k,j);
                const int ukm1 = idx_u(k-1,j);
                const double w = (j==0? P.wdR : P.wddd);
                Ht.emplace_back(uk,    uk,    2.0*w);
                Ht.emplace_back(ukm1,  ukm1,  2.0*w);
                Ht.emplace_back(uk,    ukm1, -2.0*w);
                Ht.emplace_back(ukm1,  uk,   -2.0*w);
            }
        }

        // --- propulsion "jerk": (R_k - 2 R_{k-1} + R_{k-2})^2
        if (P.wddR > 0.0) {
            for (int k = 2; k < N; ++k) {
                const int Rk   = idx_u(k,   0);  // R_k
                const int Rkm1 = idx_u(k-1, 0);  // R_{k-1}
                const int Rkm2 = idx_u(k-2, 0);  // R_{k-2}

                // coefficients of the second difference: [1, -2, 1]
                const double a =  1.0, b = -2.0, c = 1.0;
                const double s = 2.0 * P.wddR;     // factor for Hessian (2 * w)

                // diagonals
                Ht.emplace_back(Rk,   Rk,   s * a*a);     // +2w * 1
                Ht.emplace_back(Rkm1, Rkm1, s * b*b);     // +2w * 4
                Ht.emplace_back(Rkm2, Rkm2, s * c*c);     // +2w * 1

                // off-diagonals (symmetric)
                Ht.emplace_back(Rk,   Rkm1, s * a*b);     // -4w
                Ht.emplace_back(Rkm1, Rk,   s * a*b);

                Ht.emplace_back(Rk,   Rkm2, s * a*c);     // +2w
                Ht.emplace_back(Rkm2, Rk,   s * a*c);

                Ht.emplace_back(Rkm1, Rkm2, s * b*c);     // -4w
                Ht.emplace_back(Rkm2, Rkm1, s * b*c);
                // no linear term because the reference for R is 0 by default
            }
        }
    }

    // terminal ey/epsi
    Ht.emplace_back(idx_x(N,0), idx_x(N,0), 2.0 * P.wyf);
    g(idx_x(N,0)) += -2.0 * P.wyf * ref.ey_ref_N;

    Ht.emplace_back(idx_x(N,1), idx_x(N,1), 2.0 * P.wpsif);

    // finally build H
    SparseMatrix<double> H(NZ, NZ);
    H.setFromTriplets(Ht.begin(), Ht.end());

    // ============================
    // CONSTRAINTS
    // ============================
    // Equality: dynamics + initial condition
    const int meq_rows = N*nx + nx;
    std::vector<Triplet<double>> Aeqt; Aeqt.reserve(meq_rows * (nx + nu));
    VectorXd beq = VectorXd::Zero(meq_rows);


    auto row_dyn = [&](int k, int i){ return k*nx + i; };      // rows 0 .. N*nx-1
    auto row_x0  = [&](int i){ return N*nx + i; };             // rows N*nx .. N*nx+nx-1

    // dynamics: x_{k+1} = Ad x_k + Bd u_k + cd
    for (int k=0; k<N; ++k){
        const auto& Ad = lm_.A[k];
        const auto& Bd = lm_.B[k];
        const auto& cd = lm_.c[k];

        for (int i=0; i<nx; ++i){
            // x_{k+1,i}
            Aeqt.emplace_back(row_dyn(k,i), idx_x(k+1,i), 1.0);
            // -Ad * x_k
            for (int j=0; j<nx; ++j){
                const double val = -Ad(i,j);
                if (val != 0.0) Aeqt.emplace_back(row_dyn(k,i), idx_x(k,j), val);
            }
            // -Bd * u_k
            for (int j=0; j<nu; ++j){
                const double val = -Bd(i,j);
                if (val != 0.0) Aeqt.emplace_back(row_dyn(k,i), idx_u(k,j), val);
            }
            // rhs = -cd
            beq(row_dyn(k,i)) = -cd(i);
        }
    }
    // initial condition x_0 = x0
    beq(row_x0(0)) = x0.ey;   Aeqt.emplace_back(row_x0(0), idx_x(0,0), 1.0);
    beq(row_x0(1)) = x0.epsi; Aeqt.emplace_back(row_x0(1), idx_x(0,1), 1.0);
    beq(row_x0(2)) = x0.v;    Aeqt.emplace_back(row_x0(2), idx_x(0,2), 1.0);
    beq(row_x0(3)) = x0.delta;Aeqt.emplace_back(row_x0(3), idx_x(0,3), 1.0);

    if (ACC_ENABLE) {
        beq(row_x0(4)) = x0.d; Aeqt.emplace_back(row_x0(4), idx_x(0,id_d), 1.0);
    }

    // Inequalities: input bounds, delta bounds, v bounds, ey corridor
    std::vector<Triplet<double>> Aint;
    std::vector<double> lin_v, uin_v;

    auto push_row_le = [&](int row, int col, double coeff, double lo, double hi){
        Aint.emplace_back(row, col, coeff); lin_v.push_back(lo); uin_v.push_back(hi);
    };

    int row = 0;

    // ey corridor for k = 0..N-1  :  ey_lower[k] <= ey_k <= ey_upper[k]
    for (int k=0; k<N; ++k){
        if (ey_upper[k] < +OSQP_INFTY) { // ey_k <= up
            push_row_le(row, idx_x(k,0), +1.0, -OSQP_INFTY, ey_upper[k]); ++row;
        }
        if (ey_lower[k] > -OSQP_INFTY) { // ey_k >= lo  ->  -ey_k <= -lo
            push_row_le(row, idx_x(k,0), -1.0, -OSQP_INFTY, -ey_lower[k]); ++row;
        }
    }

    // input box
    for (int k=0; k<N; ++k){
        push_row_le(row, idx_u(k,0), 1.0, lim_.R_min, lim_.R_max);               ++row; // R
        push_row_le(row, idx_u(k,1), 1.0, -lim_.ddelta_max, lim_.ddelta_max);    ++row; // δ̇
    }
    
    // steering angle bounds
    for (int k=0; k<=N; ++k){
        push_row_le(row, idx_x(k,3), 1.0, -lim_.delta_max, lim_.delta_max);      ++row; // δ
    }

    // v bounds (optional)
    for (int k=0; k<=N; ++k){
        push_row_le(row, idx_x(k,2), 1.0, P.v_min, P.v_max); ++row;
    }

    // --- tire force linearized inequalities
    auto force_pair = [&](double v, double delta, double ddelta, double R) {
        dynamics::tire::VehicleGeom vg{ vp_.m, vp_.L, vp_.d, vp_.JG, vp_.m0};
        const auto fr = dynamics::tire::computeForcesBody(
            v, delta, R, ddelta, vg, /*g=*/9.81);     
        return std::pair<double,double>(fr.Fy_f_body, fr.Fy_r_body); // {front, rear}
    };
    
    auto force_jac = [&](double v, double delta, double ddelta, double R) {
        const double eps = 1e-6;
        auto F0   = force_pair(v, delta, ddelta, R);
        auto F_v  = force_pair(v + eps, delta, ddelta, R);
        auto F_de = force_pair(v, delta + eps, ddelta, R);
        auto F_dd = force_pair(v, delta, ddelta + eps, R);
        auto F_R  = force_pair(v, delta, ddelta, R + eps);
    
        // rows: [front; rear], cols: [v, delta, ddelta, R]
        Eigen::Matrix<double,2,4> J;
        J(0,0) = (F_v.first   - F0.first  ) / eps;  J(1,0) = (F_v.second   - F0.second  ) / eps;
        J(0,1) = (F_de.first  - F0.first  ) / eps;  J(1,1) = (F_de.second  - F0.second  ) / eps;
        J(0,2) = (F_dd.first  - F0.first  ) / eps;  J(1,2) = (F_dd.second  - F0.second  ) / eps;
        J(0,3) = (F_R.first   - F0.first  ) / eps;  J(1,3) = (F_R.second   - F0.second  ) / eps;
    
        return std::make_pair(F0, J);
    };
    
    for (int k = 0; k < N; ++k) {
        // linearize around the same reference you used in buildLinearization
        const int idx = std::min<int>(k, (int)ref.hp.size() - 1);
        const double vref = (idx >= 0) ? ref.hp[idx].v_ref : 0.0;
        const double deltaref = 0.0;   // if you keep δ_nom = 0
        const double ddref    = 0.0;
        const double Rref     = 0.0;
    
        auto FJ = force_jac(vref, deltaref, ddref, Rref);
        const double Ffl0 = FJ.first.first;      // front force at ref
        const double Frl0 = FJ.first.second;     // rear  force at ref
        const auto&  J    = FJ.second;           // 2x4 jacobian
    
        // helper to push one affine inequality: alpha*z <= beta
        auto push_affine = [&](int which, double Fmax){
            // which: 0 = front, 1 = rear
            const double jv   = J(which, 0);
            const double jdel = J(which, 1);
            const double jdd  = J(which, 2);
            const double jR   = J(which, 3);
            const double F0   = (which == 0) ? Ffl0 : Frl0;
    
            // +Jv*v_k + Jdel*delta_k + Jdd*ddelta_k + JR*R_k <= Fmax - F0
            Aint.emplace_back(row, idx_x(k, id_v),     jv);
            Aint.emplace_back(row, idx_x(k, id_delta), jdel);
            Aint.emplace_back(row, idx_u(k, 1),        jdd);
            Aint.emplace_back(row, idx_u(k, 0),        jR);
            lin_v.push_back(-OSQP_INFTY);
            uin_v.push_back(Fmax - F0);
            ++row;
    
            // and symmetric: -F(x,u) <= Fmax  → negate coeffs, RHS = Fmax + F0
            Aint.emplace_back(row, idx_x(k, id_v),     -jv);
            Aint.emplace_back(row, idx_x(k, id_delta), -jdel);
            Aint.emplace_back(row, idx_u(k, 1),        -jdd);
            Aint.emplace_back(row, idx_u(k, 0),        -jR);
            lin_v.push_back(-OSQP_INFTY);
            uin_v.push_back(Fmax + F0);
            ++row;
        };
    
        push_affine(0, lim_.Ffl_max);  // front
        push_affine(1, lim_.Frl_max);  // rear
    }

    // Gap lower bound d >= 0 (optional but nice to keep feasibility)
    if (ACC_ENABLE){
        for (int k=0; k<=N; ++k){
            push_row_le(row, idx_x(k,id_d), 1.0, 0.0, +OSQP_INFTY); ++row;
        }
        // ACC safety: d_k - tau * v_k >= d_min  (only if there's a lead at step k)
        for (int k=0; k<N; ++k){
            const bool has = (ref.has_obj.size() > (size_t)k) ? (ref.has_obj[k]!=0) : false;
            if (!has) continue;
            // Aineq[row, id_d] = +1, Aineq[row, id_v] = -tau
            Aint.emplace_back(row, idx_x(k,id_d),  +1.0);
            Aint.emplace_back(row, idx_x(k,id_v),  -accp_.tau);
            lin_v.push_back(accp_.dmin);
            uin_v.push_back(+OSQP_INFTY);
            ++row;
        }
        }

    // Build A, l, u from triplets (no sparse blocks)
    const int meq = (int)beq.size();
    const int mineq = (int)lin_v.size();
    const int m = meq + mineq;

    std::vector<Triplet<double>> Atrips; Atrips.reserve(Aeqt.size() + Aint.size());
    Atrips.insert(Atrips.end(), Aeqt.begin(), Aeqt.end());
    for (const auto& t : Aint) Atrips.emplace_back(t.row() + meq, t.col(), t.value());

    SparseMatrix<double> A(m, NZ);
    A.setFromTriplets(Atrips.begin(), Atrips.end());

    VectorXd l(m), u(m);
    l.head(meq) = beq;
    u.head(meq) = beq;
    for (int i=0; i<mineq; ++i) { l(meq+i) = lin_v[i]; u(meq+i) = uin_v[i]; }


    // -----------------------------
    // Solve with OSQP
    // -----------------------------
    OsqpEigen::Solver solver;
    solver.settings()->setWarmStart(true);
    solver.settings()->setVerbosity(false);
    solver.settings()->setAlpha(1.6);
    solver.data()->setNumberOfVariables(NZ);
    solver.data()->setNumberOfConstraints(m);

    if (!solver.data()->setHessianMatrix(H)) return {};
    if (!solver.data()->setGradient(g))      return {};
    if (!solver.data()->setLinearConstraintsMatrix(A)) return {};
    if (!solver.data()->setLowerBound(l))     return {};
    if (!solver.data()->setUpperBound(u))     return {};
    if (!solver.initSolver())                 return {};

    if (solver.solveProblem() != OsqpEigen::ErrorExitFlag::NoError) return {};

    Eigen::VectorXd z = solver.getSolution();
    MPCControl out;
    out.R      = z(idx_u(0,0));
    out.ddelta = z(idx_u(0,1));
    out.ok     = true;
    return out;
}