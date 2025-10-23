#include "mpc_ltv.hpp"
#include "obstacles.hpp"

#include <OsqpEigen/OsqpEigen.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

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

// ------------------------------------------------------------------
// buildLinearization: simple kinematic bicycle linearization
// States: [ey, epsi, v, delta], Inputs: [a, ddelta]
//  ey_dot   ≈ v * epsi
//  epsi_dot ≈ (v/L) * delta - v * kappa
//  v_dot    = a
//  delta_dot= ddelta
// Discretization: forward Euler
// ------------------------------------------------------------------
void LTV_MPC::buildLinearization(const MPCRef& ref) {
    // const int N = P.N;
    // lm_.A.assign(N, Eigen::Matrix<double,4,4>::Identity());
    // lm_.B.assign(N, Eigen::Matrix<double,4,2>::Zero());
    // lm_.c.assign(N, Eigen::Matrix<double,4,1>::Zero());

    const int N  = P.N;
    const int nx = P.acc_enable ? 5 : 4;
    const int nu = 2;
    lm_.A.assign(N, Eigen::MatrixXd::Identity(nx, nx));
    lm_.B.assign(N, Eigen::MatrixXd::Zero(nx, nu));
    lm_.c.assign(N, Eigen::VectorXd::Zero(nx));

    for (int k = 0; k < N; ++k) {
        const int idx = std::min<int>(k, (int)ref.hp.size()-1);
        const double vref = (idx >= 0) ? ref.hp[idx].v_ref : 0.0;
        const double kap  = (idx >= 0) ? ref.hp[idx].kappa : 0.0;

        // Eigen::Matrix<double,4,4> A = Eigen::Matrix<double,4,4>::Zero();
        // Eigen::Matrix<double,4,2> B = Eigen::Matrix<double,4,2>::Zero();
        // Eigen::Matrix<double,4,1> d = Eigen::Matrix<double,4,1>::Zero();

        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(nx, nx);
        Eigen::MatrixXd B = Eigen::MatrixXd::Zero(nx, nu);
        Eigen::VectorXd d = Eigen::VectorXd::Zero(nx);

        A(0,1) = vref;            // ey_dot ≈ v*epsi
        A(1,3) = vref / P.L;      // ∂epsi_dot/∂delta
        A(1,2) = -kap;            // ∂epsi_dot/∂v (linearization at delta≈0)

        B(2,0) = 1.0;             // v_dot = a
        B(3,1) = 1.0;             // delta_dot = ddelta

        d(1)   = -vref * kap;     // affine term from -v*kappa

        // ---- ACC gap dynamics (Eq. (2) in the paper) ----
        if (P.acc_enable) {
                const int id_d = 4;              // state index for gap
                A(id_d,2) = -1.0;                // \dot d = v_obj - v
                // B(row,id) = 0
                const double vobj = (ref.v_obj.size() > (size_t)k) ? ref.v_obj[k] : vref;
                d(id_d) = vobj;                  // affine part
            }

        // if (P.acc_enable) {
        //     const int id_d = 4;
        
        //     double epsi_nom = (ref.epsi_nom.size() > (size_t)k) ? ref.epsi_nom[k] : 0.0;
        //     const double cos_epsi = std::cos(epsi_nom);
        //     const double sin_epsi = std::sin(epsi_nom);

        //     // gap dynamics
        //     A(id_d,2) = -cos_epsi;
        //     A(id_d,1) = vref * sin_epsi;
        //     const double vobj = (ref.v_obj.size() > (size_t)k) ? ref.v_obj[k] : vref;
        //     d(id_d)   = vobj;

        //     if (k == 0) std::cout << "k=" << k << " epsi_nom=" << epsi_nom << " A2=" << -cos_epsi << " A1=" << vref*sin_epsi << " d=" << vobj - vref * cos_epsi << std::endl;
        // }

        // if (P.acc_enable) {
        //     const int id_d = 4;
        //     const double epsi_nom = (ref.epsi_nom.size() > (size_t)k) ? ref.epsi_nom[k] : 0.0;
        //     const double cos_e = std::cos(epsi_nom);
        //     const double sin_e = std::sin(epsi_nom);
        //     const double vref  = (idx >= 0) ? ref.hp[idx].v_ref : 0.0;
        //     const double vobj  = (ref.v_obj.size() > (size_t)k) ? ref.v_obj[k] : vref;
          
        //     A(id_d,2) = -cos_e;               // ∂/∂v at nominal
        //     A(id_d,1) =  vref * sin_e;        // ∂/∂epsi at nominal (absolute states)
        //     d(id_d)   =  vobj - vref * sin_e * epsi_nom;  // affine f(x_nom) - A x_nom
        //   }
        

        // Discretize
        // Eigen::Matrix<double,4,4> Ad = Eigen::Matrix<double,4,4>::Identity() + P.dt * A;
        // Eigen::Matrix<double,4,2> Bd = P.dt * B;
        // Eigen::Matrix<double,4,1> cd = P.dt * d;

        Eigen::MatrixXd Ad = Eigen::MatrixXd::Identity(nx, nx) + P.dt * A;
        Eigen::MatrixXd Bd = P.dt * B;
        Eigen::VectorXd cd = P.dt * d;

        // // robust discretization (Tustin / bilinear)
        // Eigen::MatrixXd I  = Eigen::MatrixXd::Identity(nx, nx);
        // Eigen::MatrixXd M1 = I - 0.5 * P.dt * A;
        // Eigen::MatrixXd M2 = I + 0.5 * P.dt * A;

        // Eigen::PartialPivLU<Eigen::MatrixXd> lu(M1);
        // Eigen::MatrixXd Ad = lu.solve(M2);
        // Eigen::MatrixXd Bd = lu.solve(P.dt * B);
        // Eigen::VectorXd cd = lu.solve(P.dt * d);

        lm_.A[k] = Ad;
        lm_.B[k] = Bd;
        lm_.c[k] = cd;
    }

    // Basic nominal trajectory (zeros except v nominal to last preview)
    nom_.x.resize(N+1);
    nom_.u.resize(N);
    for (int k=0; k<=N; ++k) {
        nom_.x[k].ey = 0.0;
        nom_.x[k].epsi = 0.0;
        nom_.x[k].v = (k < (int)ref.hp.size()) ? ref.hp[k].v_ref
                                               : ref.hp.empty() ? 0.0 : ref.hp.back().v_ref;
        nom_.x[k].delta = 0.0;
    }
    for (int k=0; k<N; ++k) nom_.u[k].setZero();
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
    const int nx = P.acc_enable ? 5 : 4;   // [ey, epsi, v, delta, (d)]
    const int nu = 2;   // [a, ddelta]
    const int N  = P.N;
    const int id_ey = 0, id_epsi = 1, id_v = 2, id_delta = 3;
    const int id_d  = P.acc_enable ? 4 : -1;

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
            // ey_lower[k] = -P.ey_max;
            // ey_upper[k] = +P.ey_max;
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
        Ht.emplace_back(idx_u(k,0), idx_u(k,0), 2.0 * P.wa);
        Ht.emplace_back(idx_u(k,1), idx_u(k,1), 2.0 * P.wdd);

        // input slew (u_k - u_{k-1})^2
        if (k>0){
            for (int j=0;j<nu;++j){
                const int uk   = idx_u(k,j);
                const int ukm1 = idx_u(k-1,j);
                const double w = (j==0? P.wda : P.wddd);
                Ht.emplace_back(uk,    uk,    2.0*w);
                Ht.emplace_back(ukm1,  ukm1,  2.0*w);
                Ht.emplace_back(uk,    ukm1, -2.0*w);
                Ht.emplace_back(ukm1,  uk,   -2.0*w);
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

    if (P.acc_enable) {
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
        // a_min <= a_k <= a_max
        push_row_le(row, idx_u(k,0), 1.0, P.a_min, P.a_max); ++row;
        // -ddelta_max <= ddelta_k <= ddelta_max
        push_row_le(row, idx_u(k,1), 1.0, -P.ddelta_max, P.ddelta_max); ++row;
    }

    // v bounds (optional)
    for (int k=0; k<=N; ++k){
        push_row_le(row, idx_x(k,2), 1.0, P.v_min, P.v_max); ++row;
    }

    // delta bounds
    for (int k=0; k<=N; ++k){
        push_row_le(row, idx_x(k,3), 1.0, -P.delta_max, P.delta_max); ++row;
    }

    // Gap lower bound d >= 0 (optional but nice to keep feasibility)
    if (P.acc_enable){
        for (int k=0; k<=N; ++k){
            push_row_le(row, idx_x(k,id_d), 1.0, 0.0, +OSQP_INFTY); ++row;
        }
        // ACC safety: d_k - tau * v_k >= d_min  (only if there's a lead at step k)
        for (int k=0; k<N; ++k){
            const bool has = (ref.has_obj.size() > (size_t)k) ? (ref.has_obj[k]!=0) : false;
            if (!has) continue;
            // Aineq[row, id_d] = +1, Aineq[row, id_v] = -tau
            Aint.emplace_back(row, idx_x(k,id_d),  +1.0);
            Aint.emplace_back(row, idx_x(k,id_v),  -P.acc_tau);
            lin_v.push_back(P.acc_dmin);
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
    out.a      = z(idx_u(0,0));
    out.ddelta = z(idx_u(0,1));
    out.ok     = true;
    return out;
}
