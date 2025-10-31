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

void LTV_MPC::setVehicleParams(double m, double L, double d, double JG, double m0) {
    m_ = m; L_ = L; d_ = d; JG_ = JG; m0_ = m0;
    P.L = L_;                // keep heading kinematics consistent
    Lf_ = L_ - d_;   // front axle distance
    Lr_ = d_;        // rear axle distance
  }
  
void LTV_MPC::setLimits(double delta_max, double ddelta_max,
                        double R_min, double R_max,
                        double Ffl_max, double Frl_max) {
    delta_max_ = delta_max;
    ddelta_max_ = ddelta_max;
    Rmin_ = R_min; Rmax_ = R_max;
    Ffl_max_ = Ffl_max; Frl_max_ = Frl_max;
}

// ======================
// Tire slip model helpers
// ======================

// ---- slip angle helpers (put in a shared header or at top of both files) ----
static inline double clampAlpha(double a){
    const double a_max = 0.6; // ~34°
    return std::clamp(a, -a_max, a_max);
  }
  
  static inline double slipAngleFront(double vx, double vy, double dpsi, double delta, double Lf){
    const double vx_eff = std::max(0.1, vx);
    const double beta_f = std::atan2(vy + Lf * dpsi, vx_eff);
    return clampAlpha(beta_f - delta);
  }
  static inline double slipAngleRear(double vx, double vy, double dpsi, double Lr){
    const double vx_eff = std::max(0.1, vx);
    const double beta_r = std::atan2(vy - Lr * dpsi, vx_eff);
    return clampAlpha(beta_r);
  }


// Pure lateral Magic Formula (Pacejka 1996)
static inline double pacejkaFy(double B,double C,double D,double E,double alpha){
    // Fy = D * sin( C * atan( B*alpha - E*(B*alpha - atan(B*alpha)) ) )
    const double x = B * alpha;
    return D * std::sin( C * std::atan( x - E*(x - std::atan(x)) ) );
}

static inline void static_axle_loads(double m, double g, double L, double d,
    double& Fzf0, double& Fzr0){
    const double Lf = L - d;   // CoM -> front axle
    const double Lr = d;       // CoM -> rear axle
    Fzf0 = m * g * (Lr / L);   // front static load
    Fzr0 = m * g * (Lf / L);   // rear  static load
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
    const int N  = P.N;
    const int nx = P.acc_enable ? 7 : 6;   // [ey, epsi, vx, vy, dpsi, delta, (d)]
    const int nu = 2;                      // [R, ddelta]

    lm_.A.assign(N, Eigen::MatrixXd::Identity(nx, nx));
    lm_.B.assign(N, Eigen::MatrixXd::Zero(nx, nu));
    lm_.c.assign(N, Eigen::VectorXd::Zero(nx));

    // ---- continuous-time dynamics f(x,u; ref_k) for pure lateral slip ----
    auto f_dyn = [&](const Eigen::VectorXd& x,
                     const Eigen::VectorXd& u,
                     const MPCRef& ref_k) -> Eigen::VectorXd
    {
        const bool has_acc = P.acc_enable;

        // unpack state
        const double ey    = x(0);
        const double epsi  = x(1);
        double       vx    = x(2);
        const double vy    = x(3);
        const double dpsi  = x(4);
        const double delta = x(5);
        const double d_gap = has_acc ? x(6) : 0.0;   // (unused here)

        // avoid division by ~0 in slip angles
        vx = std::max(0.1, vx);

        // inputs
        const double R      = u(0);
        const double ddelta = u(1);

        // geometry
        const double Lf = L_ - d_;
        const double Lr = d_;

        // tire params
        const double Bf = tires_.Bf, Cf = tires_.Cf, Df = tires_.muf * 0.5 * m_ * 9.81 * (L_-d_) / L_; // rough Fz split
        const double Br = tires_.Br, Cr = tires_.Cr, Dr = tires_.mur * 0.5 * m_ * 9.81 * (d_)   / L_;

        // slip angles
        const double alpha_f = slipAngleFront(vx, vy, dpsi, delta, Lf);
        const double alpha_r = slipAngleRear (vx, vy, dpsi, Lr);

        // lateral forces (pure lateral MF)
        double Fy_f = - 2.0 * pacejkaFy(Bf, Cf, Df, tires_.Ef, alpha_f);
        double Fy_r = - 2.0 * pacejkaFy(Br, Cr, Dr, tires_.Er, alpha_r);

        // ---- normal loads per axle (static) ----
        const double g = 9.81;
        const double Fzf_ax = m_ * g * (L_ - d_) / L_;
        const double Fzr_ax = m_ * g * (d_)     / L_;

        // ---- split the longitudinal request to axles (example: RWD) ----
        double Fx_f_ax = 0.0;
        double Fx_r_ax = R;

        // ---- friction ellipse clamp (per axle) ----
        auto clamp_ellipse = [](double Fx_ax, double Fy_ax, double mu, double Fz_ax){
        const double Fmax = std::max(1e-6, mu * Fz_ax);
        const double n = std::sqrt(Fx_ax*Fx_ax + Fy_ax*Fy_ax);
        if (n > Fmax) {
            const double s = Fmax / n;
            Fx_ax *= s;
            Fy_ax *= s;
        }
        return std::pair<double,double>{Fx_ax, Fy_ax};
        };

        // use tire’s mu_f / mu_r, and the axle Fz you just computed
        auto [Fx_f_ax_eff, Fy_f_eff] = clamp_ellipse(Fx_f_ax, Fy_f, tires_.muf, Fzf_ax);
        auto [Fx_r_ax_eff, Fy_r_eff] = clamp_ellipse(Fx_r_ax, Fy_r, tires_.mur, Fzr_ax);

        // total longitudinal after ellipse
        const double Fx_total_eff = Fx_f_ax_eff + Fx_r_ax_eff;
          

        const double ax = (Fx_total_eff - Fy_f_eff * std::sin(delta)) / m_ + vy * dpsi;
        const double ay = (Fy_f_eff * std::cos(delta) + Fy_r_eff)      / m_ - vx * dpsi;

        Eigen::VectorXd dx = Eigen::VectorXd::Zero(has_acc ? 7 : 6);
        dx(0) = vx * std::sin(epsi) + vy * std::cos(epsi);                             // e_y_dot
        dx(1) = dpsi - ref_k.hp[0].kappa * vx;                                         // e_psi_dot
        dx(2) = ax;                                                                     // v_x_dot
        dx(3) = ay;                                                                     // v_y_dot
        dx(4) = (Lf * Fy_f_eff * std::cos(delta) - Lr * Fy_r_eff) / JG_;               // yaw-rate dot
        dx(5) = ddelta;                                                                 // delta dot
        if (has_acc) {
            const double v_obj = ref_k.hp[0].v_ref;  // (your choice; just be consistent)
            dx(6) = v_obj - vx;                      // gap dynamics
        }

        // silence unuseds (informative to keep but not used numerically)
        (void)ey; (void)d_gap;
        return dx;
    };

    // ---- per-step linearization ----
    for (int k = 0; k < N; ++k) {
        const int idx = std::min<int>(k, (int)ref.hp.size() - 1);

        // nominal about which we linearize
        Eigen::VectorXd x0(nx); x0.setZero();
        Eigen::VectorXd u0(nu); u0.setZero();

        // choose a simple nominal consistent with preview
        x0(2) = std::max(0.0, ref.hp[idx].v_ref); // vx
        x0(3) = 0.0;                               // vy
        x0(4) = 0.0;                               // dpsi
        x0(5) = 0.0;                               // delta
        if (P.acc_enable) x0(6) = 0.0;            // gap

        // make a 1-step ref view for f_dyn
        MPCRef ref_k;
        ref_k.hp.resize(1);
        ref_k.hp[0] = ref.hp[idx];

        // base vector field and numerical Jacobians
        const Eigen::VectorXd f0 = f_dyn(x0, u0, ref_k);

        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(nx, nx);
        Eigen::MatrixXd B = Eigen::MatrixXd::Zero(nx, nu);
        Eigen::VectorXd c = Eigen::VectorXd::Zero(nx);

        const double eps = 1e-6;

        // df/dx
        for (int i = 0; i < nx; ++i) {
            Eigen::VectorXd x_eps = x0; x_eps(i) += eps;
            Eigen::VectorXd f_eps = f_dyn(x_eps, u0, ref_k);
            A.col(i) = (f_eps - f0) / eps;
        }

        // df/du
        for (int j = 0; j < nu; ++j) {
            Eigen::VectorXd u_eps = u0; u_eps(j) += eps;
            Eigen::VectorXd f_eps = f_dyn(x0, u_eps, ref_k);
            B.col(j) = (f_eps - f0) / eps;
        }

        // affine term: f(x) ≈ A(x-x0) + B(u-u0) + c  ⇒ c = f0 - A x0 - B u0
        c = f0 - A * x0 - B * u0;

        // forward Euler discretization
        Eigen::MatrixXd Ad = Eigen::MatrixXd::Identity(nx, nx) + P.dt * A;
        Eigen::MatrixXd Bd = P.dt * B;
        Eigen::VectorXd cd = P.dt * c;

        // // explicit ACC row in discrete form (matches dx(6)=v_obj - vx)
        // if (P.acc_enable) {
        //     const int id_vx = 2;
        //     const int id_d  = 6;
        //     const double vobj = (ref.v_obj.size() > (size_t)k) ? ref.v_obj[k] : ref.hp[idx].v_ref;
        //     Ad(id_d, id_d) += 1.0;      // d_{k+1} depends on d_k
        //     Ad(id_d, id_vx) += -P.dt;   // -vx_k
        //     cd(id_d)        +=  P.dt * vobj;
        // }

        lm_.A[k] = Ad;
        lm_.B[k] = Bd;
        lm_.c[k] = cd;
    }

    // nominal containers (optional)
    nom_.x.resize(N+1);
    nom_.u.resize(N);
    for (int k = 0; k <= N; ++k) {
        nom_.x[k].ey   = 0.0;
        nom_.x[k].epsi = 0.0;
        nom_.x[k].vx   = (k < (int)ref.hp.size()) ? ref.hp[k].v_ref : 0.0;
        nom_.x[k].vy   = 0.0;
        nom_.x[k].dpsi = 0.0;
        nom_.x[k].delta= 0.0;
        if (P.acc_enable) nom_.x[k].d = 0.0;
    }
    for (int k = 0; k < N; ++k) nom_.u[k].setZero();
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
    const int nx = P.acc_enable ? 7 : 6;   // [ey, epsi, vx, vy, dpsi, delta, (d)]
    const int nu = 2;                      // [R, ddelta]
    const int N  = P.N;

    // canonical indices
    const int id_ey    = 0;
    const int id_epsi  = 1;
    const int id_vx    = 2;
    const int id_vy    = 3;
    const int id_r     = 4;
    const int id_delta = 5;
    const int id_d     = P.acc_enable ? 6 : -1;

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
    beq(row_x0(id_ey))   = x0.ey;    Aeqt.emplace_back(row_x0(id_ey),   idx_x(0,id_ey),   1.0);
    beq(row_x0(id_epsi)) = x0.epsi;  Aeqt.emplace_back(row_x0(id_epsi), idx_x(0,id_epsi), 1.0);
    beq(row_x0(id_vx))   = x0.vx;    Aeqt.emplace_back(row_x0(id_vx),   idx_x(0,id_vx),   1.0);
    beq(row_x0(id_vy))   = x0.vy;    Aeqt.emplace_back(row_x0(id_vy),   idx_x(0,id_vy),   1.0);
    beq(row_x0(id_r))    = x0.dpsi;      Aeqt.emplace_back(row_x0(id_r),    idx_x(0,id_r),    1.0);
    beq(row_x0(id_delta))= x0.delta; Aeqt.emplace_back(row_x0(id_delta),idx_x(0,id_delta),1.0);
    if (P.acc_enable) {
        beq(row_x0(id_d)) = x0.d;
        Aeqt.emplace_back(row_x0(id_d), idx_x(0,id_d), 1.0);
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
        push_row_le(row, idx_u(k,0), 1.0, Rmin_, Rmax_);                 ++row; // R
        push_row_le(row, idx_u(k,1), 1.0, -ddelta_max_, ddelta_max_);    ++row; // δ̇
    }
    
    // steering angle bounds
    for (int k=0; k<=N; ++k){
        push_row_le(row, idx_x(k,id_delta), 1.0, -delta_max_, delta_max_);      ++row; // δ
    }

    // v bounds (optional)
    for (int k=0; k<=N; ++k){
        push_row_le(row, idx_x(k,id_vx), 1.0, P.v_min, P.v_max); ++row;
    }
    

    const double g_acc = 9.81;
    double Fzf0, Fzr0;
    static_axle_loads(m_, g_acc, L_, d_, Fzf0, Fzr0);

    // Per-axle *peak* pure-lateral capacity (sum over two wheels on an axle).
    // D_f = mu_f * Fz_f0 (per wheel) → front axle peak ≈ 2 * D_f
    // Ditto for the rear. If you have wheel-wise Fz, change the 2× factor.
    const double Fy_front_peak = 2.0 * (tires_.muf * Fzf0);
    const double Fy_rear_peak  = 2.0 * (tires_.mur * Fzr0);

    for (int k = 0; k < N; ++k){
        const int idx = std::min<int>(k, (int)ref.hp.size()-1);
        const double vref = (idx >= 0) ? ref.hp[idx].v_ref : 0.0;

        // Required total lateral force (bicycle) ~ m * v^2 * kappa, with kappa ≈ delta/L
        // We linearize w.r.t. delta and plug v_ref^2 to keep it affine.
        const double coeff = m_ * (vref * vref) / L_;
        const double Fy_cap = Fy_front_peak + Fy_rear_peak;

        //  coeff * (+delta_k) <= Fy_cap
        Aint.emplace_back(row, idx_x(k, id_delta), +coeff);
        lin_v.push_back(-OSQP_INFTY);
        uin_v.push_back(Fy_cap);
        ++row;

        //  coeff * (-delta_k) <= Fy_cap   ⇔   -coeff*delta_k <= Fy_cap
        Aint.emplace_back(row, idx_x(k, id_delta), -coeff);
        lin_v.push_back(-OSQP_INFTY);
        uin_v.push_back(Fy_cap);
        ++row;
    }

    // Gap lower bound d >= 0 (optional but nice to keep feasibility)
    if (P.acc_enable){
        for (int k=0; k<=N; ++k){
            push_row_le(row, idx_x(k,id_d), 1.0, 0.0, +OSQP_INFTY); ++row;  // d >= 0
        }
        for (int k=0; k<N; ++k){
            const bool has = (ref.has_obj.size() > (size_t)k) ? (ref.has_obj[k]!=0) : false;
            if (!has) continue;
            // d_k - tau * vx_k >= d_min  <=>  (+1)*d_k + (-tau)*vx_k >= d_min
            Aint.emplace_back(row, idx_x(k,id_d),  +1.0);
            Aint.emplace_back(row, idx_x(k,id_vx), -P.acc_tau);
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
    out.R      = z(idx_u(0,0));
    out.ddelta = z(idx_u(0,1));
    out.ok     = true;
    return out;
}
