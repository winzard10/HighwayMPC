#include "centerline_map.hpp"
#include "mpc_ltv.hpp"
#include "obstacles.hpp"
#include "corridor_planner.hpp"
#include "vehicle_model.hpp"
#include "tire_model.hpp"
#include "acc.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

using VParams  = dynamics::vehicle::Params;
using VLimits  = dynamics::vehicle::Limits;
using VState   = dynamics::vehicle::State;
using VControl = dynamics::vehicle::Control;
using TTire    = dynamics::tire::TireParams;

namespace fs = std::filesystem;

// ---------- CLI options ----------
struct CLI {
  CenterlineMap::LaneRef lane_from{CenterlineMap::LaneRef::Right};
  CenterlineMap::LaneRef lane_to  {CenterlineMap::LaneRef::Right};
  double t_change{1e18};     // default: no lane change
  double T_change{3.0};
  std::string map_file{"data/lane_centerlines.csv"};
  std::string obs_file_primary{"data/obstacles.csv"};
  std::string obs_file_fallback{"data/obstacles_example.csv"};
  std::string log_file{"sim_log.csv"};
  double T{180.0};
};

// ---------- Small math & utils ----------
static inline double clamp(double u, double lo, double hi) {
  return std::min(std::max(u, lo), hi);
}

static inline double smoothstep01(double u) {
  // 3u^2 - 2u^3, clamped to [0,1]
  if (u <= 0.0) return 0.0;
  if (u >= 1.0) return 1.0;
  return u * u * (3.0 - 2.0 * u);
}

static inline double wrapAngle(double a) {
  while (a >  M_PI) a -= 2.0 * M_PI;
  while (a <=-M_PI) a += 2.0 * M_PI;
  return a;
}

static inline CenterlineMap::LaneRef parseLane(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), ::tolower);
  if (s == "right") return CenterlineMap::LaneRef::Right;
  if (s == "left")  return CenterlineMap::LaneRef::Left;
  return CenterlineMap::LaneRef::Center;
}

// ---------- Vehicle model (dynamic bicycle + pure lateral slip) ----------
static VState stepVehicle(const VState& s, const VControl& u,
  const VParams& vp, const VLimits& lim,
  const TTire& tp) {
return dynamics::vehicle::step(s, u, vp, lim, tp);
}


// ---------- Lane pose helpers ----------
struct LanePose {
  double x{0}, y{0}, psi{0}, kappa{0}, v_ref{0}, lane_width{3.7};
};

static inline LanePose lanePoseAt(const CenterlineMap& map, double s,
                                  CenterlineMap::LaneRef which) {
  if (which == CenterlineMap::LaneRef::Right) {
    auto r = map.right_lane_at(s); return {r.x, r.y, r.psi, r.kappa, r.v_ref, r.lane_width};
  } else if (which == CenterlineMap::LaneRef::Left) {
    auto l = map.left_lane_at(s);  return {l.x, l.y, l.psi, l.kappa, l.v_ref, l.lane_width};
  }
  auto c = map.center_at(s);       return {c.x, c.y, c.psi, c.kappa, c.v_ref, c.lane_width};
}

// Compute lateral offset (ey) of a point (x,y) from the centerline
double lateralOffsetFromCenterline(const CenterlineMap& map, double x, double y, CenterlineMap::LaneRef which) {
  // Project obstacle position onto centerline
  auto proj = map.project(x, y);    // expects .s_proj, .x_ref, .y_ref, .psi, etc.

  LanePose cref = lanePoseAt(map, proj.s_proj, which);

  // Normal vector to centerline heading (left-positive)
  const double nx = -std::sin(cref.psi);
  const double ny =  std::cos(cref.psi);

  // Lateral offset (dot product with normal)
  const double ey = (x - cref.x) * nx + (y - cref.y) * ny;
  return ey;
}

// ---------- CLI parsing ----------
static void parseCLI(int argc, char** argv, CLI& cli) {
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--lane-from"   && i + 1 < argc) cli.lane_from = parseLane(argv[++i]);
    else if (a == "--lane-to"   && i + 1 < argc) cli.lane_to   = parseLane(argv[++i]);
    else if (a == "--t-change"  && i + 1 < argc) cli.t_change  = std::stod(argv[++i]);
    else if (a == "--T-change"  && i + 1 < argc) cli.T_change  = std::stod(argv[++i]);
    else if (a == "--map"       && i + 1 < argc) cli.map_file  = argv[++i];
    else if (a == "--log"       && i + 1 < argc) cli.log_file  = argv[++i];
    else if (a == "--T"         && i + 1 < argc) cli.T         = std::stod(argv[++i]);
    else if (a == "--help" || a == "-h") {
      std::cout <<
        "Usage: sim_main [options]\n"
        "  --lane-from {left|center|right}\n"
        "  --lane-to   {left|center|right}\n"
        "  --t-change  <sec>     (default: no change)\n"
        "  --T-change  <sec>     (default: 3.0)\n"
        "  --map       <file>    (default: data/lane_centerlines.csv)\n"
        "  --log       <file>    (default: sim_log.csv)\n"
        "  --T         <sec>     (total sim time, default: 180)\n";
      std::exit(0);
    }
  }
}

// For logging: recompute axle lateral forces at the *current* state and inputs
static inline std::pair<double,double>
compute_tire_forces(const VState& s, double delta, double R,
                    const VParams& vp, const TTire& tp)
{
  dynamics::tire::VehicleGeom vg{vp.m, vp.L, vp.d, vp.JG};
  const auto fr = dynamics::tire::computeForcesBody(
      s.vx, s.vy, s.dpsi, delta, R, vg, tp /* g=9.81 default */);
  return {fr.Fy_f_body, fr.Fy_r_body};
}


// ---------- Main ----------
int main(int argc, char** argv) {
  // --- CLI ---
  CLI cli;
  parseCLI(argc, argv, cli);

  // --- Map ---
  CenterlineMap map;
  if (!map.load_csv(cli.map_file)) {
    std::cerr << "Failed to load centerlines: " << cli.map_file << "\n";
    return 1;
  }
  std::cout << "Loaded map: " << cli.map_file
            << "  (s ∈ [" << map.s_min() << ", " << map.s_max() << "])\n";

  // --- Obstacles (optional) ---
  Obstacles obstacles;
  std::string obs_file = fs::exists(cli.obs_file_primary)
                         ? cli.obs_file_primary
                         : (fs::exists(cli.obs_file_fallback) ? cli.obs_file_fallback : "");
  if (!obs_file.empty()) {
    if (obstacles.load_csv(obs_file)) {
      std::cout << "Loaded obstacles: " << obs_file
                << " (" << obstacles.items.size() << " items)\n";
    } else {
      std::cout << "Failed to parse obstacles file: " << obs_file << "\n";
    }
  } else {
    std::cout << "No obstacles file found; running without obstacles.\n";
  }

  // --- Vehicle & MPC ---
  VParams  vp;        // has m, L, d, JG, dt, track_w, etc.
  VLimits  lim;       // delta_max, ddelta_max, R_min/max, etc.
  VState   st;
  MPCParams mpcp; mpcp.dt = vp.dt;
  acc::reset_defaults();
  acc::Params accp = acc::Params{};
  acc::reset_gap(accp);          // sets internal gap to accp.d_init

  LTV_MPC mpc(mpcp);
  mpc.setVehicleParams(vp);
  mpc.setLimits(lim);
  mpc.setACCParams(accp);        // controller now reads accp.enable internally

  // NOTE: No tire params set here.
  //       MPC and plant will read from dynamics::tire::current().

  // Initialize vehicle pose on chosen starting lane at s_min
  st.s = map.s_min();
  {
    LanePose p0 = lanePoseAt(map, st.s, cli.lane_from);
    st.x = p0.x; st.y = p0.y; st.psi = p0.psi;
    st.vx = 25.0;
    st.vy = 0.0;
    st.dpsi = 0.0;
    st.delta = 0.0;
  }

  double ax_prev = 0.0;  // for logging jerk

  // --- Logging ---
  std::ofstream log(cli.log_file);
  if (!log) {
    std::cerr << "Cannot open log file: " << cli.log_file << "\n";
    return 1;
  }
  log << std::fixed << std::setprecision(6);
  log << "t,s,x,y,psi,vx,vy,dpsi,delta,ax,ay,ddpsi,jerk,R_cmd,ddelta_cmd,ey,epsi,dv,"
         "v_ref,x_ref,y_ref,psi_ref,alpha,dmin,v_lead,d_gap,Fy_f,Fy_r\n";

  // --- Simulation loop ---
  const int steps = static_cast<int>(cli.T / vp.dt);
  for (int k = 0; k <= steps; ++k) {
    const double t = k * vp.dt;

    // Project current pose to centerline
    auto projC = map.project(st.x, st.y);
    auto cref  = map.center_at(projC.s_proj);

    // Lane-change blend for target XY
    const double alpha = smoothstep01((t - cli.t_change) / cli.T_change);
    LanePose fromP = lanePoseAt(map, projC.s_proj, cli.lane_from);
    LanePose toP   = lanePoseAt(map, projC.s_proj, cli.lane_to);

    auto blendYaw = [](double psi1, double psi2, double a) {
      const double d = std::atan2(std::sin(psi2 - psi1), std::cos(psi2 - psi1));
      return psi1 + a * d;
    };

    const double psi_ref = blendYaw(fromP.psi, toP.psi, alpha);
    const double x_ref   = (1.0 - alpha) * fromP.x + alpha * toP.x;
    const double y_ref   = (1.0 - alpha) * fromP.y + alpha * toP.y;

    // Frenet errors
    const double nx = -std::sin(psi_ref);
    const double ny =  std::cos(psi_ref);
    const double ey   = (st.x - x_ref) * nx + (st.y - y_ref) * ny;
    const double epsi = wrapAngle(st.psi - psi_ref);
    // const double dv   = st.vx - cref.v_ref;

    // --- MPC preview (reference horizon & corridor frames) ---
    MPCRef pref; pref.hp.resize(mpcp.N);
    std::vector<Eigen::Vector2d>              p_ref_h(mpcp.N);
    std::vector<double>                       psi_ref_h(mpcp.N);
    std::vector<double>                       s_grid(mpcp.N);
    std::vector<CenterlineMap::LaneRef>       lane_ref_h(mpcp.N);
    std::vector<double>                       lane_w_h(mpcp.N);

    double s_h = projC.s_proj;
    double t_h = t;

    for (int i = 0; i < mpcp.N; ++i) {
      s_grid[i] = s_h;

      CenterlineMap::LaneRef which;
      if (t_h < cli.t_change) which = CenterlineMap::LaneRef::Right;
      else if (t_h > cli.t_change + cli.T_change) which = CenterlineMap::LaneRef::Left;
      else which = CenterlineMap::LaneRef::Center;

      LanePose c = lanePoseAt(map, s_h, which);

      // MPC preview
      pref.hp[i].kappa = c.kappa;
      pref.hp[i].v_ref = c.v_ref;

      // corridor metadata
      p_ref_h[i]   = {c.x, c.y};
      psi_ref_h[i] = c.psi;
      lane_ref_h[i]= which;
      lane_w_h[i]  = std::max(0.1, c.lane_width);

      // propagate obstacles' ey in this lane frame (if any active)
      for (const auto &a : obstacles.active_at(t)) {
        double ey_o = lateralOffsetFromCenterline(map, a.x, a.y, which);
        obstacles.items[a.idx].ey_obs = ey_o;
      }

      // advance along centerline with v_ref
      const double v_step = std::max(1e-3, c.v_ref);
      s_h = std::min(s_h + v_step * mpcp.dt, map.s_max());
      t_h += mpcp.dt;
    }

    std::vector<double> lo(mpcp.N), up(mpcp.N);
    for (int i = 0; i < mpcp.N; ++i) {
      const double w = lane_w_h[i];
      switch (lane_ref_h[i]) {
        case CenterlineMap::LaneRef::Right: lo[i] = -0.5 * w; up[i] = +1.5 * w; break;
        case CenterlineMap::LaneRef::Left:  lo[i] = -1.5 * w; up[i] = +0.5 * w; break;
        default:                            lo[i] = -w;       up[i] = +w;       break;
      }
    }

    // Current state in Frenet error coordinates
    MPCState xk;
    xk.ey    = ey;
    xk.epsi  = epsi;
    xk.vx    = st.vx;
    xk.vy    = st.vy;
    xk.dpsi  = st.dpsi;
    xk.delta = st.delta;
    if (accp.enable) xk.d = acc::gap();

    // --- Corridor planning ---
    const double t0     = t;
    const double margin = 0.1;
    const double L_look = 70.0;
    const double slope  = 0.25;

    corridor::buildRawBounds(
        p_ref_h, psi_ref_h, lane_w_h, lane_ref_h,
        t0, mpcp.dt, obstacles, vp.track_w, margin, L_look, lo, up);

    corridor::Output cor = corridor::planGraph(s_grid, lo, up, slope);

    pref.ey_ref.resize(mpcp.N + 1);
    for (int i = 0; i < mpcp.N; ++i) pref.ey_ref[i] = cor.ey_ref[i];
    pref.ey_ref.back() = pref.ey_ref[mpcp.N - 1];
    pref.ey_ref_N      = pref.ey_ref.back();

    mpc.setCorridorBounds(cor.lo, cor.up);

    // --- Synthetic lead profile for ACC ---
    if (accp.enable) {
      pref.v_obj.resize(mpcp.N);
      pref.has_obj.resize(mpcp.N);
      acc::fill_preview(pref, t, mpcp.dt, mpcp.N);
      xk.d = acc::gap();
    } else {
      pref.v_obj.clear();
      pref.has_obj.clear();
    }

    // --- Solve MPC ---
    MPCControl u_mpc = mpc.solve(xk, pref);
    if (!u_mpc.ok) { u_mpc.R = 0.0; u_mpc.ddelta = 0.0; }

    // --- Measure min distance to active obstacles (for logging) ---
    double dmin = std::numeric_limits<double>::infinity();
    for (const auto& a : obstacles.active_at(t)) {
      const double dx = st.x - a.x;
      const double dy = st.y - a.y;
      const double dd = std::sqrt(dx*dx + dy*dy) - a.radius;
      if (dd < dmin) dmin = dd;
    }
    if (!std::isfinite(dmin)) dmin = std::numeric_limits<double>::quiet_NaN();

    // --- Advance vehicle dynamics ---
    const double R_cmd      = clamp(u_mpc.R,      lim.R_min, lim.R_max);
    const double ddelta_cmd = clamp(u_mpc.ddelta, -lim.ddelta_max, lim.ddelta_max);
    const VControl u_cmd{R_cmd, ddelta_cmd};

    // advance plant with tire-slip dynamics using the same centralized params
    st = stepVehicle(st, u_cmd, vp, lim, dynamics::tire::current());

    // Stop if we reach end of map
    if (projC.s_proj > map.s_max() - 1.0) break;
    
    // --- Update ACC state ---
    double v_lead_now = std::numeric_limits<double>::quiet_NaN();
    double d_gap      = std::numeric_limits<double>::quiet_NaN();

    if (accp.enable) {
      v_lead_now = acc::lead_speed(t);
      acc::update_gap(v_lead_now, st.vx, mpcp.dt);
      d_gap = acc::gap();
    }

    printf("vx: %.2f m/s, ey: %.2f m, epsi: %.2f rad, R_cmd: %.1f N, ddelta_cmd: %.3f rad/s, d_gap: %.2f m\n",
          st.vx, ey, epsi, R_cmd, ddelta_cmd, d_gap);
    std::cout << "--------------------------------------------\n";

    // --- Log ---
    // If you want the actual axle Fy for log, recompute once with current params:
    dynamics::tire::VehicleGeom vg{vp.m, vp.L, vp.d, vp.JG};
    auto fr_now = dynamics::tire::computeForcesBody(
        st.vx, st.vy, st.dpsi, st.delta, u_cmd.R, vg, dynamics::tire::current());

    log << t << "," << projC.s_proj << "," << st.x << "," << st.y << ","
        << st.psi << "," << st.vx << "," << st.vy << "," << st.dpsi << "," << st.delta << "," 
        << st.ax << "," << st.ay << "," << st.ddpsi << "," << (st.ax-ax_prev)/mpcp.dt << ","
        << u_cmd.R << "," << u_cmd.ddelta << ","
        << ey << "," << epsi << "," << (st.vx - cref.v_ref) << ","
        << cref.v_ref << "," << x_ref << "," << y_ref << "," << psi_ref << ","
        << alpha << "," << dmin << "," << v_lead_now << "," << d_gap << ","
        << fr_now.Fy_f_body << "," << fr_now.Fy_r_body << "\n";
    
    ax_prev = st.ax;
  }

  std::cout << "Log written to: " << cli.log_file << "\n";
  return 0;
}

