/// ------------------------------- /// 
/// Obstacle module (obstacles.cpp) ///
/// ------------------------------- ///

#include "obstacles.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <limits>

#ifndef OSQP_INFTY
#define OSQP_INFTY 1e20
#endif

// Convert string to lowercase (ASCII-only, safe for CSV tokens)
static inline std::string lower(std::string s){
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return char(std::tolower(c)); });
    return s;
}

// -----------------------------------------------------------------------------
// Obstacles::load_csv
//
// Load a list of obstacles from a CSV file with header:
//
//   id,kind,x0,y0,vx,vy,radius,t_start,t_end
//
// where:
//   - kind ∈ {"static","moving"} (case-insensitive, anything else -> Static)
//   - x0,y0 are initial position in world frame
//   - vx,vy are velocities for moving obstacles
//   - radius is the safety radius (already inflated)
//   - t_start,t_end define active time window
//
// Malformed rows are silently skipped. On success, fills `items` and returns true.
// -----------------------------------------------------------------------------
bool Obstacles::load_csv(const std::string& path) {
    items.clear();
    std::ifstream fin(path);
    if(!fin) return false;

    std::string line;
    // try to read header if present (ignored content)
    if (!std::getline(fin, line)) return false;

    while(std::getline(fin, line)){
        if(line.empty()) continue;
        std::stringstream ss(line);
        std::string cell; std::vector<std::string> cols;
        // split by comma
        while(std::getline(ss, cell, ',')) cols.push_back(cell);
        if(cols.size() < 9) continue;  // require all fields

        Obstacle ob;
        try {
            ob.id = std::stoi(cols[0]);
            const auto kind = lower(cols[1]);
            // classify as Moving only if explicitly "moving", otherwise Static
            ob.kind = (kind=="moving") ? Obstacle::Kind::Moving : Obstacle::Kind::Static;
            ob.x0 = std::stod(cols[2]); ob.y0 = std::stod(cols[3]);
            ob.vx = std::stod(cols[4]); ob.vy = std::stod(cols[5]);
            ob.radius  = std::stod(cols[6]);
            ob.t_start = std::stod(cols[7]);
            ob.t_end   = std::stod(cols[8]);
        } catch (...) {
            // Skip row if any parse fails
            continue;
        }
        items.push_back(ob);
    }
    return true;
}

// -----------------------------------------------------------------------------
// Obstacles::position_of
//
// Evaluate obstacle position at time t.
//
// For Moving:
//   x(t) = x0 + vx * (t - t_start)
//   y(t) = y0 + vy * (t - t_start)
//
// For Static:
//   x(t) = x0
//   y(t) = y0
//
// Returns std::nullopt if t is outside [t_start, t_end].
// -----------------------------------------------------------------------------
std::optional<std::pair<double,double>>
Obstacles::position_of(const Obstacle& ob, double t) const {
    if (t < ob.t_start || t > ob.t_end) return std::nullopt;
    if (ob.kind == Obstacle::Kind::Moving) {
        const double dt = t - ob.t_start;
        return std::make_pair(ob.x0 + ob.vx*dt, ob.y0 + ob.vy*dt);
    }
    return std::make_pair(ob.x0, ob.y0);
}

// -----------------------------------------------------------------------------
// Obstacles::active_at
//
// Enumerate all obstacles that are active at time t,
// returning their current positions and radii.
//
// ey_obs is passed through from Obstacle (to be filled by a higher-level
// Frenet projection if needed).
// -----------------------------------------------------------------------------
std::vector<Obstacles::Active> Obstacles::active_at(double t) const {
    std::vector<Active> out;
    out.reserve(items.size());
    for (size_t i = 0; i < items.size(); ++i) {
        const auto &ob = items[i];
        if (auto p = position_of(ob, t)) {
            out.push_back(Active{i, ob.id, p->first, p->second, ob.radius, ob.ey_obs});
        }
    }
    return out;
}
