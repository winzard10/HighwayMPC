/// ----------------------------------------- ///
/// Centerline Map Demo. (demo_centerline.cpp) ///
/// ----------------------------------------- ///

#include "centerline_map.hpp"
#include <iostream>
#include <filesystem>
#include <vector>
namespace fs = std::filesystem;

// Try to locate a default lane_centerlines.csv if the user doesn't pass a path.
//
// Search order:
//   1) ./data/lane_centerlines.csv                       (relative to CWD)
//   2) <exe_dir>/data/lane_centerlines.csv               (post-build layout)
//   3) <exe_dir>/../share/hwy_mpc/data/lane_centerlines.csv  (installed layout, optional)
//
// Returns:
//   - valid path if found
//   - empty path if nothing exists
static fs::path find_default_map_path(const char* argv0) {
    // Candidate locations to try when no path is provided:
    // 1) ./data/
    // 2) <exe_dir>/data/                (post-build copy)
    // 3) <exe_dir>/../share/hwy_mpc/data/  (after 'cmake --install')
    std::vector<fs::path> candidates;
    candidates.emplace_back("data/lane_centerlines.csv");

    fs::path exe = fs::absolute(argv0);
    fs::path exe_dir = exe.parent_path();
    candidates.emplace_back(exe_dir / "data" / "lane_centerlines.csv");
#ifdef HWY_MPC_INSTALLED_DATA_SUBDIR
    candidates.emplace_back(exe_dir / ".." / HWY_MPC_INSTALLED_DATA_SUBDIR / "lane_centerlines.csv");
#endif

    for (auto& p : candidates) {
        if (fs::exists(p)) return p;
    }
    return {}; // empty -> not found
}

int main(int argc, char** argv) {
    fs::path csv_path;
    if (argc > 1) {
        // If user provided a path on the command line, use that.
        csv_path = argv[1];
    } else {
        // Otherwise, try to auto-discover using default search paths.
        csv_path = find_default_map_path(argv[0]);
    }

    if (csv_path.empty()) {
        std::cerr << "Could not locate lane_centerlines.csv.\n"
                  << "Pass a path explicitly, or ensure data/ is next to the binary.\n";
        return 1;
    }

    CenterlineMap map;
    if (!map.load_csv(csv_path.string())) {
        std::cerr << "Failed to load: " << csv_path << "\n";
        return 1;
    }

    std::cout << "Loaded " << map.size() << " samples from: " << csv_path << "\n";
    std::cout << "s in [" << map.s_min() << ", " << map.s_max() << "]\n";

    // Query a few sample arclengths across the map range:
    //   - s_min
    //   - 25%, 50%, 75% of the span
    //   - s_max
    double qs[] = {map.s_min(), 0.25*(map.s_min()+map.s_max()),
                   0.5*(map.s_min()+map.s_max()),
                   0.75*(map.s_min()+map.s_max()),
                   map.s_max()};
    for (double s : qs) {
        auto R = map.right_lane_at(s);
        auto L = map.left_lane_at(s);
        std::cout << "s=" << s
                  << " | R=(" << R.x << "," << R.y << ")  L=(" << L.x << "," << L.y << ")"
                  << "  psi=" << R.psi << "  v_ref=" << R.v_ref << "\n";
    }
    return 0;
}
