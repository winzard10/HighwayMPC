cmake_minimum_required(VERSION 3.16)
project(hwy_mpc LANGUAGES CXX)

# =========================
# Global C++ configuration
# =========================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(MSVC)
  add_compile_options(/W4 /permissive-)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Where user-local packages (Eigen, OSQP, etc.) might live
list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/.local")

# =========================
# Dependencies
# =========================

# --- Eigen ---
find_package(Eigen3 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
  message(STATUS "Eigen3 not found via package config; falling back to /usr/include/eigen3")
  add_library(Eigen3::Eigen INTERFACE IMPORTED)
  set_target_properties(Eigen3::Eigen PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "/usr/include/eigen3")
endif()

# --- OsqpEigen ---
find_package(OsqpEigen REQUIRED)  # provides OsqpEigen::OsqpEigen

# =========================
# Source/include layout
# =========================
set(HWY_MPC_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(HWY_MPC_SRC_DIR     "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(HWY_MPC_DATA_DIR    "${CMAKE_CURRENT_SOURCE_DIR}/data")

# =========================
# Library: centerline
# =========================
add_library(centerline STATIC
  "${HWY_MPC_SRC_DIR}/centerline_map.cpp"
)

target_include_directories(centerline
  PUBLIC
    "${HWY_MPC_INCLUDE_DIR}"
)

target_link_libraries(centerline
  PUBLIC
    Eigen3::Eigen
)

# =========================
# Library: dynamics (vehicle + tire)
# =========================
add_library(dynamics STATIC
  "${HWY_MPC_SRC_DIR}/vehicle_model.cpp"
  "${HWY_MPC_SRC_DIR}/tire_model.cpp"
)

target_include_directories(dynamics
  PUBLIC
    "${HWY_MPC_INCLUDE_DIR}"
)

target_link_libraries(dynamics
  PUBLIC
    Eigen3::Eigen
)

# =========================
# Library: obstacles
# =========================
add_library(obstacles STATIC
  "${HWY_MPC_SRC_DIR}/obstacles.cpp"
)

target_include_directories(obstacles
  PUBLIC
    "${HWY_MPC_INCLUDE_DIR}"
)

# =========================
# Library: corridor_planner
# =========================
add_library(corridor_planner STATIC
  "${HWY_MPC_SRC_DIR}/corridor_planner.cpp"
)

target_include_directories(corridor_planner
  PUBLIC
    "${HWY_MPC_INCLUDE_DIR}"
)

target_link_libraries(corridor_planner
  PUBLIC
    Eigen3::Eigen
)

# =========================
# Library: acc
# =========================
add_library(acc STATIC
  "${HWY_MPC_SRC_DIR}/acc.cpp"
)

target_include_directories(acc
  PUBLIC
    "${HWY_MPC_INCLUDE_DIR}"
)

target_link_libraries(acc
  PUBLIC
    Eigen3::Eigen
)

# =========================
# Library: mpc (LTV-MPC)
# =========================
add_library(mpc STATIC
  "${HWY_MPC_SRC_DIR}/mpc_ltv.cpp"
)

target_include_directories(mpc
  PUBLIC
    "${HWY_MPC_INCLUDE_DIR}"
)

target_link_libraries(mpc
  PUBLIC
    Eigen3::Eigen
    OsqpEigen::OsqpEigen
    dynamics          # uses vehicle/tire symbols
    acc               # uses ACC helpers
)

# =========================
# Executable: centerline_demo
# =========================
add_executable(centerline_demo
  "${HWY_MPC_SRC_DIR}/demo_centerline.cpp"
)

target_include_directories(centerline_demo
  PRIVATE
    "${HWY_MPC_INCLUDE_DIR}"
)

target_link_libraries(centerline_demo
  PRIVATE
    centerline
)

# =========================
# Executable: sim_lane_follow
# =========================
add_executable(sim_lane_follow
  "${HWY_MPC_SRC_DIR}/sim_lane_follow.cpp"
)

target_include_directories(sim_lane_follow
  PRIVATE
    "${HWY_MPC_INCLUDE_DIR}"
)

target_link_libraries(sim_lane_follow
  PRIVATE
    centerline
    obstacles
    corridor_planner
    mpc
    dynamics      # sim calls vehicle/tire functions directly
    acc
    Eigen3::Eigen
)

# =========================
# Data handling (copy runtime CSVs)
# =========================
add_custom_command(TARGET centerline_demo POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:centerline_demo>/data"
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          "${HWY_MPC_DATA_DIR}"
          "$<TARGET_FILE_DIR:centerline_demo>/data"
  COMMENT "Copying data/ next to centerline_demo"
)

add_custom_command(TARGET sim_lane_follow POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:sim_lane_follow>/data"
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          "${HWY_MPC_DATA_DIR}"
          "$<TARGET_FILE_DIR:sim_lane_follow>/data"
  COMMENT "Copying data/ next to sim_lane_follow"
)

# Define the installed data subdir so the demo/sim can also locate CSVs after install
target_compile_definitions(centerline_demo PRIVATE HWY_MPC_INSTALLED_DATA_SUBDIR="share/hwy_mpc/data")
target_compile_definitions(sim_lane_follow  PRIVATE HWY_MPC_INSTALLED_DATA_SUBDIR="share/hwy_mpc/data")

# =========================
# Install rules (optional)
# =========================
install(TARGETS centerline_demo sim_lane_follow
  RUNTIME DESTINATION bin
)

install(DIRECTORY "${HWY_MPC_DATA_DIR}/"
  DESTINATION share/hwy_mpc/data
)

message(STATUS "Configured hwy_mpc at ${CMAKE_SOURCE_DIR}")
