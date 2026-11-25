from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

for _p in glob.glob(os.path.expanduser("~/venv/lib/python3.*/site-packages")) + glob.glob(
    os.path.expanduser("~/agent-work/telegram-bridge/venv/lib/python3.*/site-packages")
):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

PROJECT_NAME = "Prompt"
AUTHOR = "bio-informatician"
LICENSE_TYPE = "MIT"
SCENE_TAG = f"bio-informatician/{PROJECT_NAME.lower()}"

FRAME_WIDTH: int = 1080
FRAME_HEIGHT: int = 1920
DEFAULT_FPS: int = 30

CHAR_WIDTH: int = 12
CHAR_HEIGHT: int = 18
GRID_COLUMNS: int = 90
GRID_ROWS_CODE: int = 72
GRID_ROWS_FULL: int = 106

W, H, FPS = FRAME_WIDTH, FRAME_HEIGHT, DEFAULT_FPS
CW, CH, COLS = CHAR_WIDTH, CHAR_HEIGHT, GRID_COLUMNS
ROWS_CODE, ROWS_FULL = GRID_ROWS_CODE, GRID_ROWS_FULL
HANDLE = f"@{SCENE_TAG}"

TWO_PI: float = 2.0 * np.pi
CAMERA_DISTANCE: float = 7.0
CAM_D: float = CAMERA_DISTANCE

LIGHT_DIRECTION = np.array([-0.45, 0.75, -0.48], dtype=np.float64)
LIGHT_DIRECTION /= np.linalg.norm(LIGHT_DIRECTION)
LIGHT = LIGHT_DIRECTION

GLYPH_RAMP = "$QB%8&0#@oauenjzt[]{}<>rcvx+=1il!;:,"
RAMP = GLYPH_RAMP

def encode_glyph_index(char: str) -> int:
    return ord(char) - 32

GLYPH_RAMP_INDICES = np.array([encode_glyph_index(c) for c in GLYPH_RAMP], dtype=np.int32)
RAMP_G = GLYPH_RAMP_INDICES

SURFACE_FISH: int = 1
SURFACE_FIN: int = 2
SURFACE_ANEMONE: int = 3
SURFACE_BACKGROUND: int = 4

FISH, FIN, WEED, BACK = SURFACE_FISH, SURFACE_FIN, SURFACE_ANEMONE, SURFACE_BACKGROUND

EFFECTS: Dict[str, bool] = {
    "touch": True,
    "plankton": True,
    "shafts": True,
    "glint": True,
    "orbit": True,
    "prey": True,
}
FX = EFFECTS

SYSTEM_FONT_CANDIDATES: Tuple[str, ...] = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "/Library/Fonts/Courier New.ttf",
    "DejaVuSansMono.ttf",
    "consola.ttf",
    "cour.ttf",
)
FONT_PATHS = SYSTEM_FONT_CANDIDATES

PALETTE_ORANGE = np.array([255, 140, 28], dtype=np.float64) / 255.0
PALETTE_DEEP = np.array([242, 82, 16], dtype=np.float64) / 255.0
PALETTE_FACE = np.array([255, 192, 60], dtype=np.float64) / 255.0
PALETTE_WHITE = np.array([255, 255, 252], dtype=np.float64) / 255.0
PALETTE_DARK = np.array([65, 48, 88], dtype=np.float64) / 255.0
PALETTE_RIM = np.array([215, 235, 255], dtype=np.float64) / 255.0

ORANGE, DEEP, FACE = PALETTE_ORANGE, PALETTE_DEEP, PALETTE_FACE
WHITE, DARK, RIM = PALETTE_WHITE, PALETTE_DARK, PALETTE_RIM

TENTACLE_TIP = np.array([135, 248, 238], dtype=np.float64) / 255.0
TENTACLE_STEM = np.array([182, 118, 242], dtype=np.float64) / 255.0
TIP, STEM = TENTACLE_TIP, TENTACLE_STEM

TENTACLE_FILL_GLYPHS = np.array([encode_glyph_index(c) for c in ".::;;ioo08"], dtype=np.int32)
FILL_G = TENTACLE_FILL_GLYPHS

ROCK_GLYPHS = np.array([encode_glyph_index(c) for c in " . .:.:;:=+"], dtype=np.int32)
ROCK_G = ROCK_GLYPHS

SEABED_ELEVATION: float = -1.42
FLOOR = SEABED_ELEVATION

ANEMONE_MOUND_DIMENSIONS = np.array([1.10, 0.36, 0.80], dtype=np.float64)
MOUND = ANEMONE_MOUND_DIMENSIONS
ANEMONE_ROOT_POSITION = (0.05, -1.02, 0.40)
LAYOUT_WEED = ANEMONE_ROOT_POSITION

def calculate_cubic_ease(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    x_clamped = np.clip(x, 0.0, 1.0)
    return (3.0 - 2.0 * x_clamped) * x_clamped * x_clamped

def create_rgb_vector(*components: float) -> np.ndarray:
    return np.asarray(components, dtype=np.float64) / 255.0

def normalize_vector(vectors: np.ndarray) -> np.ndarray:
    norm = np.maximum(np.sqrt(np.sum(vectors * vectors, axis=-1, keepdims=True)), 1e-9)
    return vectors / norm

def rotate_vector_pitch_x(points: np.ndarray, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = np.moveaxis(points, -1, 0)
    return np.stack([x, c * y - s * z, s * y + c * z], axis=-1)

def rotate_vector_yaw_y(points: np.ndarray, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = np.moveaxis(points, -1, 0)
    return np.stack([c * x + s * z, y, -s * x + c * z], axis=-1)

def rotate_vector_roll_z(points: np.ndarray, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = np.moveaxis(points, -1, 0)
    return np.stack([c * x - s * y, s * x + c * y, z], axis=-1)

def determine_line_glyph(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    norm_dx, norm_dy = dx / CHAR_WIDTH, dy / CHAR_HEIGHT
    is_steep = np.abs(norm_dy) > 2.0 * np.abs(norm_dx)
    is_flat = np.abs(norm_dx) > 2.0 * np.abs(norm_dy)
    is_positive_slope = (norm_dx * norm_dy) < 0.0
    return np.select(
        [is_steep, is_flat, is_positive_slope],
        [encode_glyph_index("|"), encode_glyph_index("-"), encode_glyph_index("/")],
        encode_glyph_index("\\"),
    )

def generate_hash_float(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.modf(np.abs(np.sin(a * 127.1 + b * 311.7) * 43758.5453))[0]

PROFILE_STATIONS = np.array([0.0, 0.03, 0.08, 0.16, 0.30, 0.45, 0.60, 0.75, 0.88, 0.96, 1.0], dtype=np.float64)
PROFILE_HALF_HEIGHTS = np.array([0.0, 0.075, 0.125, 0.175, 0.222, 0.238, 0.218, 0.165, 0.10, 0.078, 0.075], dtype=np.float64)

PROFILE_S = PROFILE_STATIONS
PROFILE_H = PROFILE_HALF_HEIGHTS

CAUDAL_FIN_ROOT: float = 0.53
TAIL_ROOT = CAUDAL_FIN_ROOT

CYCLE_DURATION: float = 22.0
PECTORAL_FIN_HZ: float = 1.3
LOOP, PECT_HZ = CYCLE_DURATION, PECTORAL_FIN_HZ

ANGULAR_FREQ_1 = 3.0 * TWO_PI / CYCLE_DURATION
ANGULAR_FREQ_2 = 4.0 * TWO_PI / CYCLE_DURATION
ANGULAR_FREQ_3 = 8.0 * TWO_PI / CYCLE_DURATION
W1, W2, W3 = ANGULAR_FREQ_1, ANGULAR_FREQ_2, ANGULAR_FREQ_3

SWIM_HZ: float = 0.95
SWIM_STATE: Dict[str, float] = {
    "effort": 1.0,
    "turn": 0.0,
    "phase": 0.0,
    "hz": 0.95,
    "t": 0.0,
    "brake": 0.0,
}
SWIM = SWIM_STATE

def calculate_body_half_height(station: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return np.interp(station, PROFILE_STATIONS, PROFILE_HALF_HEIGHTS)

def calculate_body_half_width(station: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return calculate_body_half_height(station) * (0.58 - 0.22 * station)

def calculate_body_undulation_sway(station: Union[float, np.ndarray], current_time: float) -> Union[float, np.ndarray]:
    oscillation_beat = SWIM_STATE["phase"] - TWO_PI * SWIM_STATE["hz"] * (SWIM_STATE["t"] - current_time)
    spine_wave = (0.014 + 0.105 * (station ** 2)) * np.sin(oscillation_beat - 3.4 * station)
    steering_bend = SWIM_STATE["turn"] * ((station - 0.25) ** 2) * np.sign(station - 0.25)
    return SWIM_STATE["effort"] * spine_wave + steering_bend

def calculate_stripe_pattern(station: np.ndarray, flank_height: np.ndarray) -> np.ndarray:
    dorsoventral_bulge = np.exp(-((flank_height / 0.55) ** 2))
    white_bands = [
        (0.215 + 0.035 * (flank_height ** 2), 0.335 + 0.02 * (flank_height ** 2)),
        (0.515 - 0.055 * dorsoventral_bulge, 0.655 - 0.02 * dorsoventral_bulge),
        (0.905, 0.985),
    ]
    pigment_type = np.zeros(np.broadcast(station, flank_height).shape, dtype=np.int32)
    for band_start, band_end in white_bands:
        pigment_type = np.where((station > band_start - 0.016) & (station < band_end + 0.016), 2, pigment_type)
        pigment_type = np.where((station > band_start) & (station < band_end), 1, pigment_type)
    return pigment_type

def generate_fish_body_mesh(current_time: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    station_grid, theta_grid = np.meshgrid(
        np.linspace(0.0, 1.0, 280),
        np.linspace(0.0, TWO_PI, 120, endpoint=False),
        indexing="ij",
    )
    s_flat, theta_flat = station_grid.ravel(), theta_grid.ravel()
    height_profile = calculate_body_half_height(s_flat)
    width_profile = calculate_body_half_width(s_flat)
    sin_theta = np.sin(theta_flat)

    points = np.stack(
        [
            s_flat - 0.45,
            height_profile * sin_theta,
            width_profile * np.cos(theta_flat) + calculate_body_undulation_sway(s_flat, current_time),
        ],
        axis=-1,
    )
    height_gradient = np.gradient(PROFILE_HALF_HEIGHTS, PROFILE_STATIONS)
    normals = normalize_vector(
        np.stack(
            [
                -np.interp(s_flat, PROFILE_STATIONS, height_gradient) * width_profile,
                width_profile * sin_theta,
                height_profile * np.cos(theta_flat),
            ],
            axis=-1,
        )
    )

    pigment_kind = calculate_stripe_pattern(s_flat, sin_theta)
    ventral_shade = calculate_cubic_ease(sin_theta[:, None] * 0.9)
    base_color = PALETTE_ORANGE * (1.0 - 0.5 * ventral_shade) + PALETTE_DEEP * (0.5 * ventral_shade)
    base_color = np.where((s_flat < 0.2)[:, None], base_color * 0.45 + PALETTE_FACE * 0.55, base_color)
    final_color = np.select(
        [(pigment_kind == 1)[:, None], (pigment_kind == 2)[:, None]],
        [PALETTE_WHITE, PALETTE_DARK],
        base_color,
    )
    return points, normals, final_color, pigment_kind

def _generate_dorsal_fin_mesh(current_time: float, record_fn) -> None:
    for s_start, s_end, fin_height, lean_factor in [(0.25, 0.57, 0.15, 0.45), (0.53, 0.88, 0.125, 0.85)]:
        a_mesh, b_mesh = [grid.ravel() for grid in np.meshgrid(np.linspace(0.0, 1.0, 110), np.linspace(0.0, 1.0, 30), indexing="ij")]
        station = s_start + (s_end - s_start) * a_mesh
        reach = fin_height * (np.sin(np.pi * (a_mesh ** 0.8)) ** 0.8)
        root = np.stack([station - 0.45, calculate_body_half_height(station) * 0.93, np.zeros_like(station)], axis=-1)
        outward = np.stack([lean_factor * reach, reach, np.zeros_like(station)], axis=-1)
        p = root + outward * b_mesh[:, None]
        fin_phase = SWIM_STATE["phase"] - 3.4 * station - 2.0 * b_mesh
        p[:, 2] = calculate_body_undulation_sway(p[:, 0] + 0.45, current_time) + 0.05 * (b_mesh ** 1.5) * np.sin(
            fin_phase - 3.0 * (s_start > 0.4)
        )
        q = p + normalize_vector(outward) * 0.03
        record_fn(p, q, b_mesh, margin=0.84, rim=0.96)

def _generate_anal_fin_mesh(current_time: float, record_fn) -> None:
    a_mesh, b_mesh = [grid.ravel() for grid in np.meshgrid(np.linspace(0.0, 1.0, 80), np.linspace(0.0, 1.0, 26), indexing="ij")]
    station = 0.62 + 0.25 * a_mesh
    reach = 0.135 * (np.sin(np.pi * np.clip(a_mesh * 0.9 + 0.1, 0.0, 1.0)) ** 0.6)
    outward = np.stack([0.7 * reach, -reach, np.zeros_like(station)], axis=-1)
    p = np.stack([station - 0.45, -calculate_body_half_height(station) * 0.93, np.zeros_like(station)], axis=-1) + outward * b_mesh[:, None]
    fin_phase = SWIM_STATE["phase"] - 3.4 * station - 2.0 * b_mesh
    p[:, 2] = calculate_body_undulation_sway(p[:, 0] + 0.45, current_time) + 0.045 * (b_mesh ** 1.5) * np.sin(
        fin_phase + 1.0
    )
    record_fn(p, p + normalize_vector(outward) * 0.03, b_mesh, margin=0.72)

def _generate_pelvic_fins_mesh(current_time: float, record_fn) -> None:
    for lateral_side in (-1, 1):
        a_mesh, b_mesh = [grid.ravel() for grid in np.meshgrid(np.linspace(0.0, 1.0, 46), np.linspace(0.0, 1.0, 26), indexing="ij")]
        station = 0.37 + 0.09 * a_mesh
        reach = 0.17 * (0.55 + 0.45 * (np.sin(np.pi * a_mesh) ** 0.7))
        outward = np.stack([0.55 * reach, -reach, lateral_side * 0.25 * reach], axis=-1)
        p = np.stack([station - 0.45, -calculate_body_half_height(station) * 0.90, lateral_side * 0.04 + np.zeros_like(station)], axis=-1) + outward * b_mesh[:, None]
        fin_phase = SWIM_STATE["phase"] - 1.2 * b_mesh
        p[:, 2] += calculate_body_undulation_sway(station, current_time) + 0.04 * (b_mesh ** 2) * np.sin(fin_phase + lateral_side * 0.4 + 2.0 * a_mesh)
        p[:, 0] += 0.05 * (b_mesh ** 2) * np.sin(fin_phase + 1.5)
        record_fn(p, p + normalize_vector(outward) * 0.03, b_mesh, margin=0.78, rim=2.0)

def _generate_caudal_fin_mesh(current_time: float, record_fn) -> None:
    a_mesh, b_mesh = [grid.ravel() for grid in np.meshgrid(np.linspace(-1.0, 1.0, 110), np.linspace(0.0, 1.0, 46), indexing="ij")]
    fan_angle = 0.95 * a_mesh
    radius = b_mesh * 0.31 * (1.0 - 0.13 * (a_mesh ** 2))

    arc_stations = np.linspace(0.0, 1.0, 46)
    wave_sample_stations = 0.98 + 0.31 * arc_stations
    tangent_slopes = np.clip(
        np.arctan(
            np.gradient(calculate_body_undulation_sway(wave_sample_stations, current_time - 0.12 * arc_stations), wave_sample_stations)
            * (1.0 + 0.3 * arc_stations)
        ),
        -0.95,
        0.95,
    )
    arc_step = arc_stations[1] - arc_stations[0]
    integrated_x = np.concatenate([[0.0], np.cumsum(np.cos(tangent_slopes[1:]) * arc_step)])
    integrated_z = np.concatenate([[0.0], np.cumsum(np.sin(tangent_slopes[1:]) * arc_step)])

    radial_index = np.arange(len(b_mesh)) % 46
    fan_span = 0.31 * (1.0 - 0.13 * (a_mesh ** 2))
    tail_x = CAUDAL_FIN_ROOT + fan_span * np.cos(fan_angle) * integrated_x[radial_index]
    tail_y = 0.075 * a_mesh * (1.0 - b_mesh) + radius * np.sin(fan_angle)
    tail_phase = SWIM_STATE["phase"] - 3.4 * 0.98 - 1.8 * b_mesh
    p = np.stack(
        [
            tail_x,
            tail_y,
            calculate_body_undulation_sway(0.98, current_time)
            + fan_span * integrated_z[radial_index]
            + 0.035 * (b_mesh ** 1.8) * np.sin(tail_phase + 2.0 * np.abs(a_mesh)),
        ],
        axis=-1,
    )
    ray = np.stack([np.cos(fan_angle), np.sin(fan_angle), np.zeros_like(fan_angle)], axis=-1)
    record_fn(p, p + ray * 0.03, b_mesh, margin=0.76, rim=0.94)

def _generate_pectoral_fins_mesh(current_time: float, record_fn) -> None:
    for lateral_side in (-1, 1):
        a_mesh, b_mesh = [grid.ravel() for grid in np.meshgrid(np.linspace(-1.0, 1.0, 70), np.linspace(0.0, 1.0, 36), indexing="ij")]
        phase_offset = 0.0 if lateral_side > 0 else 0.4
        stroke_beat = TWO_PI * PECTORAL_FIN_HZ * current_time + phase_offset - 2.1 * b_mesh - 0.5 * np.abs(a_mesh)
        stroke_wave = np.sin(stroke_beat) + 0.25 * np.sin(2.0 * stroke_beat + 0.6)
        effort_tuck = min(SWIM_STATE["effort"] - 0.55, 1.0)
        wing_spread = 0.72 - 0.20 * effort_tuck + 0.55 * SWIM_STATE["brake"] + (0.48 - 0.14 * effort_tuck) * stroke_wave

        vector_back = np.array([0.72, -0.69, 0.0])
        vector_out = np.array([0.0, 0.0, float(lateral_side)])
        vector_up = np.array([0.69, 0.72, 0.0])
        pectoral_axis = vector_back * np.cos(wing_spread)[:, None] + vector_out * np.sin(wing_spread)[:, None]

        blade_angle = 0.8 * a_mesh
        blade_radius = b_mesh * 0.21 * (1.0 - 0.12 * (a_mesh ** 2))
        hinge_pivot = np.array([0.375 - 0.45, -0.07, lateral_side * calculate_body_half_width(0.375) * 0.90])
        fin_ray = pectoral_axis * np.cos(blade_angle)[:, None] + vector_up * np.sin(blade_angle)[:, None]
        p = (
            hinge_pivot
            + fin_ray * blade_radius[:, None]
            - vector_out * (0.07 * (b_mesh ** 2) * np.cos(stroke_beat))[:, None]
        )
        record_fn(p, p + fin_ray * 0.03, b_mesh, margin=0.74, rim=0.93)

def generate_fish_fins_mesh(current_time: float) -> List[np.ndarray]:
    fin_components: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    def record_fin_patch_quads(
        p_coords: np.ndarray,
        q_coords: np.ndarray,
        b_fraction: np.ndarray,
        margin: float = 0.74,
        rim: float = 0.93,
    ) -> None:
        color = np.select(
            [(b_fraction > rim)[:, None], (b_fraction > margin)[:, None]],
            [PALETTE_RIM, PALETTE_DARK],
            PALETTE_ORANGE * 0.94 + PALETTE_FACE * 0.06,
        )
        luminosity = np.select(
            [b_fraction > rim, b_fraction > margin],
            [0.9, 0.5],
            0.72 + 0.2 * b_fraction,
        )
        fin_components.append((p_coords, q_coords, color, luminosity))

    _generate_dorsal_fin_mesh(current_time, record_fin_patch_quads)
    _generate_anal_fin_mesh(current_time, record_fin_patch_quads)
    _generate_pelvic_fins_mesh(current_time, record_fin_patch_quads)
    _generate_caudal_fin_mesh(current_time, record_fin_patch_quads)
    _generate_pectoral_fins_mesh(current_time, record_fin_patch_quads)

    return [np.concatenate([part[dim] for part in fin_components]) for dim in range(4)]

def generate_fish_eyes_mesh(current_time: float) -> List[np.ndarray]:
    eye_segments: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for lateral_side in (-1, 1):
        radial_grid, phi_grid = [grid.ravel() for grid in np.meshgrid(np.linspace(0.0, 1.0, 9), np.linspace(0.0, TWO_PI, 28), indexing="ij")]
        radius = 0.041 * radial_grid
        station = 0.115 + radius * np.cos(phi_grid)
        y_coord = 0.05 + radius * np.sin(phi_grid)
        z_elevation = lateral_side * (
            calculate_body_half_width(station) * np.sqrt(np.clip(1.0 - (y_coord / calculate_body_half_height(station)) ** 2, 0.05, 1.0)) + 0.03
        )
        points = np.stack([station - 0.45, y_coord, z_elevation + calculate_body_undulation_sway(station, current_time)], axis=-1)

        color = np.where((radial_grid < 0.6)[:, None], create_rgb_vector(58, 34, 30), create_rgb_vector(255, 190, 96))
        is_glint = (radial_grid > 0.15) & (radial_grid < 0.42) & (np.abs(phi_grid - 2.2) < 0.45)
        color = np.where(is_glint[:, None], create_rgb_vector(235, 225, 215), color)
        luminosity = np.where(radial_grid < 0.6, 0.62, 0.72)
        eye_segments.append((points, color, luminosity))

    return [np.concatenate([segment[dim] for segment in eye_segments]) for dim in range(3)]

ORBITAL_WAYPOINTS: List[Tuple[float, float, float, float, float, float]] = [
    (-0.50, 0.42, 0.02, 0.55, 0.0, 0.30),
    (-1.30, 0.14, -0.38, 0.65, 0.0, 0.34),
    (-1.72, -0.28, -0.80, 0.60, 0.0, 0.30),
    (-1.30, -0.58, -1.08, 0.80, 0.0, 0.30),
    (-0.70, -0.50, -1.14, 2.10, 0.0, 0.30),
    (0.30, -0.42, -1.10, 1.15, 0.0, 0.34),
    (1.50, -0.10, -0.60, 0.95, 0.0, 0.34),
    (1.85, 0.35, 0.10, 0.60, 0.0, 0.30),
    (1.25, 0.66, 0.62, 0.75, 0.0, 0.34),
    (0.16, 0.05, 0.27, 0.65, 0.0, 0.16),
    (0.04, 0.01, 0.10, 0.55, 0.0, 0.15),
    (0.22, 0.00, 0.04, 0.55, 0.0, 0.15),
    (0.30, 0.04, 0.24, 0.55, 0.25, 0.15),
]
ROUND = ORBITAL_WAYPOINTS

THRUST_COEFF: float = 2.1
DRAG_QUADRATIC: float = 2.2
DRAG_LINEAR: float = 0.6
THRUST, DRAG2, DRAG1 = THRUST_COEFF, DRAG_QUADRATIC, DRAG_LINEAR

PREY_APPEAR_INDEX: int = 2
PREY_TARGET_INDEX: int = 4
PREY_SEEN, PREY_MARK = PREY_APPEAR_INDEX, PREY_TARGET_INDEX

PREY_SPAWN_POSITION = np.array([-0.02, -0.17, -1.14], dtype=np.float64)
PREY_DRIFT_VELOCITY = np.array([-0.020, -0.055, 0.0], dtype=np.float64)
PREY_FROM, PREY_DRIFT = PREY_SPAWN_POSITION, PREY_DRIFT_VELOCITY

PREY_ATTENTION_PAUSE: float = 0.55
PREY_CAPTURE_RADIUS: float = 0.12
MOUTH_FORWARD_OFFSET: float = 0.45
PREY_PAUSE = PREY_ATTENTION_PAUSE
PREY_BITE, MOUTH = PREY_CAPTURE_RADIUS, MOUTH_FORWARD_OFFSET

PREY_CAMERA_DELAY: float = 0.5
PREY_CAMERA_PAN_TIME: float = 2.6
PREY_CAMERA_PULLBACK: float = 0.36
PREY_LOOK, PREY_TURN, PREY_WIDE = PREY_CAMERA_DELAY, PREY_CAMERA_PAN_TIME, PREY_CAMERA_PULLBACK

PREY_COLOR = np.array([1.0, 0.84, 0.42], dtype=np.float64)
PREY_INK = PREY_COLOR

def sample_food_particle(age: float) -> np.ndarray:
    rocking_offset = np.array([0.035 * np.sin(2.3 * age), 0.0, 0.02 * np.sin(1.7 * age + 1.0)])
    return PREY_SPAWN_POSITION + PREY_DRIFT_VELOCITY * age + rocking_offset

_rng_motes = np.random.default_rng(23)
AMBIENT_MOTES = np.concatenate(
    [
        _rng_motes.uniform([-3.0, -1.25, -1.7], [3.1, 1.0, 1.2], (300, 3)),
        _rng_motes.uniform([-1.9, -0.85, -1.55], [1.9, -0.10, -0.70], (170, 3)),
    ]
)
MOTES = AMBIENT_MOTES

MOTE_FADE_TIME: float = 1.6
MOTE_COLOR = np.array([0.50, 0.92, 1.0], dtype=np.float64)
MOTE_FADE, MOTE_INK = MOTE_FADE_TIME, MOTE_COLOR

WAKE_POOL_SIZE: int = 460
WAKE_EMISSION_RATE: float = 130.0
WAKE_LIFESPAN: float = 2.6
WAKE, WAKE_RATE, WAKE_LIFE = WAKE_POOL_SIZE, WAKE_EMISSION_RATE, WAKE_LIFESPAN

GLOW_FADE_TIME: float = 0.7
GLOW_PROPAGATION_SPEED: float = 13.0
GLOW_COLOR = np.array([0.80, 1.0, 0.97], dtype=np.float64)
GLOW_FADE, GLOW_RUN, GLOW_INK = GLOW_FADE_TIME, GLOW_PROPAGATION_SPEED, GLOW_COLOR

SIM_TIME_STEP: float = 1.0 / 60.0
SIM_WARMUP_STEPS: int = 240
STEP, WARM = SIM_TIME_STEP, SIM_WARMUP_STEPS

TRAJECTORY_TRACK: Dict[str, Any] = {}
TRACK = TRAJECTORY_TRACK

SCENE_PIVOT_CENTER = np.array([0.05, -0.55, 0.40], dtype=np.float64)
CAMERA_ORBIT_RATE: float = 0.24
CAMERA_TILT_RATE: float = 0.22
PIVOT, ORBIT, TILT = SCENE_PIVOT_CENTER, CAMERA_ORBIT_RATE, CAMERA_TILT_RATE

MEASURE_BAR: float = 2.0
TOTAL_BARS: int = 11
BAR, BARS = MEASURE_BAR, TOTAL_BARS
LOOP_START_X: float = 1.05
START_X = LOOP_START_X

def precompute_swim_trajectory() -> None:
    global LOOP, W1, W2, W3, PECT_HZ
    position = np.array(ORBITAL_WAYPOINTS[0][:3], dtype=np.float64)
    heading_yaw = np.pi + 0.5
    pitch_elevation = 0.0
    forward_velocity = 0.0
    active_effort = 0.0
    tail_oscillation_phase = 0.0
    target_mark_idx = 1
    rest_timer = 0.0

    simulation_history: List[Tuple] = []
    cycle_home_indices: List[int] = []
    circling_timer = 0.0
    food_timer = -1.0
    predation_stare_timer = 0.0

    while len(cycle_home_indices) < 3 and len(simulation_history) < 60 * 240:
        current_target = ORBITAL_WAYPOINTS[(target_mark_idx + 1) % len(ORBITAL_WAYPOINTS)] if rest_timer > 0 else ORBITAL_WAYPOINTS[target_mark_idx]
        is_hunting = EFFECTS["prey"] and (target_mark_idx == PREY_TARGET_INDEX) and (food_timer >= 0) and (rest_timer <= 0)
        if food_timer >= 0:
            food_timer += SIM_TIME_STEP

        target_coords = sample_food_particle(food_timer) if is_hunting else np.array(current_target[:3])
        vector_to_target = target_coords - position
        distance_to_target = np.linalg.norm(vector_to_target)

        snout_mouth_pos = position + MOUTH_FORWARD_OFFSET * np.array(
            [
                np.cos(heading_yaw) * np.cos(pitch_elevation),
                np.sin(pitch_elevation),
                np.sin(heading_yaw) * np.cos(pitch_elevation),
            ]
        )

        if rest_timer > 0:
            rest_timer -= SIM_TIME_STEP
            desired_effort = 0.0
            if rest_timer <= 0:
                target_mark_idx = (target_mark_idx + 1) % len(ORBITAL_WAYPOINTS)
        else:
            arrival_scale = np.clip(distance_to_target / 0.35, 0.5, 1.0) if current_target[4] > 0 else 1.0
            desired_effort = current_target[3] * arrival_scale
            if is_hunting and predation_stare_timer > 0:
                predation_stare_timer -= SIM_TIME_STEP
                desired_effort = 0.05

            along_heading = vector_to_target[0] * np.cos(heading_yaw) + vector_to_target[2] * np.sin(heading_yaw)
            circling_timer = circling_timer + SIM_TIME_STEP if distance_to_target < 3.0 * current_target[5] else 0.0
            caught_target = (
                (np.linalg.norm(sample_food_particle(food_timer) - snout_mouth_pos) < PREY_CAPTURE_RADIUS)
                if is_hunting
                else (distance_to_target < current_target[5])
            )
            passed_waypoint = (current_target[4] > 0) and (distance_to_target < 2.5 * current_target[5]) and (along_heading < 0)

            if caught_target or passed_waypoint or (circling_timer > 1.4):
                circling_timer = 0.0
                if is_hunting:
                    food_timer = -1.0
                elif EFFECTS["prey"] and (target_mark_idx == PREY_APPEAR_INDEX - 1):
                    food_timer = 0.0

                if target_mark_idx == 0:
                    cycle_home_indices.append(len(simulation_history))

                rest_timer = current_target[4]
                if rest_timer <= 0:
                    target_mark_idx = (target_mark_idx + 1) % len(ORBITAL_WAYPOINTS)
                    predation_stare_timer = PREY_ATTENTION_PAUSE if (EFFECTS["prey"] and target_mark_idx == PREY_TARGET_INDEX and food_timer >= 0) else 0.0

        target_angle_diff = (np.arctan2(vector_to_target[2], vector_to_target[0]) - heading_yaw + np.pi) % TWO_PI - np.pi
        turn_responsiveness = 0.35 if rest_timer > 0 else 0.5 + 2.4 * active_effort
        if predation_stare_timer > 0:
            turn_responsiveness = max(turn_responsiveness, 2.2)

        desired_effort *= 0.40 + 0.60 * (max(np.cos(target_angle_diff), 0.0) ** 2)
        heading_yaw += np.clip(target_angle_diff * turn_responsiveness, -1.9, 1.9) * SIM_TIME_STEP

        target_pitch = 0.8 * np.arcsin(np.clip(vector_to_target[1] / max(distance_to_target, 1e-6), -0.6, 0.6)) * (0.25 if rest_timer > 0 else 1.0)
        pitch_elevation += (target_pitch - pitch_elevation) * 0.7 * turn_responsiveness * SIM_TIME_STEP

        effort_growth_rate = 2.2 if desired_effort < active_effort else 1.1 + 2.2 * max(desired_effort - 1.2, 0.0)
        active_effort += (desired_effort - active_effort) * effort_growth_rate * SIM_TIME_STEP

        beat_hz = 0.70 + 1.05 * active_effort
        tail_oscillation_phase += TWO_PI * beat_hz * SIM_TIME_STEP

        thrust_force = THRUST_COEFF * active_effort * (0.30 + 0.70 * (np.sin(tail_oscillation_phase - 0.9) ** 2))
        forward_velocity += (thrust_force - DRAG_QUADRATIC * forward_velocity * abs(forward_velocity) - DRAG_LINEAR * forward_velocity) * SIM_TIME_STEP

        direction_vector = np.array(
            [
                np.cos(heading_yaw) * np.cos(pitch_elevation),
                np.sin(pitch_elevation),
                np.sin(heading_yaw) * np.cos(pitch_elevation),
            ]
        )
        position = position + forward_velocity * SIM_TIME_STEP * direction_vector
        simulation_history.append(
            (
                position[0],
                position[1],
                position[2],
                heading_yaw,
                pitch_elevation,
                forward_velocity,
                active_effort,
                tail_oscillation_phase,
                *(sample_food_particle(food_timer) if food_timer >= 0 else position),
                food_timer,
                target_mark_idx,
                float(predation_stare_timer > 0),
            )
        )

    lap_data = np.array(simulation_history[cycle_home_indices[1]:cycle_home_indices[2] + 1])
    start_index = next(
        idx
        for idx in range(len(lap_data) - 1)
        if lap_data[idx, 0] > LOOP_START_X
        and lap_data[idx + 1, 0] <= LOOP_START_X
        and np.cos(lap_data[idx, 3]) < 0.0
        and lap_data[idx, 1] > 0.2
    )

    continuation_offset = (lap_data[-1] - lap_data[0]) * np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    stabilized_lap = np.concatenate([lap_data[start_index:-1], lap_data[:start_index + 1] + continuation_offset])

    raw_lap_seconds = (len(stabilized_lap) - 1) * SIM_TIME_STEP
    if abs(raw_lap_seconds / (TOTAL_BARS * MEASURE_BAR) - 1.0) < 0.08:
        LOOP = TOTAL_BARS * MEASURE_BAR
    else:
        LOOP = round(raw_lap_seconds * DEFAULT_FPS) / DEFAULT_FPS

    W1, W2, W3 = 3.0 * TWO_PI / LOOP, 4.0 * TWO_PI / LOOP, 8.0 * TWO_PI / LOOP
    PECT_HZ = round(1.3 * LOOP) / LOOP

    resample_count = int(round(LOOP / SIM_TIME_STEP))
    resample_times = np.linspace(0.0, len(stabilized_lap) - 1, resample_count + 1)
    lower_idx = np.clip(resample_times.astype(int), 0, len(stabilized_lap) - 2)
    fractional_weights = (resample_times - lower_idx)[:, None]

    aim_marks = stabilized_lap[lower_idx, 12].astype(int)
    staring_flags = stabilized_lap[lower_idx, 13] > 0
    resampled_lap = stabilized_lap[lower_idx] * (1.0 - fractional_weights) + stabilized_lap[lower_idx + 1] * fractional_weights

    cyclic_gap = resampled_lap[-1] - resampled_lap[0]
    cyclic_gap[3] = (cyclic_gap[3] + np.pi) % TWO_PI - np.pi
    cyclic_gap[7] = cyclic_gap[7] - TWO_PI * round(cyclic_gap[7] / TWO_PI)
    cyclic_gap[8:] = 0.0
    resampled_lap -= np.linspace(0.0, 1.0, resample_count + 1)[:, None] * cyclic_gap

    rate_func = lambda col: np.gradient(col, SIM_TIME_STEP)
    pos_x, pos_y, pos_z, yaw_arr, pitch_arr, vel_arr, effort_arr, phase_arr = resampled_lap[:, :8].T
    food_status = np.where(resampled_lap[:, 11] >= 0, resampled_lap[:, 11], -1.0)
    bite_moments = [i for i in range(1, resample_count + 1) if food_status[i] < 0.0 <= food_status[i - 1]]

    TRAJECTORY_TRACK.update(
        mark=aim_marks,
        spot=float(np.argmax(staring_flags)) * SIM_TIME_STEP if staring_flags.any() else -99.0,
        prey=resampled_lap[:, 8:11],
        food=food_status,
        bite=bite_moments[0] * SIM_TIME_STEP if bite_moments else -99.0,
        bite_at=resampled_lap[bite_moments[0] - 1, 8:11] if bite_moments else np.zeros(3),
        n=resample_count,
        place=resampled_lap[:, :3],
        a=yaw_arr,
        th=pitch_arr,
        v=vel_arr,
        e=effort_arr,
        phi=phase_arr,
        hz=rate_func(phase_arr) / TWO_PI,
        spin=rate_func(yaw_arr),
        slowing=np.clip(-rate_func(vel_arr), 0.0, None),
        speed=np.linalg.norm(np.gradient(resampled_lap[:, :3], SIM_TIME_STEP, axis=0), axis=1),
        lap_seconds=(cycle_home_indices[2] - cycle_home_indices[1]) * SIM_TIME_STEP,
    )

    def apply_exponential_decay_filter(signal: np.ndarray, response_tau: float) -> np.ndarray:
        buffered_signal = np.empty(2 * resample_count)
        current_val = signal[0]
        for step_idx in range(2 * resample_count):
            current_val += (signal[step_idx % resample_count] - current_val) * SIM_TIME_STEP / response_tau
            buffered_signal[step_idx] = current_val
        return np.append(buffered_signal[resample_count:], buffered_signal[resample_count])

    predation_attention = np.clip((food_status - PREY_CAMERA_DELAY) / PREY_CAMERA_PAN_TIME, 0.0, 1.0)
    predation_attention = (predation_attention * predation_attention * (3.0 - 2.0 * predation_attention))[:, None]
    prey_camera_gap = (resampled_lap[:, 8:11] - resampled_lap[:, :3]) * predation_attention
    wide_framing = 1.0 - PREY_CAMERA_PULLBACK * np.clip(np.linalg.norm(prey_camera_gap, axis=1) / 1.2, 0.0, 1.0)

    TRAJECTORY_TRACK.update(
        cam_x=apply_exponential_decay_filter(pos_x + 0.5 * prey_camera_gap[:, 0], 0.55),
        cam_y=apply_exponential_decay_filter(pos_y + 0.5 * prey_camera_gap[:, 1], 0.80),
        cam_z=apply_exponential_decay_filter(pos_z + 0.5 * prey_camera_gap[:, 2], 0.90),
        lead=apply_exponential_decay_filter(np.cos(yaw_arr), 0.90),
        zoom=apply_exponential_decay_filter(
            1.34 - 0.52 * np.clip(vel_arr / 0.75, 0.0, 1.0) - 0.22 * np.clip((vel_arr - 0.58) / 0.28, 0.0, 1.0),
            1.0,
        )
        * apply_exponential_decay_filter(wide_framing, 0.25),
    )

    is_among_anemone = (np.abs(pos_x - 0.05) < 1.3) & (pos_y < 0.12) & (pos_z > -1.35)
    active_code_section = np.where(
        is_among_anemone,
        3,
        np.where(effort_arr > 0.62, 1, np.where(np.abs(TRAJECTORY_TRACK["spin"]) > 0.7, 0, 2)),
    )
    section_history: List[int] = []
    current_section = active_code_section[0]
    switch_delay = 0

    for step_idx in range(2 * resample_count + 1):
        modulo_idx = step_idx % resample_count
        switch_delay += 1
        if (
            active_code_section[modulo_idx] != current_section
            and switch_delay > 1.7 / SIM_TIME_STEP
            and (active_code_section[(modulo_idx + np.arange(24)) % resample_count] == active_code_section[modulo_idx]).all()
        ):
            current_section = active_code_section[modulo_idx]
            switch_delay = 0
        section_history.append(current_section)

    TRAJECTORY_TRACK["section"] = np.array(section_history[resample_count:], dtype=np.int32)

precompute_swim_trajectory()

def sample_kinematic_trajectory(current_time: float) -> Dict[str, Any]:
    fractional_index = (current_time % LOOP) / SIM_TIME_STEP
    base_index = int(fractional_index) % TRAJECTORY_TRACK["n"]
    lerp_weight = fractional_index - int(fractional_index)

    sample_metric = lambda key: TRAJECTORY_TRACK[key][base_index] * (1.0 - lerp_weight) + TRAJECTORY_TRACK[key][base_index + 1] * lerp_weight
    metrics = {
        key: sample_metric(key)
        for key in (
            "place",
            "a",
            "th",
            "v",
            "speed",
            "e",
            "phi",
            "hz",
            "spin",
            "slowing",
            "cam_x",
            "cam_y",
            "cam_z",
            "lead",
            "zoom",
        )
    }
    metrics.update(
        section=int(TRAJECTORY_TRACK["section"][base_index]),
        prey=TRAJECTORY_TRACK["prey"][base_index],
        food=TRAJECTORY_TRACK["food"][base_index],
        mark=int(TRAJECTORY_TRACK["mark"][base_index]),
    )
    return metrics

def evaluate_swimming_pose(current_time: float) -> Tuple[np.ndarray, float, float]:
    state = sample_kinematic_trajectory(current_time)
    head_wag = 0.05 * (0.4 + state["e"]) * np.sin(state["phi"] + 0.8)
    return state["place"], np.pi - state["a"] + head_wag, -state["th"]

def assemble_fish_model(current_time: float) -> List[np.ndarray]:
    world_pos, yaw_heading, pitch_inclination = evaluate_swimming_pose(current_time)
    state = sample_kinematic_trajectory(current_time)
    SWIM_STATE.update(
        effort=min(0.5 + 1.25 * state["e"], 2.0),
        turn=float(np.clip(0.11 * state["spin"], -0.2, 0.2)),
        phase=state["phi"],
        hz=state["hz"],
        t=current_time,
        brake=float(np.clip(1.6 * state["slowing"], 0.0, 0.7)),
    )

    spine_stations = np.linspace(0.0, 1.5, 76)
    spine_slopes = np.gradient(calculate_body_undulation_sway(spine_stations, current_time), spine_stations)
    arc_reach = np.concatenate(
        [[0.0], np.cumsum((spine_stations[1] - spine_stations[0]) / np.sqrt(1.0 + spine_slopes[1:] ** 2))]
    )

    def transform_pose_coordinate(coords: np.ndarray, is_vector: bool = False) -> np.ndarray:
        transformed = coords if is_vector else coords.copy()
        if not is_vector:
            body_termination = np.interp(0.98, spine_stations, arc_reach) - 0.45
            transformed[:, 0] = np.where(
                transformed[:, 0] > CAUDAL_FIN_ROOT,
                body_termination + transformed[:, 0] - CAUDAL_FIN_ROOT,
                np.interp(transformed[:, 0] + 0.45, spine_stations, arc_reach) - 0.45,
            )
        rotated = rotate_vector_yaw_y(rotate_vector_roll_z(transformed, pitch_inclination), yaw_heading)
        return rotated if is_vector else rotated + world_pos

    body_points, body_normals, body_colors, body_pigments = generate_fish_body_mesh(current_time)
    world_normals = np.nan_to_num(transform_pose_coordinate(body_normals, is_vector=True))
    diffuse_illumination = np.clip(world_normals.dot(LIGHT_DIRECTION), 0.0, 1.0)

    head_vector = -transform_pose_coordinate(np.array([[-1.0, 0.0, 0.0]]), is_vector=True)[0, 2]
    glint_facing = calculate_cubic_ease((head_vector - 0.35) / 0.45) if EFFECTS["glint"] else 0.0
    specular_halfway = normalize_vector(LIGHT_DIRECTION + np.array([0.0, 0.0, -1.0]))
    specular_sheen = (
        (np.clip(world_normals.dot(specular_halfway), 0.0, 1.0) ** 14)
        * (0.25 + 0.75 * glint_facing)
        * (1.0 if EFFECTS["glint"] else 0.0)
    )

    shade_factor = 0.58 + 0.42 * diffuse_illumination
    ocean_ambient_bounce = create_rgb_vector(35, 55, 80) * (1.0 - diffuse_illumination)[:, None]
    luminosity = np.where(
        body_pigments == 2,
        0.52,
        np.where(body_pigments == 1, 0.92 + 0.08 * diffuse_illumination, 0.75 + 0.25 * diffuse_illumination),
    )
    shaded_tone = body_colors * np.where(
        body_pigments == 2,
        1.0,
        np.where(body_pigments == 1, 0.88 + 0.12 * diffuse_illumination, shade_factor),
    )[:, None] + ocean_ambient_bounce * (body_pigments != 1)[:, None]
    shaded_tone = np.clip(shaded_tone + 0.75 * specular_sheen[:, None] * (body_pigments != 2)[:, None], 0.0, 1.0)
    luminosity = np.clip(luminosity + 0.3 * specular_sheen, 0.0, 1.0)

    points_list = [transform_pose_coordinate(body_points)]
    rays_list = [transform_pose_coordinate(body_points)]
    colors_list = [shaded_tone]
    luma_list = [luminosity]
    kind_list = [np.full(len(body_points), SURFACE_FISH)]
    is_ray_list = [np.zeros(len(body_points), dtype=bool)]

    fin_p, fin_q, fin_colors, fin_luma = generate_fish_fins_mesh(current_time)
    points_list.append(transform_pose_coordinate(fin_p))
    rays_list.append(transform_pose_coordinate(fin_q))
    colors_list.append(fin_colors * 0.92)
    luma_list.append(fin_luma)
    kind_list.append(np.full(len(fin_p), SURFACE_FIN))
    is_ray_list.append(np.ones(len(fin_p), dtype=bool))

    eye_p, eye_colors, eye_luma = generate_fish_eyes_mesh(current_time)
    posed_eyes = transform_pose_coordinate(eye_p)
    posed_eyes[:, 2] -= 0.004
    points_list.append(posed_eyes)
    rays_list.append(posed_eyes)
    colors_list.append(eye_colors)
    luma_list.append(eye_luma)
    kind_list.append(np.full(len(eye_p), SURFACE_FISH))
    is_ray_list.append(np.zeros(len(eye_p), dtype=bool))

    return [np.concatenate(component) for component in (points_list, rays_list, colors_list, luma_list, kind_list, is_ray_list)]

TENTACLE_COUNT: int = 50
NODES_PER_TENTACLE: int = 14
ARMS, NODES = TENTACLE_COUNT, NODES_PER_TENTACLE

_rng_anemone = np.random.default_rng(11)
TENTACLE_LONGITUDES = _rng_anemone.random(TENTACLE_COUNT) * TWO_PI
TENTACLE_LATITUDES = 0.06 + 1.12 * (_rng_anemone.random(TENTACLE_COUNT) ** 1.15)
TENTACLE_LENGTHS = 0.60 + 0.42 * _rng_anemone.random(TENTACLE_COUNT)
_unused_phase = _rng_anemone.random(TENTACLE_COUNT) * TWO_PI
TENTACLE_REST_CURL = _rng_anemone.normal(0.0, 0.42, (TENTACLE_COUNT, 2))

_rng_anemone_b = np.random.default_rng(31)
TENTACLE_BEAT_MODES = _rng_anemone_b.integers(4, 9, (TENTACLE_COUNT, 2))
TENTACLE_INTRINSIC_PHASE = _rng_anemone_b.random((TENTACLE_COUNT, 2)) * TWO_PI
TENTACLE_REST_BEND: float = 2.2
TENTACLE_STIFFNESS_PARAMS = (0.20, 2.2, 0.012)
TENTACLE_ACTIVE_FORCE: float = 0.34
TENTACLE_RADIUS: float = 0.043

ARM_LON, ARM_LAT, ARM_LEN = TENTACLE_LONGITUDES, TENTACLE_LATITUDES, TENTACLE_LENGTHS
ARM_CURL, ARM_BEAT, ARM_PHASE = TENTACLE_REST_CURL, TENTACLE_BEAT_MODES, TENTACLE_INTRINSIC_PHASE
ARM_BEND, ARM_STIFF, ARM_LIFE, ARM_R = (
    TENTACLE_REST_BEND,
    TENTACLE_STIFFNESS_PARAMS,
    TENTACLE_ACTIVE_FORCE,
    TENTACLE_RADIUS,
)

COLLISION_ELLIPSOID_CENTER = np.array([0.08, 0.0, 0.0], dtype=np.float64)
COLLISION_ELLIPSOID_RADII = np.array([0.60, 0.30, 0.17], dtype=np.float64)
FISH_C, FISH_R = COLLISION_ELLIPSOID_CENTER, COLLISION_ELLIPSOID_RADII

def evaluate_ocean_current(pos: np.ndarray, current_time: float) -> np.ndarray:
    surge_x = 0.62 * np.sin(ANGULAR_FREQ_1 * current_time - 1.3 * pos[..., 0]) + 0.16 * np.sin(
        ANGULAR_FREQ_3 * current_time + 2.0 * pos[..., 2]
    )
    surge_y = 0.10 * np.cos(ANGULAR_FREQ_1 * current_time - 1.3 * pos[..., 0])
    surge_z = 0.08 * np.cos(ANGULAR_FREQ_2 * current_time + 3.0 * pos[..., 0])
    return np.stack([surge_x, surge_y, surge_z], axis=-1)

class SeaAnemoneColony:
    def __init__(self, mound_center: Tuple[float, float, float]):
        self.centre = np.asarray(mound_center, dtype=np.float64)
        node_fractions = np.linspace(0.0, 1.0, NODES_PER_TENTACLE)[None, :, None]
        mound_normals = np.stack(
            [
                np.sin(TENTACLE_LATITUDES) * np.cos(TENTACLE_LONGITUDES),
                np.cos(TENTACLE_LATITUDES),
                np.sin(TENTACLE_LATITUDES) * np.sin(TENTACLE_LONGITUDES),
            ],
            axis=-1,
        )
        root_positions = self.centre + mound_normals * ANEMONE_MOUND_DIMENSIONS
        aim_vectors = mound_normals * np.array([1.05, 0.70, 1.0]) + np.array([0.0, 0.42, 0.0])
        aim_vectors[:, 1] = np.maximum(aim_vectors[:, 1], 0.30)
        aim_vectors = normalize_vector(aim_vectors)

        lateral_axes = normalize_vector(np.cross(aim_vectors, np.array([0.0, 0.0, 1.0])))
        longitudinal_axes = np.cross(aim_vectors, lateral_axes)
        lengths = TENTACLE_LENGTHS[:, None, None]

        curl_offsets = (
            lateral_axes[:, None] * TENTACLE_REST_CURL[:, :1, None]
            + longitudinal_axes[:, None] * TENTACLE_REST_CURL[:, 1:, None]
        )
        rest_pose = (
            root_positions[:, None]
            + lengths
            * (
                aim_vectors[:, None] * node_fractions
                + TENTACLE_REST_BEND * curl_offsets * (node_fractions**2 - 0.55 * (node_fractions**3))
            )
        )
        rest_pose[..., 1] += 0.20 * lengths[..., 0] * (node_fractions[..., 0] ** 2)

        self.rest = rest_pose
        self.side = lateral_axes[:, None]
        self.fore = longitudinal_axes[:, None]
        self.u = node_fractions[..., 0]
        self.offset = np.diff(rest_pose, axis=1)
        self.link = np.linalg.norm(self.offset, axis=-1)
        self.stiff = (
            TENTACLE_STIFFNESS_PARAMS[0] * (1.0 - node_fractions[0, 1:, 0]) ** TENTACLE_STIFFNESS_PARAMS[1]
            + TENTACLE_STIFFNESS_PARAMS[2]
        )[None, :, None]
        self.drag = (1.2 + 4.2 * (node_fractions[0, :, 0] ** 1.2))[None, :, None]
        self.back = (root_positions[:, 2] - self.centre[2] + ANEMONE_MOUND_DIMENSIONS[2]) / (
            2.0 * ANEMONE_MOUND_DIMENSIONS[2]
        )

        self.x = rest_pose.copy()
        self.v = np.zeros_like(rest_pose)
        self.n = 0
        self.glow = np.zeros(rest_pose.shape[:2])
        self.lit = np.zeros(len(AMBIENT_MOTES))
        self.off = np.zeros((len(AMBIENT_MOTES), 3))
        self.wake = np.zeros((WAKE_POOL_SIZE, 3))
        self.born = np.full(WAKE_POOL_SIZE, -99.0)
        self.owed = 0.0
        self.dice = np.random.default_rng(7)
        self.now = 0.0

    def _enforce_tentacle_constraints(
        self,
        tentacle_x: np.ndarray,
        fish_pos: np.ndarray,
        fish_yaw: float,
        fish_pitch: float,
        contact_mask: np.ndarray,
    ) -> np.ndarray:
        for _ in range(2):
            for node_idx in range(1, NODES_PER_TENTACLE):
                target_pos = tentacle_x[:, node_idx - 1] + self.offset[:, node_idx - 1]
                tentacle_x[:, node_idx] += self.stiff[0, node_idx - 1] * (target_pos - tentacle_x[:, node_idx])

            fish_local_space = rotate_vector_roll_z(rotate_vector_yaw_y(tentacle_x - fish_pos, -fish_yaw), -fish_pitch)
            normalized_coords = (fish_local_space - COLLISION_ELLIPSOID_CENTER) / (
                COLLISION_ELLIPSOID_RADII + TENTACLE_RADIUS
            )
            scaled_distance = np.linalg.norm(normalized_coords, axis=-1, keepdims=True)
            is_penetrating = scaled_distance < 1.0
            contact_mask |= is_penetrating[..., 0]

            projected_surface = (
                np.where(is_penetrating, normalized_coords / np.maximum(scaled_distance, 1e-6), normalized_coords)
                * (COLLISION_ELLIPSOID_RADII + TENTACLE_RADIUS)
                + COLLISION_ELLIPSOID_CENTER
            )
            world_surface = rotate_vector_yaw_y(rotate_vector_roll_z(projected_surface, fish_pitch), fish_yaw) + fish_pos
            tentacle_x = np.where(is_penetrating, tentacle_x + 0.85 * (world_surface - tentacle_x), tentacle_x)
            tentacle_x[:, 0] = self.rest[:, 0]

            for node_idx in range(1, NODES_PER_TENTACLE):
                relative_vector = tentacle_x[:, node_idx] - tentacle_x[:, node_idx - 1]
                tentacle_x[:, node_idx] = (
                    tentacle_x[:, node_idx - 1] + normalize_vector(relative_vector) * self.link[:, node_idx - 1, None]
                )
        return tentacle_x

    def simulate_step(self) -> None:
        sim_time = (self.n - SIM_WARMUP_STEPS) * SIM_TIME_STEP
        fish_pos, fish_yaw, fish_pitch = evaluate_swimming_pose(sim_time)
        prior_fish_pos = evaluate_swimming_pose(sim_time - SIM_TIME_STEP)[0]

        old_x = self.x.copy()
        tentacle_x = self.x.copy()
        tentacle_v = self.v.copy()
        contact_mask = np.zeros(tentacle_x.shape[:2], dtype=bool)

        flow_velocity = evaluate_ocean_current(tentacle_x, sim_time)
        fish_distance_sq = ((tentacle_x - fish_pos) ** 2).sum(axis=-1)
        fish_proximity_wake = np.exp(-fish_distance_sq / 0.10)
        flow_velocity += 0.9 * fish_proximity_wake[..., None] * (fish_pos - prior_fish_pos) / SIM_TIME_STEP

        active_beat = (
            TENTACLE_BEAT_MODES[:, None, :] * TWO_PI / CYCLE_DURATION * sim_time
            - 5.0 * self.u[..., None]
            + TENTACLE_INTRINSIC_PHASE[:, None, :]
        )
        intrinsic_undulation = (
            np.sin(active_beat[..., :1]) * self.side + 0.7 * np.sin(active_beat[..., 1:]) * self.fore
        )
        flow_velocity += TENTACLE_ACTIVE_FORCE * (self.u**0.7)[..., None] * intrinsic_undulation

        tentacle_v += self.drag * (flow_velocity - tentacle_v) * SIM_TIME_STEP
        tentacle_v[:, 0] = 0.0
        tentacle_x += tentacle_v * SIM_TIME_STEP

        tentacle_x = self._enforce_tentacle_constraints(
            tentacle_x, fish_pos, fish_yaw, fish_pitch, contact_mask
        )

        self.v = (tentacle_x - old_x) / SIM_TIME_STEP * 0.985
        velocity_magnitude = np.linalg.norm(self.v, axis=-1, keepdims=True)
        self.v *= np.minimum(1.0, 1.1 / np.maximum(velocity_magnitude, 1e-6))
        self.v[:, 0] = 0.0
        self.x = tentacle_x

        fish_speed = np.linalg.norm(fish_pos - prior_fish_pos) / SIM_TIME_STEP
        rush_intensity = (
            np.clip((fish_speed - 0.55) * 3.0, 0.0, 1.0)
            * np.exp(-((tentacle_x - fish_pos) ** 2).sum(axis=-1) / 0.16)
        )
        glow_state = self.glow * np.exp(-SIM_TIME_STEP / GLOW_FADE_TIME)
        glow_state[:, 1:] += GLOW_PROPAGATION_SPEED * SIM_TIME_STEP * (glow_state[:, :-1] - glow_state[:, 1:])
        self.glow = np.maximum(glow_state, np.maximum(contact_mask * 1.0, 0.8 * rush_intensity))

        mote_distance_sq = ((AMBIENT_MOTES + self.off - fish_pos) ** 2).sum(axis=-1)
        mote_flash = (
            np.clip((fish_speed - 0.42) * 2.4, 0.0, 1.0) * np.exp(-mote_distance_sq / 0.22)
            + 0.35 * np.exp(-mote_distance_sq / 0.03)
        )
        self.lit = np.maximum(self.lit * np.exp(-SIM_TIME_STEP / MOTE_FADE_TIME), np.clip(mote_flash, 0.0, 1.0))
        self.off = self.off * np.exp(-SIM_TIME_STEP / 2.2) + 0.55 * np.exp(-mote_distance_sq / 0.10)[:, None] * (
            fish_pos - prior_fish_pos
        )

        self.owed += WAKE_EMISSION_RATE * np.clip((fish_speed - 0.50) / 0.30, 0.0, 1.0) * SIM_TIME_STEP
        heading_dir = normalize_vector(fish_pos - prior_fish_pos)
        while self.owed >= 1.0:
            self.owed -= 1.0
            spark_idx = int(self.dice.integers(WAKE_POOL_SIZE))
            self.wake[spark_idx] = (
                fish_pos
                - heading_dir * self.dice.uniform(0.30, 0.85)
                + self.dice.normal(0.0, [0.07, 0.20, 0.16])
            )
            self.born[spark_idx] = sim_time

        self.wake[:, 1] += 0.04 * SIM_TIME_STEP
        self.now = sim_time
        self.n += 1

    def sample_state_at(self, timestamp: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        target_step = int(round(timestamp / SIM_TIME_STEP))
        if target_step + SIM_WARMUP_STEPS < self.n:
            self.x = self.rest.copy()
            self.v = np.zeros_like(self.rest)
            self.n = 0
            self.glow[:] = 0.0
            self.lit[:] = 0.0
            self.off[:] = 0.0
            self.born[:] = -99.0
            self.owed = 0.0
            self.dice = np.random.default_rng(7)

        while self.n < target_step + SIM_WARMUP_STEPS:
            self.simulate_step()

        fine_stations = np.linspace(0.0, 1.0, 34)
        cap_contour = np.sqrt(np.clip(1.0 - ((fine_stations - 0.90) / 0.10) ** 2, 0.0, 1.0))
        tentacle_thickness = np.where(
            fine_stations > 0.90,
            (TENTACLE_RADIUS + 0.006) * cap_contour + 0.004,
            TENTACLE_RADIUS + 0.008 - 0.014 * fine_stations + 0.012 * np.exp(-((fine_stations - 0.86) / 0.08) ** 2),
        )

        spline_indices = np.linspace(0.0, NODES_PER_TENTACLE - 1, 34)
        base_indices = np.clip(spline_indices.astype(int), 0, NODES_PER_TENTACLE - 2)
        spline_fraction = (spline_indices - base_indices)[None, :, None]

        p0, p1 = self.x[:, base_indices], self.x[:, base_indices + 1]
        m0 = (
            self.x[:, np.clip(base_indices + 1, 0, NODES_PER_TENTACLE - 1)]
            - self.x[:, np.clip(base_indices - 1, 0, NODES_PER_TENTACLE - 1)]
        ) / 2.0
        m1 = (
            self.x[:, np.clip(base_indices + 2, 0, NODES_PER_TENTACLE - 1)]
            - self.x[:, base_indices]
        ) / 2.0

        spline_curve = (
            (2.0 * spline_fraction**3 - 3.0 * spline_fraction**2 + 1.0) * p0
            + (spline_fraction**3 - 2.0 * spline_fraction**2 + spline_fraction) * m0
            + (-2.0 * spline_fraction**3 + 3.0 * spline_fraction**2) * p1
            + (spline_fraction**3 - spline_fraction**2) * m1
        )
        interpolated_glow = (
            self.glow[:, base_indices] * (1.0 - spline_fraction[..., 0])
            + self.glow[:, base_indices + 1] * spline_fraction[..., 0]
        )
        depth_attenuation = np.clip(1.1 - 0.5 * self.back, 0.55, 1.0)
        return (
            spline_curve,
            tentacle_thickness,
            depth_attenuation,
            interpolated_glow * (1.0 if EFFECTS["touch"] else 0.0),
        )

LIGHT_BEAM_POSITIONS = np.array([-2.6, -1.35, -0.2, 0.95, 2.1, 3.2], dtype=np.float64)
BEAMS = LIGHT_BEAM_POSITIONS

def render_sunlight_caustics(
    char_matrix: np.ndarray,
    color_matrix: np.ndarray,
    current_time: float,
    screen_scale: float,
    camera_x: float,
    midpoint_y: float,
) -> None:
    rows = char_matrix.shape[0]
    cell_centers_x = (np.arange(GRID_COLUMNS) + 0.5) * CHAR_WIDTH
    cell_centers_y = (np.arange(rows) + 0.5) * CHAR_HEIGHT
    volumetric_light = np.zeros((rows, GRID_COLUMNS), dtype=np.float64)

    for beam_idx, beam_x0 in enumerate(LIGHT_BEAM_POSITIONS):
        beam_anchor = (
            FRAME_WIDTH / 2.0
            + 0.85 * screen_scale * (beam_x0 + 0.22 * np.sin(ANGULAR_FREQ_1 * current_time + 1.7 * beam_idx) - camera_x)
        )
        beam_slant = beam_anchor - 0.30 * (cell_centers_y - midpoint_y)
        beam_width = 60.0 + 22.0 * np.sin(ANGULAR_FREQ_2 * current_time + 2.3 * beam_idx)
        volumetric_light += (0.65 + 0.35 * np.sin(ANGULAR_FREQ_3 * current_time + 1.1 * beam_idx)) * np.exp(
            -((cell_centers_x[None, :] - beam_slant[:, None]) / beam_width) ** 2
        )

    volumetric_light *= np.clip(1.15 - 0.55 * cell_centers_y / (rows * CHAR_HEIGHT), 0.30, 1.0)[:, None]
    color_matrix *= (1.0 + 0.42 * volumetric_light)[..., None]
    np.clip(color_matrix, 0.0, 1.0, out=color_matrix)

    row_indices, col_indices = np.indices(volumetric_light.shape)
    dust_mask = (char_matrix == 0) & (volumetric_light > 0.30) & (((row_indices * 2 + col_indices) % 5) == 0)
    char_matrix[dust_mask] = encode_glyph_index("/")
    color_matrix[dust_mask] = create_rgb_vector(46, 84, 108) * volumetric_light[dust_mask, None]

def rasterize_water_motes(
    char_matrix: np.ndarray,
    color_matrix: np.ndarray,
    depth_buffer: np.ndarray,
    anemone_sim: SeaAnemoneColony,
    current_time: float,
    project_fn: Any,
) -> None:
    ambient_particles = (
        AMBIENT_MOTES + anemone_sim.off + 0.03 * np.sin(ANGULAR_FREQ_2 * current_time + AMBIENT_MOTES[:, ::-1] * 9.0)
    )
    wake_lifespans = (anemone_sim.now - anemone_sim.born) / WAKE_LIFESPAN
    wake_alive = (wake_lifespans >= 0.0) & (wake_lifespans < 1.0)

    all_particles = np.concatenate([ambient_particles, anemone_sim.wake[wake_alive]])
    all_glow = np.concatenate(
        [
            anemone_sim.lit,
            (1.0 - wake_lifespans[wake_alive]) ** 1.4
            * (0.75 + 0.25 * np.sin(40.0 * wake_lifespans[wake_alive] + np.arange(wake_alive.sum()))),
        ]
    )

    screen_x, screen_y, _ = project_fn(all_particles)
    cols = np.floor(screen_x / CHAR_WIDTH).astype(int)
    rows = np.floor(screen_y / CHAR_HEIGHT).astype(int)
    max_rows = char_matrix.shape[0]

    valid = (cols >= 0) & (cols < GRID_COLUMNS) & (rows >= 0) & (rows < max_rows)
    valid[valid] &= depth_buffer[rows[valid], cols[valid]] >= 99.0

    cols, rows, all_glow = cols[valid], rows[valid], all_glow[valid]
    sort_order = np.argsort(all_glow)
    cols, rows, all_glow = cols[sort_order], rows[sort_order], all_glow[sort_order]

    char_matrix[rows, cols] = np.select(
        [all_glow > 0.55, all_glow > 0.18], [encode_glyph_index("*"), encode_glyph_index("+")], encode_glyph_index(".")
    )
    color_matrix[rows, cols] = np.clip(
        MOTE_COLOR * (0.16 + 0.84 * all_glow[:, None]) + 0.5 * np.clip(all_glow[:, None] - 0.5, 0.0, 1.0),
        0.0,
        1.0,
    )

    for delta_c in (-1, 1):
        hot_motes = (all_glow > 0.60) & (cols + delta_c >= 0) & (cols + delta_c < GRID_COLUMNS)
        free_cells = hot_motes.copy()
        free_cells[hot_motes] = char_matrix[rows[hot_motes], cols[hot_motes] + delta_c] == 0
        char_matrix[rows[free_cells], cols[free_cells] + delta_c] = encode_glyph_index("-")
        color_matrix[rows[free_cells], cols[free_cells] + delta_c] = MOTE_COLOR * 0.5 * all_glow[free_cells, None]

def rasterize_prey_particles(
    char_matrix: np.ndarray,
    color_matrix: np.ndarray,
    depth_buffer: np.ndarray,
    state: Dict[str, Any],
    current_time: float,
    project_fn: Any,
) -> None:
    max_rows = char_matrix.shape[0]

    def render_particle_glyph(coords: np.ndarray, char: str, intensity: float) -> Tuple[int, int]:
        sx, sy, _ = project_fn(np.asarray(coords, dtype=np.float64)[None])
        c_idx, r_idx = int(sx[0] // CHAR_WIDTH), int(sy[0] // CHAR_HEIGHT)
        if 0 <= c_idx < GRID_COLUMNS and 0 <= r_idx < max_rows and depth_buffer[r_idx, c_idx] > coords[2] - 0.05:
            char_matrix[r_idx, c_idx] = encode_glyph_index(char)
            color_matrix[r_idx, c_idx] = np.clip(PREY_COLOR * intensity, 0.0, 1.0)
        return c_idx, r_idx

    if state["food"] >= 0:
        base_luminance = np.clip(state["food"] / 0.8, 0.0, 1.0) * (0.85 + 0.15 * np.sin(9.0 * current_time))
        center_c, center_r = render_particle_glyph(state["prey"], "@", 1.3 * base_luminance)
        whisker_flick = int(current_time * 5.0) % 2
        time_since_spotted = (current_time % LOOP) - TRAJECTORY_TRACK["spot"]
        glint_flash = np.clip(1.0 - time_since_spotted / 0.6, 0.0, 1.0) if time_since_spotted >= 0 else 0.0

        body_elements = [
            (-1, 0, "8", 1.0),
            (1, 0, "8", 1.0),
            (-2, 0, "~-"[whisker_flick], 0.7),
            (2, 0, "-~"[whisker_flick], 0.7),
            (0, -1, "o", 0.8),
            (0, 1, "o", 0.6),
        ]
        ray_elements = [
            (-1, -1, "\\", 1.0),
            (1, -1, "/", 1.0),
            (-1, 1, "/", 1.0),
            (1, 1, "\\", 1.0),
            (0, -2, "|", 0.9),
            (0, 2, "|", 0.9),
            (-3, 0, "-", 0.9),
            (3, 0, "-", 0.9),
        ]
        for dc, dr, char_glyph, dim_weight in body_elements + (ray_elements if glint_flash > 0.0 else []):
            shine_val = (
                dim_weight * base_luminance * (1.0 + glint_flash)
                if (dc, dr, char_glyph, dim_weight) in body_elements
                else dim_weight * glint_flash
            )
            if (
                0 <= center_c + dc < GRID_COLUMNS
                and 0 <= center_r + dr < max_rows
                and depth_buffer[center_r + dr, center_c + dc] > state["prey"][2] - 0.05
            ):
                char_matrix[center_r + dr, center_c + dc] = encode_glyph_index(char_glyph)
                color_matrix[center_r + dr, center_c + dc] = np.clip(
                    PREY_COLOR * shine_val + 0.5 * glint_flash, 0.0, 1.0
                )

    time_since_bite = (current_time % LOOP) - TRAJECTORY_TRACK["bite"]
    if 0.0 <= time_since_bite < 0.7:
        for scatter_idx in range(6):
            scatter_vec = np.array(
                [np.cos(2.4 * scatter_idx), np.sin(2.4 * scatter_idx) - 0.3, 0.4 * np.sin(1.3 * scatter_idx)]
            )
            render_particle_glyph(
                TRAJECTORY_TRACK["bite_at"] + scatter_vec * 0.16 * (time_since_bite**0.6),
                ".",
                0.9 * (1.0 - time_since_bite / 0.7),
            )

def rasterize_anemone_tentacles(
    char_matrix: np.ndarray,
    color_matrix: np.ndarray,
    depth_buffer: np.ndarray,
    anemone_geometry: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    project_fn: Any,
) -> None:
    spine_curves, thicknesses, depth_attenuation, lum_glow = anemone_geometry
    max_rows = char_matrix.shape[0]

    fine_sampling = np.linspace(0.0, spine_curves.shape[1] - 1, 90)
    lower_idx = fine_sampling.astype(int).clip(0, spine_curves.shape[1] - 2)
    sample_weights = (fine_sampling - lower_idx)[None, :, None]

    spine_curves = (
        spine_curves[:, lower_idx] * (1.0 - sample_weights) + spine_curves[:, lower_idx + 1] * sample_weights
    )
    lum_glow = (
        lum_glow[:, lower_idx] * (1.0 - sample_weights[..., 0])
        + lum_glow[:, lower_idx + 1] * sample_weights[..., 0]
    )
    thicknesses = np.interp(fine_sampling, np.arange(len(thicknesses)), thicknesses)
    length_fractions = fine_sampling / fine_sampling[-1]

    screen_x, screen_y, shrink_factors = project_fn(spine_curves.reshape(-1, 3))
    screen_x, screen_y, shrink_factors = [
        arr.reshape(spine_curves.shape[:2]) for arr in (screen_x, screen_y, shrink_factors)
    ]
    projected_radii = thicknesses[None] * shrink_factors

    cell_x = (np.arange(GRID_COLUMNS) + 0.5) * CHAR_WIDTH
    cell_y = (np.arange(max_rows) + 0.5) * CHAR_HEIGHT

    for arm_idx in np.argsort(-spine_curves[:, :, 2].mean(axis=1)):
        c_min = max(int((screen_x[arm_idx] - projected_radii[arm_idx]).min() // CHAR_WIDTH), 0)
        c_max = min(int((screen_x[arm_idx] + projected_radii[arm_idx]).max() // CHAR_WIDTH) + 1, GRID_COLUMNS)
        r_min = max(int((screen_y[arm_idx] - projected_radii[arm_idx]).min() // CHAR_HEIGHT), 0)
        r_max = min(int((screen_y[arm_idx] + projected_radii[arm_idx]).max() // CHAR_HEIGHT) + 1, max_rows)

        if c_min >= c_max or r_min >= r_max:
            continue

        offset_x = cell_x[None, c_min:c_max, None] - screen_x[arm_idx][None, None, :]
        offset_y = cell_y[r_min:r_max, None, None] - screen_y[arm_idx][None, None, :]
        distance_ratios = np.sqrt(offset_x * offset_x + offset_y * offset_y) / projected_radii[arm_idx][None, None, :]

        closest_node_idx = distance_ratios.argmin(axis=-1)
        distance_ratios = np.take_along_axis(distance_ratios, closest_node_idx[..., None], axis=-1)[..., 0]
        offset_x = np.take_along_axis(offset_x, closest_node_idx[..., None], axis=-1)[..., 0]
        offset_y = np.take_along_axis(offset_y, closest_node_idx[..., None], axis=-1)[..., 0]

        node_z = spine_curves[arm_idx, closest_node_idx, 2]
        is_hit = (distance_ratios < 1.0) & (node_z < depth_buffer[r_min:r_max, c_min:c_max])
        if not is_hit.any():
            continue

        tip_transition = calculate_cubic_ease((length_fractions[closest_node_idx] - 0.25) / 0.65)
        base_tone = (
            TENTACLE_STEM * (1.0 - tip_transition[..., None]) + TENTACLE_TIP * tip_transition[..., None]
        ) * depth_attenuation[arm_idx]
        diffuse_light = np.clip(
            0.50
            - 0.50
            * (offset_x * 0.80 + offset_y * 0.60)
            / np.maximum(projected_radii[arm_idx][closest_node_idx], 1.0),
            0.0,
            1.0,
        )
        is_rim_edge = distance_ratios > 0.70
        is_broken_edge = np.sin(closest_node_idx * 0.90 + arm_idx * 2.1) > -0.86
        biolum_shine = (
            lum_glow[arm_idx][closest_node_idx]
            * (0.65 + 0.35 * depth_attenuation[arm_idx])
            * (0.35 + 0.65 * tip_transition)
        )
        glyph_value = 0.20 + 0.30 * diffuse_light + 0.40 * (tip_transition**1.5) + 0.30 * biolum_shine

        tentacle_glyph = np.where(
            is_rim_edge,
            determine_line_glyph(-offset_y, offset_x),
            TENTACLE_FILL_GLYPHS[
                np.clip(
                    (glyph_value * 1.25 * len(TENTACLE_FILL_GLYPHS)).astype(int), 0, len(TENTACLE_FILL_GLYPHS) - 1
                )
            ],
        )
        color_val = np.where(
            is_rim_edge[..., None],
            base_tone
            * (0.65 + 0.35 * diffuse_light[..., None] + 0.25 * tip_transition[..., None])
            * np.where(is_broken_edge, 1.0, 0.65)[..., None],
            base_tone * (0.45 + 0.35 * diffuse_light[..., None] + 0.35 * (tip_transition[..., None] ** 1.5)),
        )
        color_val = np.clip(
            color_val + GLOW_COLOR * biolum_shine[..., None] * np.where(is_rim_edge, 0.75, 0.55)[..., None],
            0.0,
            1.0,
        )

        sub_grid = (slice(r_min, r_max), slice(c_min, c_max))
        char_matrix[sub_grid][is_hit] = tentacle_glyph[is_hit]
        color_matrix[sub_grid][is_hit] = color_val[is_hit]
        depth_buffer[sub_grid][is_hit] = node_z[is_hit]

_rng_water = np.random.default_rng(5)
WATER_SPECKS = _rng_water.random((150, 3))
WATER_MOTES = _rng_water.random((46, 4))
DOT, MOTE = WATER_SPECKS, WATER_MOTES

def render_background_water(
    current_time: float,
    char_matrix: np.ndarray,
    color_matrix: np.ndarray,
    rows: int,
    deep_factor: float = 1.0,
    camera_slide: Tuple[float, float] = (0.0, 0.0),
) -> None:
    r_coords = (
        (WATER_SPECKS[:, 1] * rows + 0.18 * current_time * (0.5 + WATER_SPECKS[:, 2]) + camera_slide[1] * 0.3) % rows
    ).astype(int)
    c_coords = (
        (
            WATER_SPECKS[:, 0] * GRID_COLUMNS
            + 0.25 * np.sin(0.4 * current_time + WATER_SPECKS[:, 2] * 9.0)
            - camera_slide[0] * 0.3
        )
        % GRID_COLUMNS
    ).astype(int)
    char_matrix[r_coords, c_coords] = encode_glyph_index(".")
    color_matrix[r_coords, c_coords] = create_rgb_vector(64, 84, 112) * (0.5 + 0.5 * WATER_SPECKS[:, 2:3]) * deep_factor

    r_coords = (
        (WATER_MOTES[:, 1] * rows - 0.3 * current_time * (0.4 + WATER_MOTES[:, 2]) + camera_slide[1] * 0.55) % rows
    ).astype(int)
    c_coords = (
        (
            WATER_MOTES[:, 0] * GRID_COLUMNS
            + 0.8 * np.sin(0.5 * current_time + WATER_MOTES[:, 3] * 20.0)
            - camera_slide[0] * 0.55
        )
        % GRID_COLUMNS
    ).astype(int)
    twinkle = np.sin(current_time * (1.0 + 2.0 * WATER_MOTES[:, 2]) + WATER_MOTES[:, 3] * 40.0)
    char_matrix[r_coords, c_coords] = np.where(twinkle > 0.7, encode_glyph_index("*"), encode_glyph_index("+"))
    color_matrix[r_coords, c_coords] = (
        create_rgb_vector(70, 118, 160) * (0.5 + 0.5 * np.clip(twinkle + 0.4, 0.0, 1.0))[:, None] * deep_factor
    )

def rasterize_vector_stroke(
    char_matrix: np.ndarray,
    color_matrix: np.ndarray,
    owner_matrix: np.ndarray,
    pixel_polyline: List[Tuple[float, float]],
    stroke_color: np.ndarray,
    rows: int,
    custom_glyph: Optional[str] = None,
) -> None:
    coords = np.asarray(pixel_polyline, dtype=np.float64)
    for (x0, y0), (x1, y1) in zip(coords[:-1], coords[1:]):
        steps = int(max(abs(x1 - x0) / CHAR_WIDTH, abs(y1 - y0) / CHAR_HEIGHT) * 1.6) + 2
        fractions = np.linspace(0.0, 1.0, steps)
        col_indices = ((x0 + (x1 - x0) * fractions) // CHAR_WIDTH).astype(int)
        row_indices = ((y0 + (y1 - y0) * fractions) // CHAR_HEIGHT).astype(int)
        valid = (col_indices >= 0) & (col_indices < GRID_COLUMNS) & (row_indices >= 0) & (row_indices < rows)

        char_matrix[row_indices[valid], col_indices[valid]] = (
            determine_line_glyph(x1 - x0, y1 - y0) if custom_glyph is None else encode_glyph_index(custom_glyph)
        )
        color_matrix[row_indices[valid], col_indices[valid]] = stroke_color
        owner_matrix[row_indices[valid], col_indices[valid]] = SURFACE_BACKGROUND

CORAL_STRUCTURE_CACHE: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
_corals = CORAL_STRUCTURE_CACHE

def generate_coral_branch_profile(random_seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if random_seed not in CORAL_STRUCTURE_CACHE:
        rng = np.random.default_rng(random_seed)
        twigs: List[Tuple[float, float, float, float, int]] = []
        buds: List[Tuple[float, float, bool]] = []

        def propagate_coral_branch(x: float, y: float, angle: float, branch_len: float, depth: int) -> None:
            if depth == 0:
                buds.append((x, y, rng.random() < 0.6))
                return
            x_end, y_end = x + branch_len * np.sin(angle), y + branch_len * np.cos(angle)
            twigs.append((x, y, x_end, y_end, depth))
            spread_angle = 0.35 + 0.30 * rng.random()
            propagate_coral_branch(x_end, y_end, angle - spread_angle, branch_len * (0.62 + 0.20 * rng.random()), depth - 1)
            propagate_coral_branch(
                x_end, y_end, angle + spread_angle * (0.70 + 0.50 * rng.random()), branch_len * (0.62 + 0.20 * rng.random()), depth - 1
            )
            if rng.random() < 0.3:
                propagate_coral_branch(x_end, y_end, angle + 0.1 * rng.normal(), branch_len * 0.60, depth - 1)

        propagate_coral_branch(0.0, 0.0, 0.0, 1.0, 5)
        CORAL_STRUCTURE_CACHE[random_seed] = (np.array(twigs), np.array(buds))
    return CORAL_STRUCTURE_CACHE[random_seed]

def generate_coral_formations(
    char_matrix: np.ndarray,
    color_matrix: np.ndarray,
    owner_matrix: np.ndarray,
    rows: int,
    screen_x: float,
    screen_y: float,
    display_size: float,
    coral_color: np.ndarray,
    random_seed: int,
    current_time: float = 0.0,
    bend_amount: float = 0.0,
) -> None:
    twigs, buds = generate_coral_branch_profile(random_seed)
    surge_lean = lambda height: bend_amount * 0.06 * np.sin(
        ANGULAR_FREQ_1 * current_time + 0.7 * random_seed
    ) * (np.clip(height, 0.0, None) ** 1.5)

    for x0, y0, x1, y1, level in twigs:
        rasterize_vector_stroke(
            char_matrix,
            color_matrix,
            owner_matrix,
            [
                (screen_x + display_size * (x0 + surge_lean(y0)), screen_y - display_size * y0),
                (screen_x + display_size * (x1 + surge_lean(y1)), screen_y - display_size * y1),
            ],
            coral_color * (0.55 + 0.10 * (5 - level)),
            rows,
        )

    for bx, by, is_round in buds:
        c_idx = int((screen_x + display_size * (bx + surge_lean(by))) // CHAR_WIDTH)
        r_idx = int((screen_y - display_size * by) // CHAR_HEIGHT)
        if 0 <= c_idx < GRID_COLUMNS and 0 <= r_idx < rows:
            char_matrix[r_idx, c_idx] = encode_glyph_index("o" if is_round else "*")
            color_matrix[r_idx, c_idx] = np.minimum(coral_color * 1.5, 1.0)
            owner_matrix[r_idx, c_idx] = SURFACE_BACKGROUND

GEOLOGICAL_ROCKS = [
    (0.05, 0.42, 1.50, 0.44, 1.10, 0.5),
    (2.25, 0.80, 0.95, 0.80, 0.85, 1.0),
    (-3.3, 1.5, 0.8, 0.5, 0.7, 2.0),
    (1.2, 2.9, 1.4, 0.9, 0.8, 3.0),
    (-1.2, 3.3, 1.1, 0.65, 0.7, 4.0),
    (4.6, 1.9, 0.9, 0.6, 0.7, 5.0),
    (-4.9, 0.4, 0.7, 0.45, 0.6, 6.0),
]
ROCKS = GEOLOGICAL_ROCKS

REEF_CORALS = [
    (-2.25, SEABED_ELEVATION, 0.85, 0.34, create_rgb_vector(236, 96, 132), 3),
    (-2.85, SEABED_ELEVATION, 0.45, 0.24, create_rgb_vector(240, 150, 90), 8),
    (2.60, SEABED_ELEVATION + 0.72, 1.25, 0.27, create_rgb_vector(80, 200, 200), 5),
    (1.0, SEABED_ELEVATION, 2.3, 0.30, create_rgb_vector(180, 110, 220), 12),
    (-0.9, SEABED_ELEVATION + 0.5, 3.4, 0.30, create_rgb_vector(236, 96, 132), 21),
    (4.3, SEABED_ELEVATION, 0.9, 0.25, create_rgb_vector(240, 150, 90), 30),
]
CORALS = REEF_CORALS

def generate_seabed_terrain(
    char_matrix: np.ndarray,
    color_matrix: np.ndarray,
    rows: int,
    project_fn: Any,
    camera_x: float,
    current_time: float = 0.0,
) -> None:
    x_indices, z_indices = np.meshgrid(
        np.arange(-34, 35) + int(camera_x / 0.13),
        np.arange(-12, 42),
        indexing="ij",
    )
    x_flat, z_flat = x_indices.ravel(), z_indices.ravel()
    noise_hash = generate_hash_float(x_flat, z_flat)
    keep_mask = noise_hash > 0.52

    pos_x = x_flat * 0.13 + 0.09 * generate_hash_float(z_flat, x_flat)
    pos_z = z_flat * 0.13 + 0.09 * noise_hash
    ripple_contour = np.sin(pos_x * 4.2 + 1.6 * np.sin(pos_z * 2.1)) > 0.72
    keep_mask |= ripple_contour & (generate_hash_float(x_flat + 7, z_flat) > 0.35)

    sand_points = np.stack([pos_x, SEABED_ELEVATION + 0.03 * np.sin(pos_x * 1.7 + pos_z), pos_z], axis=-1)[keep_mask]
    screen_x, screen_y, _ = project_fn(sand_points)

    cols = np.floor(screen_x / CHAR_WIDTH).astype(int)
    row_idx = np.floor(screen_y / CHAR_HEIGHT).astype(int)
    valid = (cols >= 0) & (cols < GRID_COLUMNS) & (row_idx >= 0) & (row_idx < rows)

    depth_fade = np.clip(1.0 - (sand_points[:, 2] + 1.5) / 6.5, 0.40, 1.0)
    if EFFECTS["shafts"]:
        caustic_wave = np.sin(
            sand_points[:, 0] * 5.0 + 1.3 * np.sin(sand_points[:, 2] * 3.0 + ANGULAR_FREQ_3 * current_time)
        ) * np.sin(sand_points[:, 2] * 4.0 - ANGULAR_FREQ_2 * current_time + sand_points[:, 0])
        depth_fade = depth_fade * (1.0 + 1.1 * (np.clip(caustic_wave, 0.0, 1.0) ** 2))

    char_matrix[row_idx[valid], cols[valid]] = np.where(ripple_contour[keep_mask], encode_glyph_index("~"), encode_glyph_index("."))[valid]
    color_matrix[row_idx[valid], cols[valid]] = (
        create_rgb_vector(210, 180, 130)
        * np.where(ripple_contour[keep_mask], 0.75, 0.55)[:, None]
        * depth_fade[:, None]
    )[valid]

def generate_seabed_boulders() -> List[np.ndarray]:
    rock_pieces: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for cx, cz, rx, ry, rz, seed in GEOLOGICAL_ROCKS:
        theta_grid, phi_grid = [
            grid.ravel()
            for grid in np.meshgrid(
                np.linspace(0.02, 1.57, 46),
                np.linspace(0.0, TWO_PI, 110, endpoint=False),
                indexing="ij",
            )
        ]
        surface_bumps = (
            1.0
            + 0.10 * np.sin(5.0 * phi_grid + seed) * np.sin(3.0 * theta_grid + seed)
            + 0.06 * np.sin(11.0 * phi_grid + 2.0 * seed)
        )
        normals = np.stack(
            [np.sin(theta_grid) * np.cos(phi_grid), np.cos(theta_grid), np.sin(theta_grid) * np.sin(phi_grid)],
            axis=-1,
        )
        points = np.array([cx, SEABED_ELEVATION, cz]) + normals * surface_bumps[:, None] * np.array([rx, ry, rz])

        diffuse_lit = np.clip(normalize_vector(normals / np.array([rx, ry, rz])).dot(LIGHT_DIRECTION), 0.0, 1.0)
        distance_falloff = np.clip(1.0 - (cz + 1.0) / 6.0, 0.50, 1.0)
        rock_color = create_rgb_vector(125, 140, 175) * (0.35 + 0.55 * diffuse_lit[:, None]) * distance_falloff
        rock_luma = 0.15 + 0.75 * (diffuse_lit**1.4)
        rock_pieces.append((points, rock_color, rock_luma))

    points_all, colors_all, luma_all = [np.concatenate([piece[i] for piece in rock_pieces]) for i in range(3)]
    return [
        points_all,
        points_all,
        colors_all,
        luma_all,
        np.full(len(points_all), SURFACE_BACKGROUND),
        np.zeros(len(points_all), dtype=bool),
    ]

def generate_anemone_pedal_disc() -> List[np.ndarray]:
    theta_grid, phi_grid = [
        grid.ravel()
        for grid in np.meshgrid(
            np.linspace(0.02, 1.57, 40),
            np.linspace(0.0, TWO_PI, 120, endpoint=False),
            indexing="ij",
        )
    ]
    normals = np.stack(
        [np.sin(theta_grid) * np.cos(phi_grid), np.cos(theta_grid), np.sin(theta_grid) * np.sin(phi_grid)],
        axis=-1,
    )
    points = np.array(ANEMONE_ROOT_POSITION) + normals * ANEMONE_MOUND_DIMENSIONS - np.array([0.0, 0.06, 0.0])
    diffuse_lit = np.clip(normalize_vector(normals / ANEMONE_MOUND_DIMENSIONS).dot(LIGHT_DIRECTION), 0.0, 1.0)
    base_color = TENTACLE_STEM * (0.40 + 0.45 * diffuse_lit[:, None])
    luminosity = 0.25 + 0.45 * diffuse_lit
    return [
        points,
        points,
        base_color,
        luminosity,
        np.full(len(points), SURFACE_BACKGROUND),
        np.zeros(len(points), dtype=bool),
    ]

ROCK_POINTS = [np.concatenate([r, f]) for r, f in zip(generate_seabed_boulders(), generate_anemone_pedal_disc())]

FONT_CACHE: Dict[int, ImageFont.FreeTypeFont] = {}
_font = FONT_CACHE

def load_default_font(size_pt: int) -> ImageFont.FreeTypeFont:
    if size_pt not in FONT_CACHE:
        custom_font_path = os.environ.get("ASCII_LAB_FONT")
        if custom_font_path and os.path.isfile(custom_font_path):
            FONT_CACHE[size_pt] = ImageFont.truetype(custom_font_path, size_pt)
        else:
            for candidate in SYSTEM_FONT_CANDIDATES:
                try:
                    FONT_CACHE[size_pt] = ImageFont.truetype(candidate, size_pt)
                    break
                except OSError:
                    continue
            else:
                FONT_CACHE[size_pt] = ImageFont.load_default()
    return FONT_CACHE[size_pt]

def rasterize_glyph_atlas() -> np.ndarray:
    atlas = np.zeros((96, CHAR_HEIGHT, CHAR_WIDTH), dtype=np.float32)
    glyph_font = load_default_font(15)
    for glyph_code in range(95):
        char = chr(32 + glyph_code)
        char_image = Image.new("L", (CHAR_WIDTH, CHAR_HEIGHT), 0)
        draw = ImageDraw.Draw(char_image)
        bbox = glyph_font.getbbox(char)
        if bbox is not None:
            gw = bbox[2] - bbox[0]
            gh = bbox[3] - bbox[1]
            ox = max(0, (CHAR_WIDTH - gw) // 2 - bbox[0])
            oy = max(0, (CHAR_HEIGHT - gh) // 2 - bbox[1])
            if char == "_":
                oy = CHAR_HEIGHT - gh - 1 - bbox[1]
            draw.text((ox, oy), char, font=glyph_font, fill=255)
        atlas[glyph_code] = np.asarray(char_image, dtype=np.float32) / 255.0
    return atlas

SCENE_LAYOUT_CONFIG = {
    "scale": (660, 740),
    "weed": ANEMONE_ROOT_POSITION,
    "mid": (640, 960),
}
LAYOUT = SCENE_LAYOUT_CONFIG

def _enhance_depth_contours(
    char_matrix: np.ndarray,
    color_matrix: np.ndarray,
    owner_matrix: np.ndarray,
    rows: int,
    current_time: float,
) -> None:
    padded_fish = np.pad(owner_matrix == SURFACE_FISH, 1)
    boundary_up = ~padded_fish[:-2, 1:-1]
    boundary_down = ~padded_fish[2:, 1:-1]
    boundary_left = ~padded_fish[1:-1, :-2]
    boundary_right = ~padded_fish[1:-1, 2:]
    fish_cells = owner_matrix == SURFACE_FISH

    contour_edges = np.full((rows, GRID_COLUMNS), -1, dtype=np.int32)
    contour_edges[fish_cells & boundary_up & (owner_matrix != SURFACE_FIN)] = encode_glyph_index("-")
    contour_edges[fish_cells & boundary_down] = encode_glyph_index("-")
    contour_edges[fish_cells & boundary_left] = encode_glyph_index("(")
    contour_edges[fish_cells & boundary_left & boundary_up] = encode_glyph_index("/")
    contour_edges[fish_cells & boundary_left & boundary_down] = encode_glyph_index("\\")
    contour_edges[fish_cells & boundary_right & boundary_up] = encode_glyph_index("\\")
    contour_edges[fish_cells & boundary_right & boundary_down] = encode_glyph_index("/")

    padded_fin = np.pad(owner_matrix == SURFACE_FIN, 1)
    near_fin = padded_fin[:-2, 1:-1] | padded_fin[2:, 1:-1] | padded_fin[1:-1, :-2] | padded_fin[1:-1, 2:]
    broken_outline_mask = (contour_edges >= 0) & ~near_fin
    char_matrix[broken_outline_mask] = contour_edges[broken_outline_mask]
    color_matrix[broken_outline_mask] = np.clip(color_matrix[broken_outline_mask] * 0.85 + 0.15, 0.0, 1.0)

def render_depth_and_color_buffers(current_time: float, rows: int) -> Tuple[np.ndarray, np.ndarray]:
    layout = SCENE_LAYOUT_CONFIG
    mode_index = 0 if rows == GRID_ROWS_CODE else 1
    screen_scale, midpoint_y = layout["scale"][mode_index], layout["mid"][mode_index]

    char_matrix = np.zeros((rows, GRID_COLUMNS), dtype=np.int32)
    color_matrix = np.zeros((rows, GRID_COLUMNS, 3), dtype=np.float64)
    owner_matrix = np.zeros((rows, GRID_COLUMNS), dtype=np.int32)

    st = sample_kinematic_trajectory(current_time)
    camera_x = st["cam_x"]
    camera_y = 0.62 * st["cam_y"] - 0.10
    camera_lead = 0.11 * st["lead"]
    screen_scale *= st["zoom"]

    orbit_swing, orbit_tilt = 0.0, 0.0
    aim_target = np.zeros(3, dtype=np.float64)
    if EFFECTS["orbit"]:
        orbit_swing = CAMERA_ORBIT_RATE * (camera_x - SCENE_PIVOT_CENTER[0])
        orbit_tilt = -CAMERA_TILT_RATE * (camera_y - 0.15)
        aim_target = np.array([camera_x + camera_lead, camera_y, st["cam_z"]])

    def convert_world_to_camera_view(points: np.ndarray) -> np.ndarray:
        if orbit_swing == 0.0 and orbit_tilt == 0.0:
            return points
        return rotate_vector_pitch_x(rotate_vector_yaw_y(points - SCENE_PIVOT_CENTER, orbit_swing), orbit_tilt) + SCENE_PIVOT_CENTER

    center_point = (
        convert_world_to_camera_view(aim_target[None])[0]
        if (orbit_swing or orbit_tilt)
        else np.array([camera_x + camera_lead, camera_y, 0.0])
    )
    fish_depth_near = convert_world_to_camera_view(np.array([[camera_x, camera_y, st["cam_z"]]]))[0, 2]
    screen_scale *= ((CAMERA_DISTANCE + fish_depth_near) / CAMERA_DISTANCE) ** 0.85

    def project_camera_to_screen_grid(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        shrinkage = screen_scale * CAMERA_DISTANCE / (CAMERA_DISTANCE + points[:, 2])
        proj_x = FRAME_WIDTH / 2.0 + shrinkage * (points[:, 0] - center_point[0])
        proj_y = midpoint_y - shrinkage * (points[:, 1] - center_point[1])
        return proj_x, proj_y, shrinkage

    render_background_water(
        current_time,
        char_matrix,
        color_matrix,
        rows,
        camera_slide=(camera_x * screen_scale / CHAR_WIDTH, camera_y * screen_scale / CHAR_HEIGHT),
    )
    fish_geometry = assemble_fish_model(current_time)
    generate_seabed_terrain(char_matrix, color_matrix, rows, lambda v: project_camera_to_screen_grid(convert_world_to_camera_view(v)), camera_x, current_time)

    for cx, cy, cz, csize, ccolor, cseed in REEF_CORALS:
        coral_bx, coral_by, coral_shrink = project_camera_to_screen_grid(convert_world_to_camera_view(np.array([[cx, cy, cz]])))
        if -300 < coral_bx[0] < FRAME_WIDTH + 300:
            generate_coral_formations(
                char_matrix,
                color_matrix,
                owner_matrix,
                rows,
                coral_bx[0],
                coral_by[0],
                csize * coral_shrink[0],
                ccolor * np.clip(0.75 - 0.12 * cz, 0.3, 0.7),
                cseed,
                current_time,
                bend_amount=1.0,
            )

    combined_solids = [np.concatenate([a, b]) for a, b in zip(fish_geometry, ROCK_POINTS)]
    p_pts, q_pts, col_pts, lum_pts, kind_pts, is_ray = combined_solids
    p_view, q_view = convert_world_to_camera_view(p_pts), convert_world_to_camera_view(q_pts)

    screen_px, screen_py, _ = project_camera_to_screen_grid(p_view)
    screen_qx, screen_qy, _ = project_camera_to_screen_grid(q_view)

    surface_glyphs = np.where(
        is_ray,
        determine_line_glyph(screen_qx - screen_px, screen_qy - screen_py),
        GLYPH_RAMP_INDICES[
            np.clip(((1.0 - lum_pts) ** 1.3 * (len(GLYPH_RAMP) - 1)).astype(int), 0, len(GLYPH_RAMP) - 1)
        ],
    )
    surface_glyphs = np.where(
        kind_pts == SURFACE_BACKGROUND,
        ROCK_GLYPHS[np.clip((lum_pts * len(ROCK_GLYPHS)).astype(int), 0, len(ROCK_GLYPHS) - 1)],
        surface_glyphs,
    )

    c_coords = np.floor(screen_px / CHAR_WIDTH).astype(int)
    r_coords = np.floor(screen_py / CHAR_HEIGHT).astype(int)
    in_bounds = (c_coords >= 0) & (c_coords < GRID_COLUMNS) & (r_coords >= 0) & (r_coords < rows)

    depth_order = np.argsort(-p_view[in_bounds, 2], kind="stable")
    c_sorted = c_coords[in_bounds][depth_order]
    r_sorted = r_coords[in_bounds][depth_order]
    g_sorted = surface_glyphs[in_bounds][depth_order]
    col_sorted = col_pts[in_bounds][depth_order]
    kind_sorted = kind_pts[in_bounds][depth_order]
    z_sorted = p_view[in_bounds, 2][depth_order]

    depth_buffer = np.full((rows, GRID_COLUMNS), 99.0, dtype=np.float64)
    char_matrix[r_sorted, c_sorted] = g_sorted
    color_matrix[r_sorted, c_sorted] = col_sorted
    owner_matrix[r_sorted, c_sorted] = kind_sorted
    depth_buffer[r_sorted, c_sorted] = z_sorted

    _enhance_depth_contours(char_matrix, color_matrix, owner_matrix, rows, current_time)

    if "reef" not in layout:
        layout["reef"] = SeaAnemoneColony(layout["weed"])
    tentacles = layout["reef"].sample_state_at(current_time)
    rasterize_anemone_tentacles(char_matrix, color_matrix, depth_buffer, (convert_world_to_camera_view(tentacles[0]),) + tentacles[1:], project_camera_to_screen_grid)

    if EFFECTS["plankton"]:
        rasterize_water_motes(
            char_matrix,
            color_matrix,
            depth_buffer,
            layout["reef"],
            current_time,
            lambda v: project_camera_to_screen_grid(convert_world_to_camera_view(v)),
        )
    if EFFECTS["shafts"]:
        render_sunlight_caustics(
            char_matrix,
            color_matrix,
            current_time,
            screen_scale,
            camera_x + camera_lead + 1.6 * orbit_swing,
            midpoint_y,
        )
    if EFFECTS["prey"]:
        rasterize_prey_particles(
            char_matrix,
            color_matrix,
            depth_buffer,
            st,
            current_time,
            lambda v: project_camera_to_screen_grid(convert_world_to_camera_view(v)),
        )

    return char_matrix, color_matrix

GLOW_WEIGHT_NEAR: float = 0.6
GLOW_WEIGHT_FAR: float = 0.5
GLOW_WEIGHT_RAYS: float = 0.10
GLOW_STREAK_RADIUS: int = 90
GLOW_NEAR, GLOW_FAR, GLOW_RAYS, STREAK = (
    GLOW_WEIGHT_NEAR,
    GLOW_WEIGHT_FAR,
    GLOW_WEIGHT_RAYS,
    GLOW_STREAK_RADIUS,
)

GLOBAL_ATLAS: Optional[np.ndarray] = None
ATLAS = GLOBAL_ATLAS

def render_colored_ascii_image(char_matrix: np.ndarray, color_matrix: np.ndarray) -> np.ndarray:
    global GLOBAL_ATLAS
    if GLOBAL_ATLAS is None:
        GLOBAL_ATLAS = rasterize_glyph_atlas()

    rows = char_matrix.shape[0]
    canvas_height = rows * CHAR_HEIGHT
    glyph_mask = GLOBAL_ATLAS[char_matrix].transpose(0, 2, 1, 3).reshape(canvas_height, FRAME_WIDTH)
    pixel_colors = np.repeat(np.repeat(color_matrix.astype(np.float32), CHAR_HEIGHT, 0), CHAR_WIDTH, 1)
    base_image = glyph_mask[..., None] * pixel_colors

    small_preview = Image.fromarray((np.clip(base_image, 0.0, 1.0) * 255).astype(np.uint8)).resize(
        (FRAME_WIDTH // 4, canvas_height // 4), Image.BOX
    )
    near_glow = np.asarray(
        small_preview.filter(ImageFilter.GaussianBlur(2.2)).resize((FRAME_WIDTH, canvas_height), Image.BILINEAR),
        dtype=np.float32,
    )
    far_glow = np.asarray(
        small_preview.filter(ImageFilter.GaussianBlur(9.0)).resize((FRAME_WIDTH, canvas_height), Image.BILINEAR),
        dtype=np.float32,
    )

    padded_image = np.pad(base_image, ((GLOW_STREAK_RADIUS, GLOW_STREAK_RADIUS), (0, 0), (0, 0)))
    cumulative_streak = np.cumsum(padded_image, axis=0)
    vertical_rays = (
        cumulative_streak[2 * GLOW_STREAK_RADIUS :] - cumulative_streak[: -2 * GLOW_STREAK_RADIUS]
    ) / (2.0 * GLOW_STREAK_RADIUS)

    composite = (
        base_image * 255.0
        + GLOW_WEIGHT_NEAR * near_glow
        + GLOW_WEIGHT_FAR * far_glow
        + GLOW_WEIGHT_RAYS * 255.0 * vertical_rays
    )
    return np.clip(composite, 0.0, 255.0).astype(np.uint8)

try:
    with open(__file__, encoding="utf-8") as source_fh:
        SOURCE_CODE_LINES = source_fh.read().splitlines()
except Exception:
    SOURCE_CODE_LINES = [
        "Prompt Procedural Simulator",
        "def execute_cli_entrypoint(): pass",
    ]

SOURCE = SOURCE_CODE_LINES
CODE_INSPECTION_SECTIONS = [
    "def generate_fish_body_mesh(",
    "def generate_fish_fins_mesh(",
    "class SeaAnemoneColony:",
    "def generate_scene_image_frame(",
]
SECTIONS = CODE_INSPECTION_SECTIONS

def generate_composite_preview_panel(current_time: float, rows: int) -> Image.Image:
    panel_height = FRAME_HEIGHT - rows * CHAR_HEIGHT
    panel_image = Image.new("RGB", (FRAME_WIDTH, panel_height), (8, 10, 14))
    draw_ctx = ImageDraw.Draw(panel_image)
    draw_ctx.line([(0, 0), (FRAME_WIDTH, 0)], fill=(40, 48, 60))

    target_section_idx = sample_kinematic_trajectory(current_time)["section"] % len(CODE_INSPECTION_SECTIONS)
    marker_str = CODE_INSPECTION_SECTIONS[target_section_idx]
    start_line_idx = next((idx for idx, line in enumerate(SOURCE_CODE_LINES) if marker_str in line), 0)

    for line_offset in range(panel_height // 30):
        src_line_num = start_line_idx + line_offset
        code_text = SOURCE_CODE_LINES[src_line_num][:74] if src_line_num < len(SOURCE_CODE_LINES) else ""
        draw_ctx.text((14, 14 + 30 * line_offset), f"{src_line_num + 1:4d}", font=load_default_font(21), fill=(70, 80, 96))
        text_color = (110, 150, 120) if code_text.strip().startswith("#") else (176, 186, 204)
        draw_ctx.text(
            (88, 14 + 30 * line_offset),
            code_text,
            font=load_default_font(21),
            fill=(150, 220, 170) if line_offset == 0 else text_color,
        )
    return panel_image

def generate_scene_image_frame(current_time: float, code: bool = False) -> Image.Image:
    rows = GRID_ROWS_CODE if code else GRID_ROWS_FULL
    char_matrix, color_matrix = render_depth_and_color_buffers(current_time, rows)

    output_image = Image.new("RGB", (FRAME_WIDTH, FRAME_HEIGHT), (0, 0, 0))
    rendered_raster = Image.fromarray(render_colored_ascii_image(char_matrix, color_matrix))
    output_image.paste(rendered_raster, (0, 0))

    draw_ctx = ImageDraw.Draw(output_image)
    st = sample_kinematic_trajectory(current_time)
    diagnostics_label = f"tail {st['hz']:.1f} Hz   speed {st['speed']:.2f}   chars {int((char_matrix > 0).sum())}"
    draw_ctx.text((96, 130), diagnostics_label, font=load_default_font(20), fill=(120, 132, 150))
    draw_ctx.text(
        (FRAME_WIDTH - 96 - draw_ctx.textlength(HANDLE, font=load_default_font(20)), 130),
        HANDLE,
        font=load_default_font(20),
        fill=(120, 132, 150),
    )

    if code:
        output_image.paste(generate_composite_preview_panel(current_time, rows), (0, rows * CHAR_HEIGHT))
    return output_image

def format_byte_size(bytes_total: float) -> str:
    scales = ("B", "KB", "MB", "GB", "TB")
    val = float(max(0.0, bytes_total))
    for unit in scales[:-1]:
        if val < 1024.0:
            return f"{val:.1f} {unit}"
        val /= 1024.0
    return f"{val:.1f} {scales[-1]}"

def format_duration_string(seconds: float) -> str:
    if not (isinstance(seconds, (int, float)) and np.isfinite(seconds) and seconds >= 0):
        return "--:--"
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"

def _find_ffmpeg_executable() -> str:
    candidate = shutil.which("ffmpeg")
    if not candidate:
        try:
            import imageio_ffmpeg
            candidate = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            candidate = None

    if not candidate:
        raise FileNotFoundError(
            "FFmpeg executable not found in PATH.\n"
            "FFmpeg is required to encode MP4 videos.\n\n"
            "To install FFmpeg:\n"
            "  • Ubuntu / Debian:      sudo apt update && sudo apt install -y ffmpeg\n"
            "  • macOS (Homebrew):     brew install ffmpeg\n"
            "  • Conda (macOS/Linux):  conda install -c conda-forge ffmpeg\n"
            "  • Windows:              winget install Gyan.FFmpeg"
        )
    return candidate

def _construct_progress_bar(current_step: int, total_steps: int, width: int = 24) -> str:
    filled_cells = int(width * current_step // total_steps) if total_steps > 0 else 0
    bar_body = "=" * filled_cells
    if filled_cells < width:
        bar_body += ">" + " " * (width - filled_cells - 1)
    return bar_body

def encode_mp4_video_sequence(path: str, code: bool = False, dur: Optional[float] = None, quiet: bool = False) -> None:
    if dur is None:
        if isinstance(code, (int, float)):
            dur = float(code)
            code = False
        else:
            dur = CYCLE_DURATION

    ffmpeg_bin = _find_ffmpeg_executable()
    total_frames = int(dur * DEFAULT_FPS)

    if not quiet:
        print(f"Encoding MP4 video to '{path}':")
        print(f"  • Resolution:  {FRAME_WIDTH}x{FRAME_HEIGHT}")
        print(f"  • Duration:    {dur:g}s ({total_frames} frames @ {DEFAULT_FPS} fps)")
        print(f"  • Code panel:  {'visible' if code else 'hidden'}")
        print("Starting frame render...")

    ffmpeg_cmd = [
        ffmpeg_bin,
        "-v", "error",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{FRAME_WIDTH}x{FRAME_HEIGHT}",
        "-r", str(DEFAULT_FPS),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        path,
    ]

    pipe_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    t_start = time.time()

    for idx in range(total_frames):
        current_img = generate_scene_image_frame(idx / DEFAULT_FPS, code=code)
        pipe_proc.stdin.write(current_img.tobytes())

        if not quiet:
            elapsed = time.time() - t_start
            speed = (idx + 1) / elapsed if elapsed > 0 else 0.0
            eta = (total_frames - (idx + 1)) / speed if speed > 0 else 0.0
            pct = ((idx + 1) / total_frames) * 100.0
            meter = _construct_progress_bar(idx + 1, total_frames, 24)
            sys.stdout.write(
                f"\r  [{meter}] {pct:5.1f}% | Frame {idx + 1}/{total_frames} | {speed:.2f} fps | ETA: {format_duration_string(eta)}"
            )
            sys.stdout.flush()

    pipe_proc.stdin.close()
    exit_code = pipe_proc.wait()
    if exit_code != 0:
        raise subprocess.CalledProcessError(exit_code, pipe_proc.args)

    if not quiet:
        elapsed_total = time.time() - t_start
        file_bytes = os.path.getsize(path) if os.path.exists(path) else 0
        sys.stdout.write("\n")
        print(f"✓ Video saved successfully: '{path}' ({format_byte_size(file_bytes)}, rendered in {format_duration_string(elapsed_total)})")

def detect_ansi_color_support() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    is_interactive = getattr(sys.stdout, "isatty", lambda: False)()
    if not is_interactive:
        return False
    if os.environ.get("COLORTERM"):
        return True
    term_type = os.environ.get("TERM", "").lower()
    return any(identifier in term_type for identifier in ("color", "xterm", "screen", "ansi", "linux"))

def _calculate_terminal_viewport(
    total_rows: int,
    total_cols: int,
    full: bool = False,
) -> Tuple[int, int, int, int]:
    if full:
        return 0, total_rows, 0, total_cols

    term_dims = shutil.get_terminal_size((90, 36))
    lines_wanted = max(18, min(term_dims.lines - 2, 36))
    focus_center = 30
    half_span = lines_wanted // 2

    row_start = max(0, focus_center - half_span)
    row_end = min(total_rows, row_start + lines_wanted)
    col_start = 0
    col_end = min(total_cols, max(80, term_dims.columns))
    return row_start, row_end, col_start, col_end

def draw_ascii_terminal_frame(
    timestamp: float = 0.8,
    color: Optional[bool] = None,
    full: bool = False,
    clear_eol: bool = False,
) -> str:
    use_color = detect_ansi_color_support() if color is None else bool(color)
    chars, colors = render_depth_and_color_buffers(timestamp, GRID_ROWS_CODE)
    r_start, r_end, c_start, c_end = _calculate_terminal_viewport(chars.shape[0], chars.shape[1], full=full)

    eol_escape = "\033[K" if clear_eol else ""
    rendered_rows: List[str] = []
    for r in range(r_start, r_end):
        char_row = chars[r, c_start:c_end]
        if not use_color:
            row_str = "".join(chr(val + 32) if 0 <= val < 95 else " " for val in char_row)
            if not clear_eol:
                row_str = row_str.rstrip()
            rendered_rows.append(row_str + eol_escape)
        else:
            color_row = colors[r, c_start:c_end]
            row_tokens: List[str] = []
            for val, rgb in zip(char_row, color_row):
                glyph = chr(val + 32) if 0 <= val < 95 else " "
                if glyph == " ":
                    row_tokens.append(" ")
                else:
                    r_col = int(np.clip(rgb[0] * 255.0, 0, 255))
                    g_col = int(np.clip(rgb[1] * 255.0, 0, 255))
                    b_col = int(np.clip(rgb[2] * 255.0, 0, 255))
                    row_tokens.append(f"\033[38;2;{r_col};{g_col};{b_col}m{glyph}\033[0m")
            row_str = "".join(row_tokens)
            if not clear_eol:
                row_str = row_str.rstrip()
            rendered_rows.append(row_str + eol_escape)

    return "\n".join(rendered_rows)

class _TerminalCursorController:
    def __enter__(self):
        sys.stdout.write("\033[?25l\033[2J\033[H")
        sys.stdout.flush()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.write("\033[?25h\033[0m\n")
        sys.stdout.flush()

def stream_terminal_animation(
    fps: float = 10.0,
    dur: float = CYCLE_DURATION,
    color: Optional[bool] = None,
    full: bool = False,
    loop: bool = False,
) -> None:
    use_color = detect_ansi_color_support() if color is None else bool(color)
    time_increment = 1.0 / max(fps, 0.1)
    sim_time = 0.0

    with _TerminalCursorController():
        try:
            while True:
                cycle_start = time.time()
                frame_text = draw_ascii_terminal_frame(sim_time, color=use_color, full=full, clear_eol=True)
                sys.stdout.write(f"\033[H{frame_text}\n\033[J\033[2Kt={sim_time:5.2f}s | Press Ctrl+C to stop")
                sys.stdout.flush()

                sim_time += time_increment
                if dur and sim_time >= dur:
                    if loop:
                        sim_time = 0.0
                    else:
                        break

                cycle_cost = time.time() - cycle_start
                nap_time = max(0.0, time_increment - cycle_cost)
                if nap_time > 0:
                    time.sleep(nap_time)
        except KeyboardInterrupt:
            pass

def _setup_export_arguments(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument(
        "--code",
        "--with-code",
        dest="code",
        action="store_true",
        default=False,
        help="Display source code panel at the bottom",
    )
    subparser.add_argument(
        "--no-code",
        dest="no_code",
        action="store_true",
        default=False,
        help="Hide bottom source code panel (default behavior)",
    )
    subparser.add_argument(
        "-q",
        "--quiet",
        dest="quiet",
        action="store_true",
        default=False,
        help="Suppress progress and status logs",
    )

def _handle_terminal_view_command(args: argparse.Namespace) -> None:
    time_point = getattr(args, "time", 0.8)
    vertical_full = getattr(args, "full", False)
    color_mode = getattr(args, "color", None)
    rendered_scene = draw_ascii_terminal_frame(timestamp=time_point, color=color_mode, full=vertical_full)
    print(rendered_scene)

def _handle_terminal_play_command(args: argparse.Namespace) -> None:
    stream_terminal_animation(
        fps=args.fps,
        dur=args.duration,
        color=args.color,
        full=args.full,
        loop=args.loop,
    )

def _handle_png_export_command(args: argparse.Namespace) -> None:
    include_code_panel = bool(args.code and not args.no_code)
    if not args.quiet:
        print(f"Rendering still frame at t={args.time:g}s ({FRAME_WIDTH}x{FRAME_HEIGHT})...")
    render_start = time.time()
    still_img = generate_scene_image_frame(args.time, code=include_code_panel)
    still_img.save(args.output)
    if not args.quiet:
        render_duration = time.time() - render_start
        file_size_bytes = os.path.getsize(args.output) if os.path.exists(args.output) else 0
        print(f"✓ Still frame saved: '{args.output}' ({format_byte_size(file_size_bytes)}, {render_duration:.2f}s)")

def _handle_mp4_export_command(args: argparse.Namespace, mp4_subparser: argparse.ArgumentParser) -> None:
    include_code_panel = bool(args.code and not args.no_code)
    min_video_duration = 1.0 / DEFAULT_FPS
    if not np.isfinite(args.duration) or args.duration < min_video_duration:
        mp4_subparser.error(f"Duration must be at least {min_video_duration:g} seconds")

    try:
        encode_mp4_video_sequence(args.output, code=include_code_panel, dur=args.duration, quiet=args.quiet)
    except FileNotFoundError as err:
        sys.exit(f"\n[Error] {err}\n")
    except subprocess.CalledProcessError as err:
        sys.exit(f"\n[Error] FFmpeg execution failed with code {err.returncode}\n")

def _create_cli_argument_parser() -> tuple[argparse.ArgumentParser, dict[str, argparse.ArgumentParser]]:
    parser = argparse.ArgumentParser(
        description=f"{PROJECT_NAME} - Procedural ASCII Marine Simulator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="format")

    cmd_show = subparsers.add_parser(
        "show",
        aliases=["terminal", "print", "view"],
        help="Display ASCII scene directly in terminal",
    )
    cmd_show.add_argument("-t", "--time", type=float, default=0.8, help="Timestamp in seconds")
    cmd_show.add_argument("--full", action="store_true", default=False, help="Display full vertical frame")
    cmd_show.add_argument("--color", dest="color", action="store_true", default=None, help="Force ANSI TrueColor output")
    cmd_show.add_argument("--no-color", dest="color", action="store_false", help="Disable color (monochrome ASCII)")

    cmd_play = subparsers.add_parser(
        "play",
        aliases=["live", "anim"],
        help="Stream live ASCII animation in terminal",
    )
    cmd_play.add_argument("--fps", type=float, default=10.0, help="Target playback frame rate")
    cmd_play.add_argument("-d", "--duration", type=float, default=CYCLE_DURATION, help="Playback duration in seconds")
    cmd_play.add_argument("--loop", action="store_true", default=False, help="Loop playback indefinitely")
    cmd_play.add_argument("--full", action="store_true", default=False, help="Display full vertical frame")
    cmd_play.add_argument("--color", dest="color", action="store_true", default=None, help="Force ANSI TrueColor output")
    cmd_play.add_argument("--no-color", dest="color", action="store_false", help="Disable color (monochrome ASCII)")

    cmd_png = subparsers.add_parser(
        "png",
        help="Export high-resolution still image (PNG/WEBP/JPG)",
    )
    cmd_png.add_argument("time", type=float, help="Timestamp in seconds")
    cmd_png.add_argument("output", help="Output file path (e.g. frame.png)")
    _setup_export_arguments(cmd_png)

    cmd_mp4 = subparsers.add_parser(
        "mp4",
        help="Export encoded video sequence via FFmpeg",
    )
    cmd_mp4.add_argument("output", help="Output MP4 video file path (e.g. video.mp4)")
    cmd_mp4.add_argument("--duration", type=float, default=CYCLE_DURATION, help="Video duration in seconds")
    _setup_export_arguments(cmd_mp4)

    sub_map = {
        "show": cmd_show,
        "play": cmd_play,
        "png": cmd_png,
        "mp4": cmd_mp4,
    }
    return parser, sub_map

def execute_cli_entrypoint(argv: list[str] | None = None) -> None:
    cli_tokens = sys.argv[1:] if argv is None else argv
    if not cli_tokens:
        print(draw_ascii_terminal_frame(0.8))
        return

    parser, sub_map = _create_cli_argument_parser()
    parsed_args = parser.parse_args(cli_tokens)

    subcommand = parsed_args.format
    if subcommand in ("show", "terminal", "print", "view") or subcommand is None:
        _handle_terminal_view_command(parsed_args)
    elif subcommand in ("play", "live", "anim"):
        _handle_terminal_play_command(parsed_args)
    elif subcommand == "png":
        _handle_png_export_command(parsed_args)
    elif subcommand == "mp4":
        _handle_mp4_export_command(parsed_args, sub_map["mp4"])

main = execute_cli_entrypoint

if __name__ == "__main__":
    execute_cli_entrypoint()
