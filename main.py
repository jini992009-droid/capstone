import base64
import html
import random
import time
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

from ice_generator import generate_ice_chunks
from geometry_utils import point_to_polygon_distance
from pathfinding import (
    PathResult,
    _build_open_mask,
    find_astar_path,
    find_dijkstra_path,
    find_safety_weighted_dijkstra_path,
    find_theta_star_path,
)
from renderer import (
    figure_to_png_bytes,
    get_visible_ice_polygons,
    is_open_water_point,
    render_ice_field,
)

Point = Tuple[float, float]

TITLE = "북극 해빙 환경에서 선박 항로 생성 및 알고리즘 비교 웹앱"
CAPTION = "해빙 환경을 생성하고, 바다 위를 클릭해 선박의 출발점과 도착점을 지정한 뒤, 경로 탐색 알고리즘별 결과를 비교할 수 있습니다."
LOGO_PATH = "assets/kmou_ocean_engineering_logo.png"
STREAMLIT_URL = "https://capstone-djppq74gfrafbwsc6dpkub.streamlit.app/"
GITHUB_URL = "https://github.com/jini992009-droid/capstone.git"
START_TEXT = "\ucd9c\ubc1c\uc810"
GOAL_TEXT = "\ub3c4\ucc29\uc810"
PATH_CELL_SIZE = 0.5
METERS_PER_UNIT = 100.0
DEFAULT_EDGE_MARGIN = 0.1
DEFAULT_OUTLINE_WIDTH = 0.5

WEIGHT_PRESETS = {
    "안전 중심": {"detour": 20, "safety": 40, "time": 10, "nodes": 10, "turn": 20},
    "속도 중심": {"detour": 10, "safety": 20, "time": 30, "nodes": 30, "turn": 10},
    "균형": {"detour": 30, "safety": 30, "time": 15, "nodes": 15, "turn": 10},
}

PATH_ALGORITHMS = {
    "dijkstra": ("Dijkstra Algorithm 실행", "Dijkstra Algorithm", find_dijkstra_path),
    "safe_dijkstra": (
        "Safety-Weighted Dijkstra 실행",
        "Safety-Weighted Dijkstra",
        find_safety_weighted_dijkstra_path,
    ),
    "astar": ("A* Algorithm 실행", "A* Algorithm", find_astar_path),
    "theta_star": ("Theta* Algorithm 실행", "Theta* Algorithm", find_theta_star_path),
}


def _image_to_data_uri(path: str) -> str:
    image_path = Path(path)
    if not image_path.exists():
        return ""
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    suffix = image_path.suffix.lower().lstrip(".") or "png"
    if suffix == "jpg":
        suffix = "jpeg"
    return f"data:image/{suffix};base64,{encoded}"


def _units_to_meters(value: float) -> float:
    return float(value) * METERS_PER_UNIT


def _meters_to_units(value: float) -> float:
    return float(value) / METERS_PER_UNIT


def _format_distance(value: float, *, km_decimals: int = 2, m_decimals: int = 0) -> str:
    meters = _units_to_meters(value)
    if abs(meters) >= 1000.0:
        return f"{meters / 1000.0:.{km_decimals}f} km"
    return f"{meters:.{m_decimals}f} m"


def _format_distance_range(min_value: float, max_value: float) -> str:
    return f"{_format_distance(min_value)} ~ {_format_distance(max_value)}"


def _format_coordinate(point: Point) -> str:
    return f"({_format_distance(point[0], km_decimals=2)}, {_format_distance(point[1], km_decimals=2)})"


def _parse_distance_label(value: str) -> float:
    text = str(value).strip().lower().replace(",", "")
    if text.endswith("km"):
        return float(text[:-2].strip()) * 1000.0
    if text.endswith("m"):
        return float(text[:-1].strip())
    return float(text)

def _inject_ppt_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ppt-navy: #1f3a63;
            --ppt-blue: #4b88ff;
            --ppt-blue-soft: #eef5ff;
            --ppt-blue-mid: #dce9ff;
            --ppt-border: #d9e5f4;
            --ppt-text: #344767;
            --ppt-muted: #6d819f;
            --ppt-green: #1dbb63;
            --ppt-red: #ef4444;
            --ppt-shadow: 0 12px 32px rgba(31, 58, 99, 0.10);
        }
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(75, 136, 255, 0.09), transparent 22%),
                linear-gradient(180deg, #f3faff 0%, #fbfdff 38%, #f7fbff 100%);
            color: var(--ppt-text);
        }
        [data-testid="stAppViewContainer"] > .main { background: transparent; }
        .block-container { padding-top: 2.1rem; padding-bottom: 2.2rem; }
        [data-testid="stSidebar"] {
            background:
                radial-gradient(circle at 18% 0%, rgba(198, 236, 255, 0.92), transparent 34%),
                linear-gradient(180deg, #f4fcff 0%, #edf8fb 52%, #f8fbff 100%);
            border-right: 1px solid #cfe3ee;
            box-shadow: inset -1px 0 0 rgba(255, 255, 255, 0.72);
        }
        [data-testid="stSidebar"] .block-container { padding-top: 1.35rem; padding-bottom: 1.4rem; }
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.52rem !important; }
        [data-testid="stSidebar"] hr { display: none !important; margin: 0 !important; }
        [data-testid="stSidebar"] [data-testid="stExpander"] { margin-bottom: 0.22rem; }
        h1, h2, h3 { color: var(--ppt-navy) !important; letter-spacing: 0; }
        h1 { font-weight: 800 !important; }
        h2, h3 { font-weight: 760 !important; }
        .stCaption, .stMarkdown p, .stMarkdown li, .stText, label { color: var(--ppt-text); }
        .stCaption { color: var(--ppt-muted) !important; }
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { font-size: 1.16rem !important; margin-top: 0.15rem; }
        [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"], [data-testid="stExpander"] { border: 1px solid var(--ppt-border); border-radius: 22px; background: rgba(255, 255, 255, 0.86); box-shadow: var(--ppt-shadow); }
        [data-testid="stExpander"] details summary p { color: var(--ppt-navy) !important; font-weight: 650; }
        .stButton > button { border-radius: 999px; border: 1px solid var(--ppt-blue-mid); color: var(--ppt-navy); background: linear-gradient(180deg, #ffffff 0%, #f5f9ff 100%); font-weight: 650; box-shadow: 0 8px 20px rgba(75, 136, 255, 0.10); }
        .stButton > button:hover { border-color: var(--ppt-blue); color: var(--ppt-blue); background: linear-gradient(180deg, #f7fbff 0%, #edf5ff 100%); }
        .stButton > button[kind="primary"] { color: white; border-color: var(--ppt-blue); background: linear-gradient(135deg, #4b88ff 0%, #3b6fdd 100%); }
        [data-testid="stRadio"] label { border-radius: 999px; padding: 0.12rem 0.55rem; }
        [data-testid="stSlider"] [data-baseweb="slider"] > div > div:nth-child(2) { background: linear-gradient(90deg, #4b88ff 0%, #7ca9ff 100%) !important; }
        [data-testid="stSlider"] [role="slider"] { border: 2px solid white !important; box-shadow: 0 4px 10px rgba(75, 136, 255, 0.25); background: var(--ppt-blue) !important; }
        [data-testid="stSlider"] *::selection { background: transparent !important; color: inherit !important; }
        [data-testid="stSlider"] p, [data-testid="stSlider"] span { user-select: none; }
        [data-testid="stInfo"], [data-testid="stSuccess"], [data-testid="stWarning"], [data-testid="stError"] { border-radius: 18px; border-width: 1px; }
        [data-testid="stInfo"] { background: #edf5ff; border-color: #cfe0ff; }
        [data-testid="stSuccess"] { background: #effaf4; border-color: #cbeed8; }
        [data-testid="stWarning"] { background: #fff8eb; border-color: #f5dfaa; }
        [data-testid="stError"] { background: #fff1f2; border-color: #f5c3c7; }
        [data-testid="stImage"] img { border-radius: 24px; border: 1px solid var(--ppt-border); box-shadow: var(--ppt-shadow); }
        .ppt-weight-card, .ppt-comparison-card { border: 1px solid var(--ppt-border); border-radius: 24px; background: linear-gradient(180deg, #ffffff 0%, #f9fbff 100%); box-shadow: var(--ppt-shadow); padding: 1.15rem 1.2rem; }
        .ppt-weight-title, .ppt-comparison-title { color: var(--ppt-navy); font-size: 1.05rem; font-weight: 760; }
        .ppt-weight-subtitle { color: var(--ppt-muted); font-size: 0.84rem; margin-bottom: 1rem; }
        .ppt-weight-row { display: grid; grid-template-columns: 76px minmax(64px, 1fr) 42px; gap: 0.7rem; align-items: center; margin-bottom: 0.8rem; font-size: 0.9rem; color: var(--ppt-text); }
        .ppt-weight-row > div:first-child { white-space: nowrap; }
        .ppt-weight-track { height: 10px; border-radius: 999px; background: #e8eef7; overflow: hidden; }
        .ppt-weight-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #4b88ff 0%, #78a4ff 100%); }
        .ppt-weight-pills { display: flex; gap: 0.55rem; margin-top: 0.9rem; flex-wrap: wrap; }
        .ppt-pill { padding: 0.45rem 0.9rem; border-radius: 999px; font-size: 0.84rem; font-weight: 700; border: 1px solid var(--ppt-blue-mid); color: var(--ppt-navy); background: #eff5ff; }
        .ppt-pill.active { color: white; border-color: var(--ppt-blue); background: linear-gradient(135deg, #4b88ff 0%, #3b6fdd 100%); box-shadow: 0 8px 18px rgba(75, 136, 255, 0.24); }
        .ppt-comparison-shell { position: relative; }
        .ppt-comparison-toggle { position: absolute; opacity: 0; pointer-events: none; }
        .ppt-comparison-header { display: flex; align-items: center; justify-content: space-between; gap: 0.8rem; margin-bottom: 0.9rem; }
        .ppt-comparison-title { margin-bottom: 0; }
        .ppt-comparison-expand, .ppt-comparison-close { display: inline-flex; align-items: center; justify-content: center; width: 2.2rem; height: 2.2rem; border-radius: 999px; border: 1px solid var(--ppt-border); background: #ffffff; color: var(--ppt-navy); font-weight: 800; cursor: pointer; box-shadow: 0 8px 18px rgba(31, 58, 99, 0.10); user-select: none; }
        .ppt-comparison-expand:hover, .ppt-comparison-close:hover { color: var(--ppt-blue); border-color: var(--ppt-blue-mid); background: #f5f9ff; }
        .ppt-comparison-close { display: none; position: fixed; top: 1.2rem; right: 1.2rem; z-index: 1000001; width: auto; min-width: 4.4rem; padding: 0 0.9rem; }
        .ppt-comparison-toggle:checked + .ppt-comparison-card { position: fixed; inset: 1.4rem; z-index: 1000000; overflow: auto; padding: 1.45rem; background: #ffffff; }
        .ppt-comparison-toggle:checked + .ppt-comparison-card .ppt-comparison-close { display: inline-flex; }
        .ppt-comparison-toggle:checked + .ppt-comparison-card .ppt-comparison-scroll { overflow: auto; max-height: calc(100vh - 8rem); }
        .ppt-comparison-toggle:checked + .ppt-comparison-card .ppt-comparison-table { min-width: 1180px; width: 100%; font-size: 1rem; }
        .ppt-comparison-scroll { width: 100%; overflow-x: auto; overflow-y: hidden; padding-bottom: 0.35rem; }
        .ppt-comparison-scroll::-webkit-scrollbar { height: 8px; }
        .ppt-comparison-scroll::-webkit-scrollbar-thumb { background: #c7d6ec; border-radius: 999px; }
        .ppt-comparison-table { width: max-content; min-width: 1080px; border-collapse: separate; border-spacing: 0; table-layout: auto; font-size: 0.92rem; overflow: hidden; border-radius: 18px; border: 1px solid var(--ppt-border); }
        .ppt-comparison-table thead th { background: #eff5ff; color: var(--ppt-navy); padding: 0.78rem 0.72rem; font-weight: 760; text-align: center; border-bottom: 1px solid var(--ppt-border); white-space: nowrap; line-height: 1.25; }
        .ppt-comparison-table thead th:first-child, .ppt-comparison-table tbody td:first-child { text-align: left; font-weight: 760; min-width: 132px; }
        .ppt-comparison-table tbody td { padding: 0.78rem 0.72rem; text-align: center; color: var(--ppt-text); border-bottom: 1px solid #edf2f9; background: rgba(255, 255, 255, 0.95); white-space: nowrap; line-height: 1.35; }
        .ppt-comparison-table tbody tr:last-child td { border-bottom: none; }
        .ppt-comparison-table tbody tr:hover td { background: #f8fbff; }
        .ppt-comparison-table .best { color: var(--ppt-green); font-weight: 760; }
        .ppt-comparison-table .warn { color: var(--ppt-red); font-weight: 760; }
        .ppt-comparison-table .emph { color: var(--ppt-blue); font-weight: 760; }
        .asv-main-divider {
            width: 1px;
            min-height: 44rem;
            margin: 0.25rem auto 0;
            background: linear-gradient(180deg, transparent 0%, #dbe7f1 8%, #dbe7f1 92%, transparent 100%);
        }
        .asv-right-divider {
            width: 1px;
            min-height: 15.5rem;
            margin: 0.35rem auto 0;
            background: linear-gradient(180deg, transparent 0%, #dbe7f1 12%, #dbe7f1 88%, transparent 100%);
        }
        .asv-section-divider {
            height: 1px;
            width: 100%;
            margin: 1.15rem 0 0.85rem;
            background: linear-gradient(90deg, transparent 0%, #dbe7f1 8%, #dbe7f1 92%, transparent 100%);
        }
        .asv-hero {
            position: relative;
            overflow: hidden;
            border: 1px solid #cfe5f2;
            border-radius: 24px;
            padding: 1.45rem 1.65rem 1.5rem;
            margin: 0 0 1.55rem 0;
            background:
                radial-gradient(circle at 88% 18%, rgba(112, 197, 231, 0.32), transparent 34%),
                linear-gradient(135deg, #ffffff 0%, #eef9ff 48%, #dff4fb 100%);
            box-shadow: 0 16px 38px rgba(31, 58, 99, 0.11);
        }
        .asv-hero::after {
            content: "";
            position: absolute;
            right: -4rem;
            top: -5rem;
            width: 16rem;
            height: 16rem;
            border-radius: 999px;
            background: rgba(75, 136, 255, 0.10);
        }
        .asv-hero-badge {
            display: inline-flex;
            align-items: center;
            border: 1px solid #a8d8ef;
            border-radius: 999px;
            padding: 0.32rem 0.72rem;
            color: #1670a5;
            background: rgba(255, 255, 255, 0.68);
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 0.72rem;
        }
        .asv-hero-title {
            position: relative;
            z-index: 1;
            color: var(--ppt-navy) !important;
            font-size: 2.18rem;
            line-height: 1.08;
            font-weight: 850;
            margin: 0 0 0.72rem 0;
        }
        .asv-hero-caption {
            position: relative;
            z-index: 1;
            color: #61758e;
            font-size: 0.98rem;
            margin: 0;
        }
        [data-testid="stSidebar"] [data-testid="stImage"] img {
            box-sizing: border-box;
            width: 100% !important;
            object-fit: contain;
            padding: 0.68rem 0.82rem;
            background: rgba(255, 255, 255, 0.72);
            border-radius: 18px;
        }
        .asv-sidebar-card-heading {
            color: var(--ppt-navy);
            font-size: 1.08rem;
            font-weight: 800;
            line-height: 1.35;
            margin: 0.1rem 0 0.28rem;
        }
        .asv-sidebar-card-subtitle {
            color: var(--ppt-muted);
            font-size: 0.82rem;
            line-height: 1.45;
            margin: 0 0 0.7rem;
        }
        .asv-sidebar-project-card {
            margin-top: 1.1rem;
            padding: 1rem 1rem 1.05rem;
            border: 1px solid #cfe5f2;
            border-radius: 20px;
            background:
                radial-gradient(circle at 88% 8%, rgba(75, 136, 255, 0.12), transparent 34%),
                linear-gradient(180deg, rgba(255, 255, 255, 0.88) 0%, rgba(243, 250, 255, 0.92) 100%);
            box-shadow: 0 10px 26px rgba(31, 58, 99, 0.08);
        }
        .asv-sidebar-project-kicker {
            color: var(--ppt-blue);
            font-size: 0.72rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.45rem;
        }
        .asv-sidebar-project-title {
            color: var(--ppt-navy);
            font-size: 1.02rem;
            font-weight: 780;
            line-height: 1.35;
            margin-bottom: 0.65rem;
        }
        .asv-sidebar-project-meta {
            display: grid;
            gap: 0.32rem;
            color: var(--ppt-text);
            font-size: 0.86rem;
            line-height: 1.45;
        }
        .asv-sidebar-project-meta span {
            color: var(--ppt-muted);
            font-weight: 650;
            margin-right: 0.35rem;
        }
        .asv-sidebar-links {
            display: flex;
            gap: 0.45rem;
            flex-wrap: wrap;
            margin-top: 0.85rem;
        }
        .asv-sidebar-links a {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0.34rem 0.62rem;
            border-radius: 999px;
            border: 1px solid var(--ppt-blue-mid);
            color: var(--ppt-navy) !important;
            background: rgba(255, 255, 255, 0.78);
            font-size: 0.78rem;
            font-weight: 720;
            text-decoration: none !important;
        }
        .asv-sidebar-card-logo {
            margin-top: 0.9rem;
            padding: 0.64rem 0.72rem;
            border: 1px solid rgba(207, 229, 242, 0.92);
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.78);
        }
        .asv-sidebar-card-logo img {
            display: block;
            width: 100%;
            height: auto;
            object-fit: contain;
        }
        .asv-sidebar-links a:hover {
            color: var(--ppt-blue) !important;
            border-color: var(--ppt-blue);
            background: #ffffff;
        }
        .asv-reference-section {
            margin-top: 1.6rem;
            padding: 1.25rem 1.35rem 1.35rem;
            border: 1px solid #cfe5f2;
            border-radius: 24px;
            background:
                radial-gradient(circle at 92% 12%, rgba(112, 197, 231, 0.18), transparent 28%),
                linear-gradient(135deg, #ffffff 0%, #f6fbff 54%, #edf8ff 100%);
            box-shadow: 0 14px 34px rgba(31, 58, 99, 0.09);
        }
        .asv-reference-kicker {
            color: var(--ppt-blue);
            font-size: 0.76rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.35rem;
        }
        .asv-reference-section h2 {
            margin: 0 0 0.35rem !important;
            color: var(--ppt-navy) !important;
            font-size: 1.45rem !important;
            line-height: 1.3 !important;
        }
        .asv-reference-details {
            display: grid;
            gap: 0.72rem;
            margin-top: 1rem;
        }
        .asv-reference-details details {
            border: 1px solid var(--ppt-border);
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.86);
            overflow: hidden;
        }
        .asv-reference-details summary {
            cursor: pointer;
            padding: 0.92rem 1rem;
            color: var(--ppt-navy);
            font-weight: 780;
            list-style-position: inside;
        }
        .asv-reference-details summary::marker {
            color: var(--ppt-blue);
        }
        .asv-reference-detail-body {
            padding: 0 1rem 0.95rem 1.35rem;
            color: var(--ppt-text);
            font-size: 0.92rem;
            line-height: 1.62;
        }
        .asv-reference-detail-body p {
            margin: 0.32rem 0;
        }
        .asv-reference-detail-body ul {
            margin: 0.2rem 0 0;
            padding-left: 1.1rem;
        }
        .asv-reference-detail-body li {
            margin: 0.24rem 0;
        }
        .asv-reference-detail-body code {
            color: #087a3e;
            background: #f4fbf7;
            border-radius: 6px;
            padding: 0.06rem 0.22rem;
        }
        .asv-reference-detail-body a {
            color: var(--ppt-blue) !important;
            font-weight: 720;
            text-decoration: none !important;
        }
        .asv-reference-detail-body a:hover {
            text-decoration: underline !important;
        }
        .asv-reference-caption {
            color: var(--ppt-muted);
            margin-bottom: 1.05rem;
            line-height: 1.55;
        }
        .asv-reference-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.9rem;
        }
        .asv-reference-card {
            border: 1px solid var(--ppt-border);
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.84);
            padding: 0.95rem 1rem;
        }
        .asv-reference-card h3 {
            margin: 0 0 0.55rem !important;
            color: var(--ppt-navy) !important;
            font-size: 1rem !important;
            line-height: 1.35 !important;
        }
        .asv-reference-card p, .asv-reference-card li {
            color: var(--ppt-text);
            font-size: 0.91rem;
            line-height: 1.58;
            margin: 0.28rem 0;
        }
        .asv-reference-card ul {
            margin: 0;
            padding-left: 1.1rem;
        }
        .asv-reference-card code {
            color: #087a3e;
            background: #f4fbf7;
            border-radius: 6px;
            padding: 0.06rem 0.22rem;
        }
        .asv-reference-card a {
            color: var(--ppt-blue) !important;
            font-weight: 720;
            text-decoration: none !important;
        }
        .asv-reference-card a:hover { text-decoration: underline !important; }
        @media (max-width: 1100px) {
            .asv-reference-grid { grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_weight_preview_card(
    detour_weight: int,
    safety_weight: int,
    time_weight: int,
    nodes_weight: int,
    turn_weight: int,
) -> str:
    def _row(label: str, value: int) -> str:
        fill_width = min(100, max(0, value * 2))
        return (
            '<div class="ppt-weight-row">'
            f"<div>{html.escape(label)}</div>"
            f'<div class="ppt-weight-track"><div class="ppt-weight-fill" style="width:{fill_width}%;"></div></div>'
            f"<div>{value}%</div>"
            '</div>'
        )

    rows = [
        _row("우회율", detour_weight),
        _row("안전성", safety_weight),
        _row("계산 시간", time_weight),
        _row("방문 노드", nodes_weight),
        _row("굴곡 정도", turn_weight),
    ]

    return (
        '<div class="ppt-weight-card">'
        '<div class="ppt-weight-title">가중치 조절</div>'
        '<div class="ppt-weight-subtitle">5가지 평가 항목 가중치 설정</div>'
        f"{''.join(rows)}"
        '</div>'
    )


def _render_safety_radius_preview_card(safety_radius_m: int) -> str:
    circle_diameter = max(44, min(124, 24 + int(safety_radius_m * 0.45)))
    return (
        '<div class="ppt-weight-card" style="padding:1rem 1.1rem; margin-top:0.45rem;">'
        '<div class="ppt-weight-title">안전 반경 미리보기</div>'
        '<div class="ppt-weight-subtitle">중심점 기준 원형 반경 표현</div>'
        '<div style="display:flex; justify-content:center; align-items:center; padding:0.4rem 0 0.75rem;">'
        '<div style="position:relative; width:132px; height:132px; display:flex; align-items:center; justify-content:center;">'
        f'<div style="width:{circle_diameter}px; height:{circle_diameter}px; border-radius:50%; border:2px solid #4b88ff; background:rgba(75, 136, 255, 0.10);"></div>'
        '<div style="position:absolute; width:10px; height:10px; border-radius:50%; background:#1f3a63; box-shadow:0 0 0 4px rgba(75, 136, 255, 0.12);"></div>'
        '</div>'
        '</div>'
        f'<div style="text-align:center; color:#1f3a63; font-weight:700; font-size:0.92rem;">현재 안전 반경: {safety_radius_m} m</div>'
        '</div>'
    )


def _render_comparison_table_html(comparison_rows: List[dict]) -> str:
    if not comparison_rows:
        return ""

    headers = list(comparison_rows[0].keys())
    numeric_columns = {
        "경로 길이": _parse_distance_label,
        "방문 노드 수": lambda value: int(value),
        "우회율": lambda value: float(value),
        "굴곡 횟수": lambda value: int(value),
        "최소 해빙 거리": _parse_distance_label,
        "위험 표시 지점 수": lambda value: int(value),
    }

    duration_values = []
    for row in comparison_rows:
        duration_text = row["계산 시간"]
        if isinstance(duration_text, str) and duration_text.endswith("초"):
            duration_values.append(float(duration_text.replace("초", "")))
    best_duration = min(duration_values) if duration_values else None

    best_values = {}
    worst_values = {}
    for column, parser in numeric_columns.items():
        values = [parser(row[column]) for row in comparison_rows]
        if column == "최소 해빙 거리":
            best_values[column] = max(values)
            worst_values[column] = min(values)
        else:
            best_values[column] = min(values)
            worst_values[column] = max(values)

    score_values = [float(row["종합 점수"].split("/")[0].strip()) for row in comparison_rows]
    best_score = max(score_values)
    head_html = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body_rows = []

    for row in comparison_rows:
        cells = []
        for header in headers:
            text = str(row[header])
            class_name = ""
            if header == "계산 시간" and best_duration is not None and text.endswith("초"):
                duration_value = float(text.replace("초", ""))
                if duration_value == best_duration:
                    class_name = "best"
            elif header in numeric_columns:
                value = numeric_columns[header](text)
                if header == "최소 해빙 거리":
                    if value == best_values[header]:
                        class_name = "best"
                    elif value == worst_values[header]:
                        class_name = "warn"
                elif header == "위험 표시 지점 수":
                    if value == best_values[header]:
                        class_name = "best"
                    elif value == worst_values[header]:
                        class_name = "warn"
                elif value == best_values[header] and header in ("우회율", "굴곡 횟수", "방문 노드 수"):
                    class_name = "best"
            elif header == "종합 점수":
                score_value = float(text.split("/")[0].strip())
                if score_value == best_score:
                    class_name = "best"
            elif header == "안전 등급":
                if text in ("안전", "A+", "A"):
                    class_name = "best"
                elif text in ("위험", "B", "C"):
                    class_name = "warn"
            elif header == "알고리즘" and "Safety-Weighted" in text:
                class_name = "emph"
            cells.append(f'<td class="{class_name}">{html.escape(text)}</td>')
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    return (
        '<div class="ppt-comparison-shell">'
        '<input class="ppt-comparison-toggle" id="ppt-comparison-fullscreen" type="checkbox">'
        '<div class="ppt-comparison-card">'
        '<div class="ppt-comparison-header">'
        '<div class="ppt-comparison-title">알고리즘별 수치 비교표</div>'
        '<label class="ppt-comparison-expand" for="ppt-comparison-fullscreen" title="전체화면으로 보기">⛶</label>'
        '</div>'
        '<label class="ppt-comparison-close" for="ppt-comparison-fullscreen">닫기</label>'
        '<div class="ppt-comparison-scroll"><table class="ppt-comparison-table">'
        f"<thead><tr>{head_html}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        '</table></div>'
        '</div>'
        '</div>'
    )
def _nearest_ice_distance(point: Point, visible_polygons) -> float:
    if not visible_polygons:
        return float("inf")
    return min(point_to_polygon_distance(point, polygon) for polygon in visible_polygons)


def _sample_path_for_safety(path_points: List[Point], sample_interval: float = 0.25) -> List[Point]:
    if len(path_points) < 2:
        return path_points[:]

    sampled: List[Point] = []
    for start, end in zip(path_points[:-1], path_points[1:]):
        distance = float(np.hypot(end[0] - start[0], end[1] - start[1]))
        sample_count = max(1, int(np.ceil(distance / sample_interval)))

        for index in range(sample_count):
            ratio = index / sample_count
            sampled.append(
                (
                    start[0] + (end[0] - start[0]) * ratio,
                    start[1] + (end[1] - start[1]) * ratio,
                )
            )

    sampled.append(path_points[-1])
    return sampled


def _find_hazard_points(
    path_points: Optional[List[Point]],
    visible_polygons,
    safety_radius: float,
    min_marker_gap: float = 2.0,
    max_markers: int = 8,
) -> List[Point]:
    if path_points is None or len(path_points) < 2:
        return []

    hazard_threshold = safety_radius * 0.5
    hazard_points: List[Point] = []
    for point in _sample_path_for_safety(path_points):
        if _nearest_ice_distance(point, visible_polygons) >= hazard_threshold:
            continue

        if hazard_points:
            last = hazard_points[-1]
            if np.hypot(point[0] - last[0], point[1] - last[1]) < min_marker_gap:
                continue

        hazard_points.append(point)
        if len(hazard_points) >= max_markers:
            break

    return hazard_points


def _evaluate_route_safety(min_clearance: float, safety_radius: float) -> Tuple[str, str]:
    warning_threshold = safety_radius * 0.5
    if min_clearance < warning_threshold:
        return "위험", "error"
    if min_clearance < safety_radius:
        return "주의", "warning"
    return "안전", "success"


def _clamp_score(value: float) -> float:
    return max(0.0, min(100.0, value))


def _total_grid_node_count(width: float, height: float, cell_size: float) -> int:
    cols = max(1, int(np.ceil(width / cell_size)))
    rows = max(1, int(np.ceil(height / cell_size)))
    return cols * rows


def _calculate_safety_score(min_clearance: float, safety_radius: float) -> float:
    if safety_radius <= 0.0:
        return 100.0

    warning_threshold = safety_radius * 0.5

    if min_clearance >= safety_radius:
        return 100.0

    if min_clearance >= warning_threshold:
        ratio = (min_clearance - warning_threshold) / max(safety_radius - warning_threshold, 1e-9)
        return _clamp_score(60.0 + ratio * 40.0)

    return _clamp_score((min_clearance / max(warning_threshold, 1e-9)) * 60.0)


def _calculate_route_score(
    result: PathResult,
    duration_seconds: Optional[float],
    safety_radius: float,
    total_grid_nodes: int,
    score_weights: dict,
) -> float:
    detour_score = _clamp_score(100.0 - max(result.detour_ratio - 1.0, 0.0) * 80.0)
    safety_score = _calculate_safety_score(result.min_clearance, safety_radius)
    turn_score = _clamp_score(100.0 - result.turn_count * 2.0 - result.total_turn_angle * 0.03)

    if duration_seconds is None:
        time_score = 0.0
    else:
        time_score = _clamp_score(100.0 - duration_seconds * 30.0)

    node_score = _clamp_score(100.0 * (1.0 - result.visited_nodes / max(total_grid_nodes, 1)))
    weight_sum = max(sum(score_weights.values()), 1e-9)

    return (
        detour_score * score_weights["detour"]
        + safety_score * score_weights["safety"]
        + time_score * score_weights["time"]
        + node_score * score_weights["nodes"]
        + turn_score * score_weights["turn"]
    ) / weight_sum


def _count_hazard_markers(
    result: PathResult,
    visible_polygons: List[np.ndarray],
    safety_radius: float,
) -> int:
    return len(
        _find_hazard_points(
            path_points=result.path_points,
            visible_polygons=visible_polygons,
            safety_radius=safety_radius,
        )
    )


def _build_comparison_row(
    algorithm_name: str,
    result: PathResult,
    duration_seconds: Optional[float],
    safety_radius: float,
    total_grid_nodes: int,
    score_weights: dict,
    visible_polygons: List[np.ndarray],
) -> dict:
    safety_grade, _ = _evaluate_route_safety(result.min_clearance, safety_radius)
    route_score = _calculate_route_score(
        result=result,
        duration_seconds=duration_seconds,
        safety_radius=safety_radius,
        total_grid_nodes=total_grid_nodes,
        score_weights=score_weights,
    )
    duration_text = "-" if duration_seconds is None else f"{duration_seconds:.2f}\ucd08"

    return {
        "\uc54c\uace0\ub9ac\uc998": algorithm_name,
        "경로 길이": _format_distance(result.distance),
        "\uacc4\uc0b0 \uc2dc\uac04": duration_text,
        "\ubc29\ubb38 \ub178\ub4dc \uc218": result.visited_nodes,
        "\uc6b0\ud68c\uc728": f"{result.detour_ratio:.2f}",
        "\uad74\uace1 \ud69f\uc218": result.turn_count,
        "최소 해빙 거리": _format_distance(result.min_clearance),
        "\uc704\ud5d8 \ud45c\uc2dc \uc9c0\uc810 \uc218": _count_hazard_markers(result, visible_polygons, safety_radius),
        "\uc548\uc804 \ub4f1\uae09": safety_grade,
        "\uc885\ud569 \uc810\uc218": f"{route_score:.1f} / 100",
    }


def _run_path_algorithm(
    algorithm_key: str,
    algorithm_finder,
    width: float,
    height: float,
    start_point: Point,
    goal_point: Point,
    visible_polygons: List[np.ndarray],
    cell_size: float,
    safety_radius: float,
    progress_callback,
    open_mask: Optional[np.ndarray] = None,
) -> Optional[PathResult]:
    common_kwargs = {
        "width": width,
        "height": height,
        "start_point": start_point,
        "goal_point": goal_point,
        "visible_polygons": visible_polygons,
        "cell_size": cell_size,
        "progress_callback": progress_callback,
        "open_mask": open_mask,
    }

    if algorithm_key == "safe_dijkstra":
        return algorithm_finder(
            **common_kwargs,
            safety_radius=safety_radius,
            penalty_strength=3.0,
        )

    return algorithm_finder(**common_kwargs)


def _render_path_result_image(
    width: float,
    height: float,
    chunks,
    outline_width: float,
    show_grid: bool,
    start_point: Optional[Point],
    goal_point: Optional[Point],
    path_result: PathResult,
    visible_polygons: List[np.ndarray],
    safety_radius: float,
) -> Image.Image:
    hazard_points = _find_hazard_points(
        path_points=path_result.path_points,
        visible_polygons=visible_polygons,
        safety_radius=safety_radius,
    )
    figure = render_ice_field(
        width=width,
        height=height,
        chunks=chunks,
        outline_width=outline_width,
        show_grid=show_grid,
        start_point=start_point,
        goal_point=goal_point,
        path_points=path_result.path_points,
        path_hazard_points=hazard_points,
    )
    image_bytes = figure_to_png_bytes(figure)
    plt.close(figure)
    return Image.open(BytesIO(image_bytes))


def _ensure_state() -> None:
    defaults = {
        "selection_target": START_TEXT,
        "selection_target_widget": START_TEXT,
        "pending_selection_target": None,
        "start_point": None,
        "goal_point": None,
        "last_processed_click": None,
        "click_feedback": None,
        "image_click_key_index": 0,
        "path_request_signature": None,
        "path_result": None,
        "path_result_signature": None,
        "path_error_message": None,
        "last_path_duration": None,
        "path_algorithm": None,
        "comparison_signature": None,
        "comparison_results": None,
        "comparison_errors": None,
        "previous_path_snapshot": None,
        "previous_comparison_snapshot": None,
        "score_weight_detour": 30,
        "score_weight_safety": 30,
        "score_weight_time": 15,
        "score_weight_nodes": 15,
        "score_weight_turn": 10,
        "weight_preset": "균형",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _remember_current_results() -> None:
    if (
        st.session_state.path_result is not None
        and st.session_state.path_result_signature is not None
        and st.session_state.path_algorithm in PATH_ALGORITHMS
    ):
        st.session_state.previous_path_snapshot = {
            "algorithm_key": st.session_state.path_algorithm,
            "signature": st.session_state.path_result_signature,
            "result": st.session_state.path_result,
            "duration": st.session_state.last_path_duration,
            "start_point": st.session_state.start_point,
            "goal_point": st.session_state.goal_point,
        }

    if st.session_state.comparison_results:
        st.session_state.previous_comparison_snapshot = {
            "signature": st.session_state.comparison_signature,
            "results": st.session_state.comparison_results,
            "errors": st.session_state.comparison_errors,
            "start_point": st.session_state.start_point,
            "goal_point": st.session_state.goal_point,
        }


def _clear_current_path_results() -> None:
    st.session_state.path_request_signature = None
    st.session_state.path_result = None
    st.session_state.path_result_signature = None
    st.session_state.path_error_message = None
    st.session_state.last_path_duration = None
    st.session_state.path_algorithm = None
    st.session_state.comparison_signature = None
    st.session_state.comparison_results = None
    st.session_state.comparison_errors = None


def _restore_previous_result_snapshot() -> None:
    snapshot = st.session_state.previous_path_snapshot
    if not snapshot:
        return

    st.session_state.start_point = snapshot.get("start_point")
    st.session_state.goal_point = snapshot.get("goal_point")
    st.session_state.path_algorithm = snapshot.get("algorithm_key")
    st.session_state.path_request_signature = snapshot.get("signature")
    st.session_state.path_result = snapshot.get("result")
    st.session_state.path_result_signature = snapshot.get("signature")
    st.session_state.path_error_message = None
    st.session_state.last_path_duration = snapshot.get("duration")

    previous_comparison = st.session_state.previous_comparison_snapshot
    snapshot_signature = snapshot.get("signature")
    previous_base_signature = snapshot_signature[:-1] if snapshot_signature else None
    if previous_comparison and previous_comparison.get("signature") == previous_base_signature:
        st.session_state.comparison_signature = previous_comparison.get("signature")
        st.session_state.comparison_results = previous_comparison.get("results")
        st.session_state.comparison_errors = previous_comparison.get("errors") or {}
    else:
        algorithm_key = snapshot.get("algorithm_key")
        st.session_state.comparison_signature = previous_base_signature
        st.session_state.comparison_results = {
            algorithm_key: {
                "result": snapshot.get("result"),
                "duration": snapshot.get("duration"),
            }
        } if algorithm_key else {}
        st.session_state.comparison_errors = {}

def _clear_points() -> None:
    st.session_state.selection_target = START_TEXT
    st.session_state.start_point = None
    st.session_state.goal_point = None
    st.session_state.last_processed_click = None
    st.session_state.click_feedback = None
    st.session_state.pending_selection_target = START_TEXT
    _clear_current_path_results()
    st.session_state.image_click_key_index += 1


def _image_pixel_to_sea_point(
    click_x: int,
    click_y: int,
    image_width: int,
    image_height: int,
    sea_width: float,
    sea_height: float,
) -> Optional[Point]:
    if click_x < 0 or click_x > image_width or click_y < 0 or click_y > image_height:
        return None

    usable_width = max(float(image_width), 1.0)
    usable_height = max(float(image_height), 1.0)

    sea_x = (click_x / usable_width) * sea_width
    sea_y = ((usable_height - click_y) / usable_height) * sea_height
    return (float(sea_x), float(sea_y))


def _validate_saved_points(
    width: float,
    height: float,
    visible_polygons,
) -> Optional[str]:
    removed_labels = []

    if st.session_state.start_point is not None:
        if not is_open_water_point(st.session_state.start_point, width, height, visible_polygons):
            st.session_state.start_point = None
            removed_labels.append(START_TEXT)

    if st.session_state.goal_point is not None:
        if not is_open_water_point(st.session_state.goal_point, width, height, visible_polygons):
            st.session_state.goal_point = None
            removed_labels.append(GOAL_TEXT)

    if removed_labels:
        st.session_state.last_processed_click = None
        _clear_current_path_results()
        return ", ".join(removed_labels)

    return None


def _build_path_signature(
    seed: int,
    ice_count: int,
    min_diameter: float,
    max_diameter: float,
    min_vertices: int,
    max_vertices: int,
    edge_margin: float,
    min_gap: float,
    safety_radius: float,
    start_point: Optional[Point],
    goal_point: Optional[Point],
) -> Optional[Tuple]:
    if start_point is None or goal_point is None:
        return None

    return (
        int(seed),
        int(ice_count),
        round(min_diameter, 3),
        round(max_diameter, 3),
        int(min_vertices),
        int(max_vertices),
        round(edge_margin, 3),
        round(min_gap, 3),
        round(start_point[0], 3),
        round(start_point[1], 3),
        round(goal_point[0], 3),
        round(goal_point[1], 3),
        round(PATH_CELL_SIZE, 3),
        round(safety_radius, 3),
    )


st.set_page_config(page_title=TITLE, layout="wide")
_ensure_state()
_inject_ppt_theme()

if st.session_state.pending_selection_target is not None:
    st.session_state.selection_target = st.session_state.pending_selection_target
    st.session_state.selection_target_widget = st.session_state.pending_selection_target
    st.session_state.pending_selection_target = None

st.markdown(
    f"""
    <section class="asv-hero">
        <div class="asv-hero-badge">Arctic Ice Route Generation & Algorithm Comparison Web App</div>
        <h1 class="asv-hero-title">{TITLE}</h1>
        <p class="asv-hero-caption">{CAPTION}</p>
    </section>
    """,
    unsafe_allow_html=True,
)
path_timer_placeholder = st.empty()

with st.sidebar:
    with st.container(border=True):
        st.markdown(
            """
            <div class="asv-sidebar-card-heading">환경 설정</div>
            <div class="asv-sidebar-card-subtitle">해빙 환경과 경로 평가 기준을 조정합니다.</div>
            """,
            unsafe_allow_html=True,
        )
        seed = st.number_input("해빙 랜덤 배치", min_value=0, value=42, step=1)
        with st.expander("해빙 세부 설정", expanded=False):
            ice_count = st.slider("얼음 개수", min_value=20, max_value=220, value=85, step=5)
            min_diameter_m = st.slider("최소 직경 (m)", min_value=40, max_value=300, value=80, step=10)
            max_diameter_m = st.slider("최대 직경 (m)", min_value=80, max_value=500, value=180, step=10)
            min_vertices = st.slider(
                "최소 꼭짓점 수", min_value=6, max_value=12, value=8, step=1
            )
            max_vertices = st.slider(
                "최대 꼭짓점 수", min_value=8, max_value=24, value=14, step=1
            )
            min_gap_m = st.slider("최소 간격 (m)", min_value=0, max_value=150, value=10, step=5)
            show_grid = st.checkbox("격자 표시", value=False, help="체크하면 30×30 격자를 표시합니다.")

        min_diameter = _meters_to_units(min_diameter_m)
        max_diameter = _meters_to_units(max_diameter_m)
        edge_margin = DEFAULT_EDGE_MARGIN
        min_gap = _meters_to_units(min_gap_m)
        outline_width = DEFAULT_OUTLINE_WIDTH

        with st.expander("선박 안전 반경 설정", expanded=False):
            ship_safety_radius_m = st.slider(
                "선박 안전 반경 (m)",
                min_value=10,
                max_value=200,
                value=100,
                step=10,
            )
            risk_threshold_m = int(round(ship_safety_radius_m * 0.5))
            st.caption(
                f"최소 해빙 거리가 `{risk_threshold_m} m` 미만이면 `위험`, "
                f"`{risk_threshold_m}~{ship_safety_radius_m} m` 구간이면 `주의`, "
                f"`{ship_safety_radius_m} m` 이상이면 `안전`으로 평가합니다."
            )
            st.markdown(_render_safety_radius_preview_card(ship_safety_radius_m), unsafe_allow_html=True)
        ship_safety_radius = _meters_to_units(ship_safety_radius_m)

        with st.expander("평가 가중치 설정", expanded=False):
            preset_cols = st.columns(3)
            for column, preset_name in zip(preset_cols, ("안전 중심", "속도 중심", "균형")):
                button_type = "primary" if st.session_state.weight_preset == preset_name else "secondary"
                if column.button(preset_name, key=f"weight_preset_button_{preset_name}", type=button_type, use_container_width=True):
                    preset_values = WEIGHT_PRESETS[preset_name]
                    st.session_state.score_weight_detour = preset_values["detour"]
                    st.session_state.score_weight_safety = preset_values["safety"]
                    st.session_state.score_weight_time = preset_values["time"]
                    st.session_state.score_weight_nodes = preset_values["nodes"]
                    st.session_state.score_weight_turn = preset_values["turn"]
                    st.session_state.weight_preset = preset_name
                    st.rerun()

            score_weight_detour = st.slider("우회율 가중치", min_value=0, max_value=50, step=5, key="score_weight_detour")
            score_weight_safety = st.slider("안전성 가중치", min_value=0, max_value=50, step=5, key="score_weight_safety")
            score_weight_time = st.slider("계산 시간 가중치", min_value=0, max_value=50, step=5, key="score_weight_time")
            score_weight_nodes = st.slider("방문 노드 가중치", min_value=0, max_value=50, step=5, key="score_weight_nodes")
            score_weight_turn = st.slider("굴곡 정도 가중치", min_value=0, max_value=50, step=5, key="score_weight_turn")
            matched_preset = next((
                name
                for name, values in WEIGHT_PRESETS.items()
                if values["detour"] == score_weight_detour
                and values["safety"] == score_weight_safety
                and values["time"] == score_weight_time
                and values["nodes"] == score_weight_nodes
                and values["turn"] == score_weight_turn
            ), None)
            st.session_state.weight_preset = matched_preset or "사용자 설정"
            st.caption("종합 점수는 위 가중치의 비율로 자동 정규화해 계산합니다.")
            st.markdown(
                _render_weight_preview_card(
                    detour_weight=score_weight_detour,
                    safety_weight=score_weight_safety,
                    time_weight=score_weight_time,
                    nodes_weight=score_weight_nodes,
                    turn_weight=score_weight_turn,
                ),
                unsafe_allow_html=True,
            )
        score_weights = {
            "detour": float(score_weight_detour),
            "safety": float(score_weight_safety),
            "time": float(score_weight_time),
            "nodes": float(score_weight_nodes),
            "turn": float(score_weight_turn),
        }

    with st.container(border=True):
        st.markdown(
            """
            <div class="asv-sidebar-card-heading">출발점/도착점 지정</div>
            <div class="asv-sidebar-card-subtitle">지도에서 클릭할 지점 유형을 선택합니다.</div>
            """,
            unsafe_allow_html=True,
        )
        st.radio(
            "지금 클릭해서 지정할 점",
            (START_TEXT, GOAL_TEXT),
            key="selection_target_widget",
        )
        st.session_state.selection_target = st.session_state.selection_target_widget

        if st.button("초기화", use_container_width=True):
            _clear_points()
            st.rerun()

        st.caption("지도 이미지를 직접 클릭하세요")

    logo_data_uri = _image_to_data_uri(LOGO_PATH)
    logo_markup = (
        f'<div class="asv-sidebar-card-logo"><img src="{logo_data_uri}" alt="국립한국해양대학교 해양공학과 로고"></div>'
        if logo_data_uri
        else ""
    )
    st.markdown(
        f"""
        <div class="asv-sidebar-project-card">
            <div class="asv-sidebar-project-kicker">Project Information</div>
            <div class="asv-sidebar-project-title">Capstone Design 2026</div>
            <div class="asv-sidebar-project-meta">
                <div><span>소속</span>국립한국해양대학교 해양공학과</div>
                <div><span>제작자</span>이영진</div>
                <div><span>지도교수</span>최원석 교수님</div>
            </div>
            <div class="asv-sidebar-links">
                <a href="{STREAMLIT_URL}" target="_blank">Streamlit</a>
                <a href="{GITHUB_URL}" target="_blank">GitHub</a>
            </div>
            {logo_markup}
        </div>
        """,
        unsafe_allow_html=True,
    )

random.seed(int(seed))
np.random.seed(int(seed))

sea_width = 30.0
sea_height = 30.0

safe_min_diameter = min(min_diameter, max_diameter - 0.1)
safe_max_diameter = max(max_diameter, min_diameter + 0.1)
safe_min_vertices = min(min_vertices, max_vertices)
safe_max_vertices = max(max_vertices, min_vertices)

chunks = generate_ice_chunks(
    width=sea_width,
    height=sea_height,
    count=ice_count,
    min_diameter=safe_min_diameter,
    max_diameter=safe_max_diameter,
    min_vertices=safe_min_vertices,
    max_vertices=safe_max_vertices,
    edge_margin=edge_margin,
    min_gap=min_gap,
    max_attempts=160,
)

visible_polygons = get_visible_ice_polygons(sea_width, sea_height, chunks)
removed_message = _validate_saved_points(sea_width, sea_height, visible_polygons)

path_result: Optional[PathResult] = None
path_error_message: Optional[str] = None
total_grid_nodes = _total_grid_node_count(sea_width, sea_height, PATH_CELL_SIZE)
base_path_signature = _build_path_signature(
    seed=int(seed),
    ice_count=int(ice_count),
    min_diameter=safe_min_diameter,
    max_diameter=safe_max_diameter,
    min_vertices=safe_min_vertices,
    max_vertices=safe_max_vertices,
    edge_margin=edge_margin,
    min_gap=min_gap,
    safety_radius=ship_safety_radius,
    start_point=st.session_state.start_point,
    goal_point=st.session_state.goal_point,
)
path_signature = None
selected_algorithm_key = st.session_state.path_algorithm
if base_path_signature is not None and selected_algorithm_key in PATH_ALGORITHMS:
    path_signature = base_path_signature + (selected_algorithm_key,)

if (
    path_signature is not None
    and st.session_state.path_request_signature == path_signature
):
    if st.session_state.path_result_signature != path_signature:
        algorithm_button_label, algorithm_display_name, algorithm_finder = PATH_ALGORITHMS[selected_algorithm_key]
        started_at = time.perf_counter()
        path_timer_placeholder.info("경로 계산용 격자를 생성하는 중입니다... `0.00`초")
        single_open_mask = _build_open_mask(
            sea_width,
            sea_height,
            PATH_CELL_SIZE,
            visible_polygons,
            PATH_CELL_SIZE * 0.42,
        )
        path_timer_placeholder.info(
            f"{algorithm_display_name} 경로 계산 중... `{time.perf_counter() - started_at:.2f}`초"
        )

        def _update_path_progress(visited_nodes: int) -> None:
            elapsed = time.perf_counter() - started_at
            path_timer_placeholder.info(
                f"{algorithm_display_name} \uacbd\ub85c \uacc4\uc0b0 \uc911... `{elapsed:.2f}`\ucd08  |  "
                f"\ubc29\ubb38 \ub178\ub4dc `{visited_nodes}`"
            )

        computed_result = _run_path_algorithm(
            algorithm_key=selected_algorithm_key,
            algorithm_finder=algorithm_finder,
            width=sea_width,
            height=sea_height,
            start_point=st.session_state.start_point,
            goal_point=st.session_state.goal_point,
            visible_polygons=visible_polygons,
            cell_size=PATH_CELL_SIZE,
            safety_radius=ship_safety_radius,
            progress_callback=_update_path_progress,
            open_mask=single_open_mask,
        )
        elapsed_seconds = time.perf_counter() - started_at

        st.session_state.path_result = computed_result
        st.session_state.path_result_signature = path_signature
        st.session_state.last_path_duration = elapsed_seconds

        if st.session_state.comparison_signature != base_path_signature:
            st.session_state.comparison_signature = base_path_signature
            st.session_state.comparison_results = {}
            st.session_state.comparison_errors = {}
        if st.session_state.comparison_results is None:
            st.session_state.comparison_results = {}
        if st.session_state.comparison_errors is None:
            st.session_state.comparison_errors = {}

        if computed_result is None:
            st.session_state.path_error_message = (
                f"{algorithm_display_name} 탐색으로 도착 가능한 경로를 찾지 못했습니다. "
                "출발점/도착점을 다시 지정하거나 해빙 조건을 변경해주세요."
            )
            st.session_state.comparison_errors[selected_algorithm_key] = st.session_state.path_error_message
            path_timer_placeholder.warning(
                f"{algorithm_display_name} \uacc4\uc0b0 \uc644\ub8cc  |  \uc18c\uc694 \uc2dc\uac04 `{elapsed_seconds:.2f}`\ucd08"
            )
        else:
            st.session_state.path_error_message = None
            st.session_state.comparison_results[selected_algorithm_key] = {
                "result": computed_result,
                "duration": elapsed_seconds,
            }
            st.session_state.comparison_errors.pop(selected_algorithm_key, None)
            path_timer_placeholder.success(
                f"{algorithm_display_name} \uacc4\uc0b0 \uc644\ub8cc  |  \uc18c\uc694 \uc2dc\uac04 `{elapsed_seconds:.2f}`\ucd08"
            )

    path_result = st.session_state.path_result
    path_error_message = st.session_state.path_error_message
else:
    path_timer_placeholder.empty()

path_hazard_points = _find_hazard_points(
    path_points=path_result.path_points if path_result is not None else None,
    visible_polygons=visible_polygons,
    safety_radius=ship_safety_radius,
)

figure = render_ice_field(
    width=sea_width,
    height=sea_height,
    chunks=chunks,
    outline_width=outline_width,
    show_grid=show_grid,
    start_point=st.session_state.start_point,
    goal_point=st.session_state.goal_point,
    path_points=path_result.path_points if path_result is not None else None,
    path_hazard_points=path_hazard_points,
)
image_bytes = figure_to_png_bytes(figure)
plt.close(figure)
image = Image.open(BytesIO(image_bytes))

left_col, main_divider_col, right_col = st.columns([1.45, 0.035, 1.15], gap="small")

with left_col:
    st.subheader("해빙 환경 지도")

    map_col_left, map_col_center, map_col_right = st.columns([1, 12, 1])
    with map_col_center:
        click_value = streamlit_image_coordinates(
            image,
            width=image.width,
            key=f"ice_map_clicks_{st.session_state.image_click_key_index}",
        )

    if click_value is not None:
        click_id = (
            int(seed),
            int(ice_count),
            click_value["x"],
            click_value["y"],
        )

        if st.session_state.last_processed_click != click_id:
            sea_point = _image_pixel_to_sea_point(
                click_x=click_value["x"],
                click_y=click_value["y"],
                image_width=image.width,
                image_height=image.height,
                sea_width=sea_width,
                sea_height=sea_height,
            )

            if sea_point is None:
                st.session_state.click_feedback = "해역 테두리 안쪽을 클릭해주세요."
            elif is_open_water_point(sea_point, sea_width, sea_height, visible_polygons):
                if st.session_state.selection_target == START_TEXT:
                    _remember_current_results()
                    st.session_state.start_point = sea_point
                    st.session_state.pending_selection_target = GOAL_TEXT
                    _clear_current_path_results()
                    st.session_state.click_feedback = f"{START_TEXT}이 지정되었습니다."
                else:
                    _remember_current_results()
                    st.session_state.goal_point = sea_point
                    _clear_current_path_results()
                    st.session_state.click_feedback = f"{GOAL_TEXT}이 지정되었습니다."
            else:
                st.session_state.click_feedback = "얼음 위는 선택할 수 없습니다. 빈 바다를 클릭해주세요."

            st.session_state.last_processed_click = click_id
            st.session_state.image_click_key_index += 1
            st.rerun()

    st.caption(
        "빈 바다를 클릭하면 선택한 지점이 지도에 표시됩니다."
    )

    if (
        base_path_signature is not None
        and st.session_state.comparison_signature == base_path_signature
        and st.session_state.comparison_results
    ):
        st.divider()
        st.subheader("\uc54c\uace0\ub9ac\uc998\ubcc4 \uacbd\ub85c \uadf8\ub9bc \ube44\uad50")
        st.caption(
            "전체 알고리즘 비교 실행 후 각 알고리즘이 생성한 경로를 한눈에 비교할 수 있습니다."
        )

        preview_columns = st.columns(2)
        preview_index = 0
        for algorithm_key, (_, algorithm_display_name, _) in PATH_ALGORITHMS.items():
            result_entry = st.session_state.comparison_results.get(algorithm_key)
            if result_entry is None:
                continue

            result = result_entry["result"]
            duration_seconds = result_entry["duration"]
            route_score = _calculate_route_score(
                result=result,
                duration_seconds=duration_seconds,
                safety_radius=ship_safety_radius,
                total_grid_nodes=total_grid_nodes,
                score_weights=score_weights,
            )
            safety_grade, _ = _evaluate_route_safety(result.min_clearance, ship_safety_radius)
            preview_image = _render_path_result_image(
                width=sea_width,
                height=sea_height,
                chunks=chunks,
                outline_width=outline_width,
                show_grid=show_grid,
                start_point=st.session_state.start_point,
                goal_point=st.session_state.goal_point,
                path_result=result,
                visible_polygons=visible_polygons,
                safety_radius=ship_safety_radius,
            )

            with preview_columns[preview_index % 2]:
                st.image(
                    preview_image,
                    caption=(
                        f"{algorithm_display_name} | "
                        f"\uc885\ud569 {route_score:.1f}/100 | "
                        f"\uc548\uc804 {safety_grade} | "
                        f"거리 {_format_distance(result.distance)}"
                    ),
                    use_container_width=True,
                )

            preview_index += 1

    if (
        base_path_signature is not None
        and st.session_state.comparison_signature == base_path_signature
        and st.session_state.comparison_results
    ):
        st.markdown('<div class="asv-section-divider"></div>', unsafe_allow_html=True)
        st.subheader("알고리즘별 수치 비교표")

        comparison_rows = []
        for algorithm_key, (_, algorithm_display_name, _) in PATH_ALGORITHMS.items():
            result_entry = st.session_state.comparison_results.get(algorithm_key)
            if result_entry is None:
                continue

            comparison_rows.append(
                _build_comparison_row(
                    algorithm_name=algorithm_display_name,
                    result=result_entry["result"],
                    duration_seconds=result_entry["duration"],
                    safety_radius=ship_safety_radius,
                    total_grid_nodes=total_grid_nodes,
                    score_weights=score_weights,
                    visible_polygons=visible_polygons,
                )
            )

        st.markdown(_render_comparison_table_html(comparison_rows), unsafe_allow_html=True)

        scored_results = [
            (
                row["\uc54c\uace0\ub9ac\uc998"],
                float(row["\uc885\ud569 \uc810\uc218"].split()[0]),
            )
            for row in comparison_rows
        ]
        if scored_results:
            best_name, best_score = max(scored_results, key=lambda item: item[1])
            st.success(f"\ud604\uc7ac \ud3c9\uac00\uc2dd \uae30\uc900 \ucd5c\uace0 \uc810\uc218: `{best_name}` `{best_score:.1f} / 100`")

        if st.session_state.comparison_errors:
            for error_message in st.session_state.comparison_errors.values():
                st.warning(error_message)

        with st.expander("\uc885\ud569 \uc810\uc218 \ud3c9\uac00 \uae30\uc900"):
            st.write(
                f"\ud604\uc7ac \uac00\uc911\uce58: \uc6b0\ud68c\uc728 `{score_weight_detour}`, \uc548\uc804\uc131 `{score_weight_safety}`, "
                f"\uacc4\uc0b0 \uc2dc\uac04 `{score_weight_time}`, \ubc29\ubb38 \ub178\ub4dc `{score_weight_nodes}`, "
                f"\uad74\uace1 \uc815\ub3c4 `{score_weight_turn}`"
            )
            st.write(
                "\uc885\ud569 \uc810\uc218\ub294 \uc0ac\uc774\ub4dc\ubc14\uc5d0\uc11c \uc870\uc808\ud55c \uac00\uc911\uce58\ub97c \ube44\uc728\ub85c \uc815\uaddc\ud654\ud558\uc5ec "
                "우회율, 안전성, 계산 시간, 방문 노드 수, 굴곡 정도를 함께 반영한 값입니다."
            )
            st.write(
                "\uc774 \uc810\uc218\ub294 \uc54c\uace0\ub9ac\uc998\uc774 \uc548\uc804\uac70\ub9ac\ub97c \uc9c1\uc811 \ucd5c\uc801\ud654\ud588\ub2e4\ub294 \ub73b\uc774 \uc544\ub2c8\ub77c, "
                "같은 해빙 환경에서 생성된 경로를 동일한 기준으로 사후 평가한 값입니다."
            )



with main_divider_col:
    st.markdown('<div class="asv-main-divider"></div>', unsafe_allow_html=True)
with right_col:
    summary_col, right_inner_divider_col, ship_col = st.columns([1.08, 0.035, 1.0], gap="medium")

    with summary_col:
        st.subheader("해빙 설정 요약")
        st.write(f"해역 크기: `{_format_distance(sea_width)} × {_format_distance(sea_height)}`")
        st.write(f"배치 얼음: `{len(chunks)} / {ice_count}개`")
        st.write(f"직경 범위: `{_format_distance_range(safe_min_diameter, safe_max_diameter)}`")
        st.write(f"꼭짓점 수: `{safe_min_vertices} ~ {safe_max_vertices}`")
        st.write(f"최소 간격: `{_format_distance(min_gap)}`")

        if len(chunks) < ice_count:
            st.warning("해역이 너무 복잡해서 요청한 얼음 개수를 모두 배치하지 못했습니다.")
        else:
            st.success("얼음 조각이 정상적으로 배치되었습니다.")


    with right_inner_divider_col:
        st.markdown('<div class="asv-right-divider"></div>', unsafe_allow_html=True)

    with ship_col:
        st.subheader("선박 위치 지정")

        if removed_message is not None:
            st.warning(f"해빙 환경이 변경되어 `{removed_message}`이 자동으로 초기화되었습니다.")

        if st.session_state.click_feedback is not None:
            if "얼음 위" in st.session_state.click_feedback:
                st.warning(st.session_state.click_feedback)
            else:
                st.success(st.session_state.click_feedback)

        if st.session_state.start_point is not None:
            st.write(f"{START_TEXT}: `{_format_coordinate(st.session_state.start_point)}`")
        else:
            st.write(f"{START_TEXT}: 아직 선택되지 않았습니다")

        if st.session_state.goal_point is not None:
            st.write(f"{GOAL_TEXT}: `{_format_coordinate(st.session_state.goal_point)}`")
        else:
            st.write(f"{GOAL_TEXT}: 아직 선택되지 않았습니다")

        if st.session_state.start_point is None:
            st.info("먼저 빈 바다를 클릭해서 출발점을 지정해주세요.")
        elif st.session_state.goal_point is None:
            st.info("이제 빈 바다를 한 번 더 클릭해서 도착점을 지정해주세요.")
        else:
            st.success("출발점과 도착점이 모두 준비되었습니다.")

    st.markdown('<div class="asv-section-divider"></div>', unsafe_allow_html=True)
    st.subheader("알고리즘 실행")

    for algorithm_key, (button_label, _, _) in PATH_ALGORITHMS.items():
        if st.button(button_label, use_container_width=True):
            if base_path_signature is None:
                st.warning(
                    "먼저 출발점과 도착점을 모두 지정해주세요."
                )
            else:
                cached_entry = None
                if (
                    st.session_state.comparison_signature == base_path_signature
                    and st.session_state.comparison_results
                ):
                    cached_entry = st.session_state.comparison_results.get(algorithm_key)

                selected_signature = base_path_signature + (algorithm_key,)
                st.session_state.path_algorithm = algorithm_key
                st.session_state.path_request_signature = selected_signature

                if cached_entry is None:
                    st.session_state.path_result = None
                    st.session_state.path_result_signature = None
                    st.session_state.path_error_message = None
                    st.session_state.last_path_duration = None
                else:
                    st.session_state.path_result = cached_entry["result"]
                    st.session_state.path_result_signature = selected_signature
                    st.session_state.path_error_message = None
                    st.session_state.last_path_duration = cached_entry["duration"]

                st.rerun()

    if (
        base_path_signature is not None
        and st.session_state.comparison_signature == base_path_signature
        and st.session_state.comparison_results
    ):
        st.caption("이미 계산된 결과는 다시 계산하지 않고 바로 불러올 수 있습니다.")
        with st.expander("\uacc4\uc0b0\ub41c \uacb0\uacfc \ubc14\ub85c \ubcf4\uae30", expanded=False):
            for algorithm_key, (_, algorithm_display_name, _) in PATH_ALGORITHMS.items():
                cached_entry = st.session_state.comparison_results.get(algorithm_key)
                if cached_entry is None:
                    continue

                if st.button(
                    f"{algorithm_display_name} \uacb0\uacfc \ubcf4\uae30",
                    key=f"restore_cached_{algorithm_key}",
                    use_container_width=True,
                ):
                    selected_signature = base_path_signature + (algorithm_key,)
                    st.session_state.path_algorithm = algorithm_key
                    st.session_state.path_request_signature = selected_signature
                    st.session_state.path_result = cached_entry["result"]
                    st.session_state.path_result_signature = selected_signature
                    st.session_state.path_error_message = None
                    st.session_state.last_path_duration = cached_entry["duration"]
                    st.rerun()

    if st.button("\uc804\uccb4 \uc54c\uace0\ub9ac\uc998 \ube44\uad50 \uc2e4\ud589", use_container_width=True):
        if base_path_signature is None:
            st.warning(
                "먼저 출발점과 도착점을 모두 지정해주세요."
            )
        else:
            comparison_results = {}
            comparison_errors = {}

            path_timer_placeholder.info("4개 알고리즘 비교용 공통 격자를 생성하는 중입니다... `0.00`초")
            comparison_started_at = time.perf_counter()
            comparison_open_mask = _build_open_mask(
                sea_width,
                sea_height,
                PATH_CELL_SIZE,
                visible_polygons,
                PATH_CELL_SIZE * 0.42,
            )

            path_timer_placeholder.info(
                f"공통 격자 생성 완료  |  소요 시간 `{time.perf_counter() - comparison_started_at:.2f}`초"
            )

            for algorithm_key, (_, algorithm_display_name, algorithm_finder) in PATH_ALGORITHMS.items():
                started_at = time.perf_counter()
                path_timer_placeholder.info(
                    f"{algorithm_display_name} \ube44\uad50\uc6a9 \uacbd\ub85c \uacc4\uc0b0 \uc911... `0.00`\ucd08"
                )

                def _update_compare_progress(visited_nodes: int) -> None:
                    elapsed = time.perf_counter() - started_at
                    path_timer_placeholder.info(
                        f"{algorithm_display_name} \ube44\uad50\uc6a9 \uacbd\ub85c \uacc4\uc0b0 \uc911... `{elapsed:.2f}`\ucd08  |  "
                        f"\ubc29\ubb38 \ub178\ub4dc `{visited_nodes}`"
                    )

                computed_result = _run_path_algorithm(
                    algorithm_key=algorithm_key,
                    algorithm_finder=algorithm_finder,
                    width=sea_width,
                    height=sea_height,
                    start_point=st.session_state.start_point,
                    goal_point=st.session_state.goal_point,
                    visible_polygons=visible_polygons,
                    cell_size=PATH_CELL_SIZE,
                    safety_radius=ship_safety_radius,
                    progress_callback=_update_compare_progress,
                    open_mask=comparison_open_mask,
                )
                elapsed_seconds = time.perf_counter() - started_at

                if computed_result is None:
                    comparison_errors[algorithm_key] = (
                        f"{algorithm_display_name} 탐색으로 경로를 찾지 못했습니다."
                    )
                else:
                    comparison_results[algorithm_key] = {
                        "result": computed_result,
                        "duration": elapsed_seconds,
                    }

            st.session_state.comparison_signature = base_path_signature
            st.session_state.comparison_results = comparison_results
            st.session_state.comparison_errors = comparison_errors

            if comparison_results:
                best_algorithm_key = max(
                    comparison_results,
                    key=lambda key: _calculate_route_score(
                        result=comparison_results[key]["result"],
                        duration_seconds=comparison_results[key]["duration"],
                        safety_radius=ship_safety_radius,
                        total_grid_nodes=total_grid_nodes,
                        score_weights=score_weights,
                    ),
                )
                st.session_state.path_algorithm = best_algorithm_key
                st.session_state.path_request_signature = base_path_signature + (best_algorithm_key,)
                st.session_state.path_result = comparison_results[best_algorithm_key]["result"]
                st.session_state.path_result_signature = st.session_state.path_request_signature
                st.session_state.last_path_duration = comparison_results[best_algorithm_key]["duration"]
                st.session_state.path_error_message = None
            else:
                st.session_state.path_algorithm = None
                st.session_state.path_request_signature = None
                st.session_state.path_result = None
                st.session_state.path_result_signature = None
                st.session_state.last_path_duration = None
                st.session_state.path_error_message = "모든 알고리즘이 경로를 찾지 못했습니다."

            st.rerun()

    has_current_path_result = (
        path_signature is not None
        and st.session_state.path_request_signature == path_signature
        and path_result is not None
    )

    if st.session_state.start_point is None or st.session_state.goal_point is None:
        st.info(
            "출발점과 도착점을 모두 지정한 뒤 원하는 알고리즘 버튼을 누르면 경로를 계산합니다."
        )
    elif path_signature is None or st.session_state.path_request_signature != path_signature:
        st.info(
            "현재 조건에서 경로를 확인하려면 `Dijkstra`, `A*`, `Theta*` 중 하나를 실행해주세요."
        )
    elif path_result is None:
        st.warning(path_error_message)
    else:
        _, algorithm_display_name, _ = PATH_ALGORITHMS[st.session_state.path_algorithm]
        st.success(f"{algorithm_display_name} 기반 경로가 생성되었습니다.")
        st.write(f"경로 길이: `{_format_distance(path_result.distance)}`")
        st.write(f"직선 거리: `{_format_distance(path_result.direct_distance)}`")
        st.write(f"\uc6b0\ud68c\uc728(\uacbd\ub85c/\uc9c1\uc120): `{path_result.detour_ratio:.2f}`")
        st.write(f"\ubc29\ubb38\ud55c \uaca9\uc790 \ub178\ub4dc \uc218: `{path_result.visited_nodes}`")
        st.write(f"경로 격자 간격: `{_format_distance(path_result.cell_size)}`")
        st.write(f"\uad74\uace1 \ud69f\uc218: `{path_result.turn_count}`")
        st.write(f"\ucd1d \ubc29\ud5a5 \ubcc0\ud654\ub7c9: `{path_result.total_turn_angle:.1f}\u00b0`")
        st.write(f"최소 해빙 거리: `{_format_distance(path_result.min_clearance)}`")
        st.write(f"평균 해빙 거리: `{_format_distance(path_result.avg_clearance)}`")
        st.write(f"선박 안전 반경: `{_format_distance(ship_safety_radius)}`")
        safety_grade, safety_status = _evaluate_route_safety(
            path_result.min_clearance,
            ship_safety_radius,
        )
        route_score = _calculate_route_score(
            result=path_result,
            duration_seconds=st.session_state.last_path_duration,
            safety_radius=ship_safety_radius,
            total_grid_nodes=total_grid_nodes,
            score_weights=score_weights,
        )
        safety_message = (
            f"\uc548\uc804 \ub4f1\uae09: `{safety_grade}`  |  "
            f"\uc704\ud5d8 \ud45c\uc2dc \uc9c0\uc810 `{len(path_hazard_points)}`\uac1c"
        )
        if safety_status == "error":
            st.error(safety_message)
        elif safety_status == "warning":
            st.warning(safety_message)
        else:
            st.success(safety_message)
        st.write(f"\uc885\ud569 \uc810\uc218: `{route_score:.1f} / 100`")
        if st.session_state.last_path_duration is not None:
            st.write(f"\uacc4\uc0b0 \uc2dc\uac04: `{st.session_state.last_path_duration:.2f}`\ucd08")

    if (
        base_path_signature is not None
        and st.session_state.comparison_signature == base_path_signature
        and st.session_state.comparison_results
    ):
        st.markdown('<div class="asv-section-divider"></div>', unsafe_allow_html=True)
        st.subheader("안전거리 가중 전후 비교")
        dijkstra_entry = st.session_state.comparison_results.get("dijkstra")
        safe_dijkstra_entry = st.session_state.comparison_results.get("safe_dijkstra")
        if dijkstra_entry is not None and safe_dijkstra_entry is not None:
            dijkstra_result = dijkstra_entry["result"]
            safe_dijkstra_result = safe_dijkstra_entry["result"]
            clearance_delta = safe_dijkstra_result.min_clearance - dijkstra_result.min_clearance
            length_delta = safe_dijkstra_result.distance - dijkstra_result.distance
            hazard_delta = _count_hazard_markers(
                safe_dijkstra_result,
                visible_polygons,
                ship_safety_radius,
            ) - _count_hazard_markers(
                dijkstra_result,
                visible_polygons,
                ship_safety_radius,
            )

            st.write(f"Dijkstra 최소 해빙 거리: `{_units_to_meters(dijkstra_result.min_clearance):.0f} m`")
            st.write(
                f"Safety-Weighted Dijkstra 최소 해빙 거리: `{_units_to_meters(safe_dijkstra_result.min_clearance):.0f} m`"
            )
            st.write(f"안전거리 개선량: `{_units_to_meters(clearance_delta):+.0f} m`")
            st.write(f"경로 길이 증가량: `{_units_to_meters(length_delta):+.0f} m`")
            st.write(f"위험 표시 지점 변화: `{hazard_delta:+d}개`")
        else:
            st.info("Dijkstra와 Safety-Weighted Dijkstra 결과가 모두 있을 때 비교가 표시됩니다.")

    previous_snapshot = st.session_state.previous_path_snapshot
    if not has_current_path_result and previous_snapshot:
        previous_result = previous_snapshot["result"]
        previous_algorithm = previous_snapshot["algorithm_key"]
        _, previous_algorithm_name, _ = PATH_ALGORITHMS[previous_algorithm]
        previous_start_point = previous_snapshot.get("start_point")
        previous_goal_point = previous_snapshot.get("goal_point")
        previous_duration = previous_snapshot.get("duration")
        st.info("현재 선택 중인 조건과 별개로, 바로 직전에 계산한 결과를 보관하고 있습니다.")
        with st.expander("직전 계산 결과 보기", expanded=True):
            st.write(f"알고리즘: `{previous_algorithm_name}`")
            if previous_start_point is not None:
                st.write(f"직전 {START_TEXT}: `{_format_coordinate(previous_start_point)}`")
            if previous_goal_point is not None:
                st.write(f"직전 {GOAL_TEXT}: `{_format_coordinate(previous_goal_point)}`")
            st.write(f"경로 길이: `{_format_distance(previous_result.distance)}`")
            st.write(f"최소 해빙 거리: `{_format_distance(previous_result.min_clearance)}`")
            if previous_duration is not None:
                st.write(f"계산 시간: `{previous_duration:.2f}`초")
            previous_comparison = st.session_state.previous_comparison_snapshot
            if previous_comparison and previous_comparison.get("results"):
                st.caption(f"직전 비교 실행 결과 {len(previous_comparison['results'])}개를 함께 보관하고 있습니다.")
            if st.button("직전 출발점/도착점 다시 불러오기", use_container_width=True):
                _restore_previous_result_snapshot()
                st.rerun()

st.markdown(
    f"""
    <section class="asv-reference-section">
        <div class="asv-reference-kicker">Data Sources & Calculation Basis</div>
        <h2>자료 및 계산 근거</h2>
        <div class="asv-reference-caption">
            본 웹앱에서 사용한 시뮬레이션 환경, 알고리즘 출처, 안전거리 가중식, 프로젝트 정보를 정리한 영역입니다.
        </div>
        <div class="asv-reference-details">
            <details>
                <summary>데이터 및 시뮬레이션 환경</summary>
                <div class="asv-reference-detail-body">
                    <p>본 웹앱의 해빙 환경은 실제 관측자료가 아닌, 북극 해빙 해역을 단순화한 2D 시뮬레이션 환경으로 생성되었습니다.</p>
                    <p>해빙 생성 초기 구현 과정에서 <a href="https://github.com/IvanIZ/predictive-asv-planner" target="_blank">IvanIZ/predictive-asv-planner</a>의 환경 생성 방식을 참고했습니다.</p>
                </div>
            </details>
            <details>
                <summary>경로 탐색 알고리즘 출처</summary>
                <div class="asv-reference-detail-body">
                    <ul>
                        <li>Dijkstra Algorithm: Edsger W. Dijkstra, 1959</li>
                        <li>A* Algorithm: Peter E. Hart, Nils J. Nilsson, Bertram Raphael, 1968</li>
                        <li>Theta* Algorithm: Alex Nash, Kenny Daniel, Sven Koenig, Ariel Felner, 2007</li>
                    </ul>
                </div>
            </details>
            <details>
                <summary>Safety-Weighted Dijkstra 계산식</summary>
                <div class="asv-reference-detail-body">
                    <p>Safety-Weighted Dijkstra는 기존 Dijkstra의 이동 비용에 해빙과의 거리 기반 안전 가중치를 곱하도록 직접 설계한 개선식입니다.</p>
                    <p><code>cost = distance × w(d_ice)</code>, <code>w(d_ice) = 1 + αλ</code></p>
                    <p><code>λ = ((3R - d_ice) / 3R)^2</code>이며, <code>d_ice ≥ 3R</code>인 경우에는 안전 가중치를 적용하지 않습니다.</p>
                    <p><code>d_ice &lt; 0.5R</code> 구간은 해빙과 매우 가까운 고위험 구간으로 보고 추가 비용을 부여합니다.</p>
                </div>
            </details>
            <details>
                <summary>평가 지표 및 제작 정보</summary>
                <div class="asv-reference-detail-body">
                    <p>경로 길이, 계산 시간, 방문 노드 수, 우회율, 굴곡 횟수, 최소 해빙 거리, 위험 표시 지점 수를 기준으로 알고리즘 결과를 비교합니다.</p>
                    <p>제작자: 이영진 · 국립한국해양대학교 해양공학과 · 지도교수: 최원석 교수님 · Capstone Design 2026</p>
                    <p><a href="{STREAMLIT_URL}" target="_blank">Streamlit 웹앱</a> · <a href="{GITHUB_URL}" target="_blank">GitHub 저장소</a></p>
                </div>
            </details>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)
