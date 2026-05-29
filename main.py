import html
import random
import time
from io import BytesIO
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

TITLE = "\ubd81\uadf9 \ud574\ube59 \uc0dd\uc131\uae30"
CAPTION = (
    "\ud574\ube59 \ud658\uacbd\uc744 \uc0dd\uc131\ud558\uace0, \ubc14\ub2e4 \uc704\ub97c "
    "\ud074\ub9ad\ud574 \uc120\ubc15\uc758 \ucd9c\ubc1c\uc810\uacfc \ub3c4\ucc29\uc810\uc744 \uc9c0\uc815\ud558\uace0, "
    "\uacbd\ub85c \ud0d0\uc0c9 \uc54c\uace0\ub9ac\uc998\ubcc4 \uacb0\uacfc\ub97c \ube44\uad50\ud560 \uc218 \uc788\ub2e4."
)
START_TEXT = "\ucd9c\ubc1c\uc810"
GOAL_TEXT = "\ub3c4\ucc29\uc810"
PATH_CELL_SIZE = 0.3
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
            background: radial-gradient(circle at top right, rgba(75, 136, 255, 0.10), transparent 20%), linear-gradient(180deg, #f8fbff 0%, #ffffff 24%, #ffffff 100%);
            color: var(--ppt-text);
        }
        [data-testid="stAppViewContainer"] > .main { background: transparent; }
        .block-container { padding-top: 2.1rem; padding-bottom: 2.2rem; }
        [data-testid="stSidebar"] { background: linear-gradient(180deg, #f8fbff 0%, #f1f7ff 100%); border-right: 1px solid var(--ppt-border); }
        [data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
        h1, h2, h3 { color: var(--ppt-navy) !important; letter-spacing: -0.02em; }
        h1 { font-weight: 800 !important; }
        h2, h3 { font-weight: 760 !important; }
        .stCaption, .stMarkdown p, .stMarkdown li, .stText, label { color: var(--ppt-text); }
        .stCaption { color: var(--ppt-muted) !important; }
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { font-size: 1.05rem !important; margin-top: 0.15rem; }
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
        .ppt-comparison-title { margin-bottom: 0.9rem; }
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

    return ('<div class="ppt-comparison-card">' '<div class="ppt-comparison-title">알고리즘 성능 비교표</div>' '<div class="ppt-comparison-scroll"><table class="ppt-comparison-table">' f"<thead><tr>{head_html}</tr></thead>" f"<tbody>{''.join(body_rows)}</tbody>" '</table></div>' '</div>')
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
) -> Optional[PathResult]:
    common_kwargs = {
        "width": width,
        "height": height,
        "start_point": start_point,
        "goal_point": goal_point,
        "visible_polygons": visible_polygons,
        "cell_size": cell_size,
        "progress_callback": progress_callback,
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


def _clear_points() -> None:
    st.session_state.selection_target = START_TEXT
    st.session_state.start_point = None
    st.session_state.goal_point = None
    st.session_state.last_processed_click = None
    st.session_state.click_feedback = None
    st.session_state.pending_selection_target = START_TEXT
    st.session_state.path_request_signature = None
    st.session_state.path_result = None
    st.session_state.path_result_signature = None
    st.session_state.path_error_message = None
    st.session_state.last_path_duration = None
    st.session_state.path_algorithm = None
    st.session_state.comparison_signature = None
    st.session_state.comparison_results = None
    st.session_state.comparison_errors = None
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
        st.session_state.path_request_signature = None
        st.session_state.path_result = None
        st.session_state.path_result_signature = None
        st.session_state.path_error_message = None
        st.session_state.last_path_duration = None
        st.session_state.path_algorithm = None
        st.session_state.comparison_signature = None
        st.session_state.comparison_results = None
        st.session_state.comparison_errors = None
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

st.title(TITLE)
st.caption(CAPTION)
path_timer_placeholder = st.empty()

with st.sidebar:
    st.header("파라미터 설정")

    seed = st.number_input("랜덤 배치", min_value=0, value=42, step=1)
    ice_count = st.slider("얼음 개수", min_value=20, max_value=220, value=85, step=5)

    min_diameter_m = st.slider("최소 직경 (m)", min_value=40, max_value=300, value=80, step=10)
    max_diameter_m = st.slider("최대 직경 (m)", min_value=80, max_value=500, value=180, step=10)
    min_diameter = _meters_to_units(min_diameter_m)
    max_diameter = _meters_to_units(max_diameter_m)

    min_vertices = st.slider(
        "최소 꼭짓점 수", min_value=6, max_value=12, value=8, step=1
    )
    max_vertices = st.slider(
        "최대 꼭짓점 수", min_value=8, max_value=24, value=14, step=1
    )
    edge_margin = DEFAULT_EDGE_MARGIN
    min_gap_m = st.slider("최소 간격 (m)", min_value=0, max_value=150, value=10, step=5)
    min_gap = _meters_to_units(min_gap_m)
    outline_width = DEFAULT_OUTLINE_WIDTH
    show_grid = st.checkbox("격자 표시", value=False, help="체크하면 30×30 격자를 표시한다.")

    st.divider()
    st.header("선박 안전 설정")
    ship_safety_radius_m = st.slider(
        "선박 안전 반경 (m)",
        min_value=10,
        max_value=200,
        value=100,
        step=10,
    )
    ship_safety_radius = _meters_to_units(ship_safety_radius_m)
    risk_threshold_m = int(round(ship_safety_radius_m * 0.5))
    st.caption(
        f"최소 해빙 거리가 `{risk_threshold_m} m` 미만이면 `위험`, "
        f"`{risk_threshold_m}~{ship_safety_radius_m} m` 구간이면 `주의`, "
        f"`{ship_safety_radius_m} m` 이상이면 `안전`으로 평가한다."
    )
    st.markdown(_render_safety_radius_preview_card(ship_safety_radius_m), unsafe_allow_html=True)

    st.divider()
    st.header("평가 가중치 설정")
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
    score_weights = {
        "detour": float(score_weight_detour),
        "safety": float(score_weight_safety),
        "time": float(score_weight_time),
        "nodes": float(score_weight_nodes),
        "turn": float(score_weight_turn),
    }
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
    st.caption(
        "종합 점수는 위 가중치의 비율로 자동 정규화해 계산한다."
    )
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


    st.divider()
    st.header("출발점/도착점 지정")
    st.radio(
        "지금 클릭해서 지정할 점",
        (START_TEXT, GOAL_TEXT),
        key="selection_target_widget",
    )
    st.session_state.selection_target = st.session_state.selection_target_widget

    if st.button("초기화", use_container_width=True):
        _clear_points()
        st.rerun()

    st.caption("지도 이미지를 직접 클릭")


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
        path_timer_placeholder.info(
            f"{algorithm_display_name} \uacbd\ub85c \uacc4\uc0b0 \uc911... `0.00`\ucd08"
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
                f"{algorithm_display_name} \ud0d0\uc0c9\uc73c\ub85c \ub3c4\ucc29 \uac00\ub2a5\ud55c \uacbd\ub85c\ub97c \ucc3e\uc9c0 \ubabb\ud588\ub2e4. "
                "\ucd9c\ubc1c\uc810/\ub3c4\ucc29\uc810\uc744 \ub2e4\uc2dc \uc9c0\uc815\ud558\uac70\ub098 \ud574\ube59 \uc870\uac74\uc744 \ubc14\uafd4\ubd10."
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

left_col, right_col = st.columns([2.7, 1.0])

with left_col:
    st.subheader("\ud574\ube59 \ubd84\ud3ec")

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
                st.session_state.click_feedback = "\ud574\uc5ed \ud14c\ub450\ub9ac \uc548\ucabd\uc744 \ud074\ub9ad\ud574\uc918."
            elif is_open_water_point(sea_point, sea_width, sea_height, visible_polygons):
                if st.session_state.selection_target == START_TEXT:
                    st.session_state.start_point = sea_point
                    st.session_state.pending_selection_target = GOAL_TEXT
                    st.session_state.path_request_signature = None
                    st.session_state.path_result = None
                    st.session_state.path_result_signature = None
                    st.session_state.path_error_message = None
                    st.session_state.last_path_duration = None
                    st.session_state.path_algorithm = None
                    st.session_state.comparison_signature = None
                    st.session_state.comparison_results = None
                    st.session_state.comparison_errors = None
                    st.session_state.click_feedback = f"{START_TEXT}\uc774 \uc9c0\uc815\ub418\uc5c8\ub2e4."
                else:
                    st.session_state.goal_point = sea_point
                    st.session_state.path_request_signature = None
                    st.session_state.path_result = None
                    st.session_state.path_result_signature = None
                    st.session_state.path_error_message = None
                    st.session_state.last_path_duration = None
                    st.session_state.path_algorithm = None
                    st.session_state.comparison_signature = None
                    st.session_state.comparison_results = None
                    st.session_state.comparison_errors = None
                    st.session_state.click_feedback = f"{GOAL_TEXT}\uc774 \uc9c0\uc815\ub418\uc5c8\ub2e4."
            else:
                st.session_state.click_feedback = "\uc5bc\uc74c \uc704\ub294 \uc120\ud0dd\ud560 \uc218 \uc5c6\ub2e4. \ube48 \ubc14\ub2e4\ub97c \ud074\ub9ad\ud574\uc918."

            st.session_state.last_processed_click = click_id
            st.session_state.image_click_key_index += 1
            st.rerun()

    st.caption(
        "\ube48 \ubc14\ub2e4\ub97c \ud074\ub9ad\ud558\uba74 \uc120\ud0dd\ud55c \uc810\uc774 \uc9c0\ub3c4\uc5d0 \ud45c\uc2dc\ub41c\ub2e4."
    )

    if (
        base_path_signature is not None
        and st.session_state.comparison_signature == base_path_signature
        and st.session_state.comparison_results
    ):
        st.divider()
        st.subheader("\uc54c\uace0\ub9ac\uc998\ubcc4 \uacbd\ub85c \uadf8\ub9bc \ube44\uad50")
        st.caption(
            "\uc804\uccb4 \uc54c\uace0\ub9ac\uc998 \ube44\uad50 \uc2e4\ud589 \ud6c4 \uac01 \uc54c\uace0\ub9ac\uc998\uc774 \uc0dd\uc131\ud55c \uacbd\ub85c\ub97c \ud55c\ub208\uc5d0 \ube44\uad50\ud560 \uc218 \uc788\ub2e4."
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

with right_col:
    st.subheader("\uc694\uc57d")
    st.write(f"해역 크기: `{_format_distance(sea_width)} × {_format_distance(sea_height)}`")
    st.write(f"\uc694\uccad\ud55c \uc5bc\uc74c \uac1c\uc218: `{ice_count}`")
    st.write(f"\ubc30\uce58\ub41c \uc5bc\uc74c \uac1c\uc218: `{len(chunks)}`")
    st.write(f"직경 범위: `{_format_distance_range(safe_min_diameter, safe_max_diameter)}`")
    st.write(f"\uaf2d\uc9d3\uc810 \uc218 \ubc94\uc704: `{safe_min_vertices} ~ {safe_max_vertices}`")
    st.write(f"최소 간격: `{_format_distance(min_gap)}`")

    if len(chunks) < ice_count:
        st.warning(
            "\ud574\uc5ed\uc774 \ub108\ubb34 \ubcf5\uc7a1\ud574\uc11c \uc694\uccad\ud55c \uc5bc\uc74c \uac1c\uc218\ub97c \ubaa8\ub450 \ubc30\uce58\ud558\uc9c0 \ubabb\ud588\ub2e4."
        )
    else:
        st.success("\uc5bc\uc74c \uc870\uac01\uc774 \uc815\uc0c1\uc801\uc73c\ub85c \ubc30\uce58\ub418\uc5c8\ub2e4.")

    st.divider()
    st.subheader("\uc120\ubc15 \uc704\uce58")

    if removed_message is not None:
        st.warning(
            f"\ud574\ube59 \ud658\uacbd\uc774 \ubc14\ub00c\uc5b4\uc11c `{removed_message}`\uc774 \uc790\ub3d9\uc73c\ub85c \ucd08\uae30\ud654\ub418\uc5c8\ub2e4."
        )

    if st.session_state.click_feedback is not None:
        if "\uc5bc\uc74c \uc704" in st.session_state.click_feedback:
            st.warning(st.session_state.click_feedback)
        else:
            st.success(st.session_state.click_feedback)

    if st.session_state.start_point is not None:
        st.write(
            f"{START_TEXT}: `{_format_coordinate(st.session_state.start_point)}`"
        )
    else:
        st.write(f"{START_TEXT}: \uc544\uc9c1 \uc120\ud0dd\ub418\uc9c0 \uc54a\uc74c")

    if st.session_state.goal_point is not None:
        st.write(
            f"{GOAL_TEXT}: `{_format_coordinate(st.session_state.goal_point)}`"
        )
    else:
        st.write(f"{GOAL_TEXT}: \uc544\uc9c1 \uc120\ud0dd\ub418\uc9c0 \uc54a\uc74c")

    if st.session_state.start_point is None:
        st.info(
            "\uba3c\uc800 \ube48 \ubc14\ub2e4\ub97c \ud074\ub9ad\ud574\uc11c \ucd9c\ubc1c\uc810\uc744 \uc9c0\uc815\ud558\uba74 \ub41c\ub2e4."
        )
    elif st.session_state.goal_point is None:
        st.info(
            "\uc774\uc81c \ube48 \ubc14\ub2e4\ub97c \ud55c \ubc88 \ub354 \ud074\ub9ad\ud574\uc11c \ub3c4\ucc29\uc810\uc744 \uc9c0\uc815\ud558\uba74 \ub41c\ub2e4."
        )
    else:
        st.success("\ucd9c\ubc1c\uc810\uacfc \ub3c4\ucc29\uc810\uc774 \ubaa8\ub450 \uc900\ube44\ub418\uc5c8\ub2e4.")

    st.divider()
    st.subheader("\uacbd\ub85c \uacb0\uacfc")

    for algorithm_key, (button_label, _, _) in PATH_ALGORITHMS.items():
        if st.button(button_label, use_container_width=True):
            if base_path_signature is None:
                st.warning(
                    "\uba3c\uc800 \ucd9c\ubc1c\uc810\uacfc \ub3c4\ucc29\uc810\uc744 \ubaa8\ub450 \uc9c0\uc815\ud574\uc918."
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
        st.caption("\uc774\ubbf8 \uacc4\uc0b0\ub41c \uacb0\uacfc\ub294 \ub2e4\uc2dc \uacc4\uc0b0\ud558\uc9c0 \uc54a\uace0 \ubc14\ub85c \ubd88\ub7ec\uc62c \uc218 \uc788\ub2e4.")
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
                "\uba3c\uc800 \ucd9c\ubc1c\uc810\uacfc \ub3c4\ucc29\uc810\uc744 \ubaa8\ub450 \uc9c0\uc815\ud574\uc918."
            )
        else:
            comparison_results = {}
            comparison_errors = {}

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
                )
                elapsed_seconds = time.perf_counter() - started_at

                if computed_result is None:
                    comparison_errors[algorithm_key] = (
                        f"{algorithm_display_name} \ud0d0\uc0c9\uc73c\ub85c \uacbd\ub85c\ub97c \ucc3e\uc9c0 \ubabb\ud588\ub2e4."
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
                st.session_state.path_error_message = "\ubaa8\ub4e0 \uc54c\uace0\ub9ac\uc998\uc774 \uacbd\ub85c\ub97c \ucc3e\uc9c0 \ubabb\ud588\ub2e4."

            st.rerun()

    if st.session_state.start_point is None or st.session_state.goal_point is None:
        st.info(
            "\ucd9c\ubc1c\uc810\uacfc \ub3c4\ucc29\uc810\uc744 \ubaa8\ub450 \uc9c0\uc815\ud55c \ub4a4 \uc6d0\ud558\ub294 \uc54c\uace0\ub9ac\uc998 \ubc84\ud2bc\uc744 \ub204\ub974\uba74 \uacbd\ub85c\ub97c \uacc4\uc0b0\ud55c\ub2e4."
        )
    elif path_signature is None or st.session_state.path_request_signature != path_signature:
        st.info(
            "\uc9c0\uae08 \uc870\uac74\uc5d0\uc11c \uacbd\ub85c\ub97c \ubcf4\ub824\uba74 `Dijkstra`, `A*`, `Theta*` \uc911 \ud558\ub098\ub97c \uc2e4\ud589\ud558\uba74 \ub41c\ub2e4."
        )
    elif path_result is None:
        st.warning(path_error_message)
    else:
        _, algorithm_display_name, _ = PATH_ALGORITHMS[st.session_state.path_algorithm]
        st.success(f"{algorithm_display_name} \uae30\ubc18 \uacbd\ub85c\uac00 \uc0dd\uc131\ub418\uc5c8\ub2e4.")
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
        st.divider()
        st.subheader("\uc54c\uace0\ub9ac\uc998\ubcc4 \ube44\uad50\ud45c")

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

            st.subheader("\uc548\uc804\uac00\uc911 \uc801\uc6a9 \uc804\ud6c4 \ube44\uad50")
            st.write(f"Dijkstra \ucd5c\uc18c \ud574\ube59 \uac70\ub9ac: `{dijkstra_result.min_clearance:.2f}`")
            st.write(
                f"Safety-Weighted Dijkstra \ucd5c\uc18c \ud574\ube59 \uac70\ub9ac: `{safe_dijkstra_result.min_clearance:.2f}`"
            )
            st.write(f"\uc548\uc804\uac70\ub9ac \uac1c\uc120\ub7c9: `{clearance_delta:+.2f}`")
            st.write(f"\uacbd\ub85c \uae38\uc774 \uc99d\uac00\ub7c9: `{length_delta:+.2f}`")
            st.write(f"\uc704\ud5d8 \ud45c\uc2dc \uc9c0\uc810 \ubcc0\ud654: `{hazard_delta:+d}`")

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
                "\uc6b0\ud68c\uc728, \uc548\uc804\uc131, \uacc4\uc0b0 \uc2dc\uac04, \ubc29\ubb38 \ub178\ub4dc \uc218, \uad74\uace1 \uc815\ub3c4\ub97c \ud568\uaed8 \ubc18\uc601\ud55c \uac12\uc774\ub2e4."
            )
            st.write(
                "\uc774 \uc810\uc218\ub294 \uc54c\uace0\ub9ac\uc998\uc774 \uc548\uc804\uac70\ub9ac\ub97c \uc9c1\uc811 \ucd5c\uc801\ud654\ud588\ub2e4\ub294 \ub73b\uc774 \uc544\ub2c8\ub77c, "
                "\uac19\uc740 \ud574\ube59 \ud658\uacbd\uc5d0\uc11c \uc0dd\uc131\ub41c \uacbd\ub85c\ub97c \ub3d9\uc77c\ud55c \uae30\uc900\uc73c\ub85c \uc0ac\ud6c4 \ud3c9\uac00\ud55c \uac12\uc774\ub2e4."
            )








