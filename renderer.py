from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Polygon, Rectangle

from geometry_utils import clip_polygon_to_rect, point_in_polygon
from ice_generator import IceChunk

Point = Tuple[float, float]
SHIP_ICON_PATH = Path(__file__).with_name("boat-ship_12247113.png")
METERS_PER_UNIT = 100.0


def _format_axis_distance(unit_value: float) -> str:
    meters = unit_value * METERS_PER_UNIT
    if meters >= 1000.0:
        return f"{meters / 1000.0:.1f} km"
    return f"{int(round(meters))} m"

def _build_sea_background(width: float, height: float) -> np.ndarray:
    x = np.linspace(0.0, 1.0, 320)
    y = np.linspace(0.0, 1.0, 320)
    xx, yy = np.meshgrid(x, y)

    wave = 0.03 * np.sin(10 * xx + 6 * yy) + 0.02 * np.cos(14 * yy - 4 * xx)

    red = 0.70 - 0.12 * yy + 0.02 * wave
    green = 0.86 - 0.10 * yy + 0.03 * wave
    blue = 0.94 - 0.05 * yy + 0.05 * wave

    background = np.dstack([red, green, blue])
    return np.clip(background, 0.0, 1.0)


def _ice_color(chunk: IceChunk) -> Tuple[float, float, float]:
    base = np.array([0.73, 0.87, 0.91])
    variation = 0.05 * np.sin(chunk.center[0] * 0.7 + chunk.center[1] * 0.35 + chunk.radius)
    color = base + np.array([variation * 0.6, variation * 0.35, variation * 0.15])
    return tuple(np.clip(color, 0.0, 1.0))


def _load_ship_icon() -> Optional[np.ndarray]:
    """
    작업 폴더의 선박 아이콘 이미지를 불러온다.
    """
    if not SHIP_ICON_PATH.exists():
        return None

    return plt.imread(str(SHIP_ICON_PATH))


def _draw_ship_icon(ax, point: Point) -> None:
    """
    출발점 위치에 선박 아이콘을 그린다.
    """
    icon_image = _load_ship_icon()
    if icon_image is None:
        ax.scatter(
            [point[0]],
            [point[1]],
            s=120,
            color="#1d7f4e",
            edgecolors="white",
            linewidths=1.8,
            zorder=5,
        )
        return

    artist = AnnotationBbox(
        OffsetImage(icon_image, zoom=0.048),
        point,
        frameon=False,
        zorder=5,
        box_alignment=(0.5, 0.25),
    )
    ax.add_artist(artist)


def _draw_goal_marker(ax, point: Point) -> None:
    """
    도착점 위치에 빨간 X 표시를 그린다.
    """
    marker_size = 0.275
    ax.plot(
        [point[0] - marker_size, point[0] + marker_size],
        [point[1] - marker_size, point[1] + marker_size],
        color="#d63b3b",
        linewidth=1.8,
        solid_capstyle="round",
        zorder=5,
    )
    ax.plot(
        [point[0] - marker_size, point[0] + marker_size],
        [point[1] + marker_size, point[1] - marker_size],
        color="#d63b3b",
        linewidth=1.8,
        solid_capstyle="round",
        zorder=5,
    )


def get_visible_ice_polygons(
    width: float,
    height: float,
    chunks: List[IceChunk],
) -> List[np.ndarray]:
    polygons: List[np.ndarray] = []

    for chunk in chunks:
        clipped = clip_polygon_to_rect(chunk.points, 0.0, 0.0, width, height)
        if len(clipped) >= 3:
            polygons.append(clipped)

    return polygons


def is_open_water_point(
    point: Point,
    width: float,
    height: float,
    visible_polygons: List[np.ndarray],
) -> bool:
    x_pos, y_pos = point

    if not (0.0 <= x_pos <= width and 0.0 <= y_pos <= height):
        return False

    for polygon in visible_polygons:
        if point_in_polygon(point, polygon):
            return False

    return True


def render_ice_field(
    width: float,
    height: float,
    chunks: List[IceChunk],
    outline_width: float,
    show_grid: bool,
    start_point: Optional[Point] = None,
    goal_point: Optional[Point] = None,
    path_points: Optional[List[Point]] = None,
    path_hazard_points: Optional[List[Point]] = None,
) -> Figure:
    fig, ax = plt.subplots(figsize=(5.4, 5.4), dpi=120)
    fig.patch.set_facecolor("#9dc7e6")
    ax.set_facecolor("#9dc7e6")
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

    sea_background = _build_sea_background(width, height)
    ax.imshow(
        sea_background,
        extent=(0, width, 0, height),
        origin="lower",
        interpolation="bilinear",
        zorder=0,
    )

    for chunk in chunks:
        visible_points = clip_polygon_to_rect(chunk.points, 0.0, 0.0, width, height)
        if len(visible_points) < 3:
            continue

        shadow_points = clip_polygon_to_rect(
            visible_points + np.array([0.18, -0.18]),
            0.0,
            0.0,
            width,
            height,
        )
        if len(shadow_points) >= 3:
            shadow_patch = Polygon(
                shadow_points,
                closed=True,
                facecolor=(0.19, 0.33, 0.39, 0.16),
                edgecolor=(0, 0, 0, 0),
                linewidth=0,
                joinstyle="round",
                zorder=1,
            )
            ax.add_patch(shadow_patch)

        ice_patch = Polygon(
            visible_points,
            closed=True,
            facecolor=_ice_color(chunk),
            edgecolor="#62727b",
            linewidth=outline_width,
            joinstyle="round",
            zorder=2,
        )
        ax.add_patch(ice_patch)

    sea_border = Rectangle(
        (0, 0),
        width,
        height,
        facecolor=(1, 1, 1, 0),
        edgecolor="#2f4c57",
        linewidth=1.8,
        zorder=3,
    )
    ax.add_patch(sea_border)

    if path_points is not None and len(path_points) >= 2:
        path_x = [point[0] for point in path_points]
        path_y = [point[1] for point in path_points]
        ax.plot(
            path_x,
            path_y,
            color="#f24f4f",
            linewidth=2.4,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=4,
        )

    if path_hazard_points is not None:
        for hazard_point in path_hazard_points:
            ax.text(
                hazard_point[0],
                hazard_point[1],
                "!",
                ha="center",
                va="center",
                fontsize=5,
                fontweight="bold",
                color="#7a1e12",
                zorder=6,
                bbox={
                    "boxstyle": "circle,pad=0.09",
                    "facecolor": "#ffd44d",
                    "edgecolor": "#d63b3b",
                    "linewidth": 0.8,
                },
            )

    if start_point is not None:
        _draw_ship_icon(ax, start_point)

    if goal_point is not None:
        _draw_goal_marker(ax, goal_point)

    if show_grid:
        major_x_ticks = np.arange(0, width + 0.1, 5)
        major_y_ticks = np.arange(0, height + 0.1, 5)
        minor_x_ticks = np.arange(0, width + 0.1, 1)
        minor_y_ticks = np.arange(0, height + 0.1, 1)

        ax.set_xticks(major_x_ticks)
        ax.set_yticks(major_y_ticks)
        ax.set_xticks(minor_x_ticks, minor=True)
        ax.set_yticks(minor_y_ticks, minor=True)
        ax.set_xticklabels([_format_axis_distance(value) for value in major_x_ticks])
        ax.set_yticklabels([_format_axis_distance(value) for value in major_y_ticks])
        ax.grid(which="minor", color=(1, 1, 1, 0.24), linewidth=0.45)
        ax.grid(which="major", color=(1, 1, 1, 0.38), linewidth=0.7)
        ax.tick_params(
            axis="both",
            which="both",
            length=0,
            labelbottom=True,
            labelleft=True,
            labeltop=False,
            labelright=False,
            labelsize=8,
            colors="#46668f",
            pad=4,
        )
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_xlim(0.0, width)
    ax.set_ylim(0.0, height)
    ax.set_aspect("equal")

    for spine in ax.spines.values():
        spine.set_visible(False)

    return fig


def figure_to_png_bytes(figure: Figure) -> bytes:
    """
    matplotlib figure를 클릭 가능한 이미지로 쓰기 위해 PNG 바이트로 변환한다.
    """
    buffer = BytesIO()
    figure.canvas.draw()
    figure.savefig(buffer, format="png", dpi=120)
    buffer.seek(0)
    return buffer.getvalue()

