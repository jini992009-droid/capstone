from collections import deque
import heapq
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from geometry_utils import point_in_polygon, polygon_bbox, point_to_polygon_distance, segment_intersects_polygon

Point = Tuple[float, float]
GridCell = Tuple[int, int]
PolygonBBox = Tuple[float, float, float, float]
PROGRESS_UPDATE_INTERVAL = 600


@dataclass
class PathResult:
    """
    다익스트라 탐색 결과를 담는 자료형이다.

    - `path_points`: 화면에 그릴 실제 경로 좌표들
    - `distance`: 경로 길이
    - `visited_nodes`: 탐색 중 방문한 노드 수
    - `cell_size`: 격자 한 칸의 크기
    """

    path_points: List[Point]
    distance: float
    visited_nodes: int
    cell_size: float
    direct_distance: float
    detour_ratio: float
    turn_count: int
    total_turn_angle: float
    min_clearance: float
    avg_clearance: float


def _grid_shape(width: float, height: float, cell_size: float) -> Tuple[int, int]:
    """
    해역 크기를 격자로 나눴을 때의 열/행 개수를 구한다.
    """
    cols = max(1, int(math.ceil(width / cell_size)))
    rows = max(1, int(math.ceil(height / cell_size)))
    return cols, rows


def _cell_center(cell: GridCell, width: float, height: float, cell_size: float) -> Point:
    """
    격자 칸 번호를 실제 해역 좌표의 중심점으로 바꾼다.
    """
    col, row = cell
    x_pos = min(width - cell_size * 0.5, (col + 0.5) * cell_size)
    y_pos = min(height - cell_size * 0.5, (row + 0.5) * cell_size)
    return (x_pos, y_pos)


def _point_to_cell(point: Point, width: float, height: float, cell_size: float) -> GridCell:
    """
    실제 해역 좌표를 가장 가까운 격자 칸으로 바꾼다.
    """
    cols, rows = _grid_shape(width, height, cell_size)
    x_pos = min(max(point[0], 0.0), width - 1e-9)
    y_pos = min(max(point[1], 0.0), height - 1e-9)
    col = min(cols - 1, max(0, int(x_pos / cell_size)))
    row = min(rows - 1, max(0, int(y_pos / cell_size)))
    return (col, row)


def _is_blocked(
    cell: GridCell,
    width: float,
    height: float,
    cell_size: float,
    visible_polygons: List[np.ndarray],
    clearance: float,
    polygon_bboxes: List[PolygonBBox],
) -> bool:
    """
    격자 칸 하나가 얼음에 막혀 있는지 검사한다.

    중심점 하나만 보지 않고 주변 몇 점도 같이 검사해서,
    경로가 얼음 가장자리에 너무 바짝 붙지 않게 만든다.
    """
    center_x, center_y = _cell_center(cell, width, height, cell_size)

    sample_offsets = [
        (0.0, 0.0),
        (clearance, 0.0),
        (-clearance, 0.0),
        (0.0, clearance),
        (0.0, -clearance),
    ]

    for offset_x, offset_y in sample_offsets:
        point = (center_x + offset_x, center_y + offset_y)

        if not (0.0 <= point[0] <= width and 0.0 <= point[1] <= height):
            return True

        for polygon, bbox in zip(visible_polygons, polygon_bboxes):
            min_x, min_y, max_x, max_y = bbox
            if point[0] < min_x or point[0] > max_x or point[1] < min_y or point[1] > max_y:
                continue
            if point_in_polygon(point, polygon):
                return True

    return False


def _build_open_mask(
    width: float,
    height: float,
    cell_size: float,
    visible_polygons: List[np.ndarray],
    clearance: float,
) -> np.ndarray:
    """
    격자 전체에서 이동 가능한 칸(True) / 막힌 칸(False)을 만든다.
    """
    cols, rows = _grid_shape(width, height, cell_size)
    open_mask = np.ones((rows, cols), dtype=bool)
    polygon_bboxes = _build_polygon_bboxes(visible_polygons)

    for row in range(rows):
        for col in range(cols):
            if _is_blocked((col, row), width, height, cell_size, visible_polygons, clearance, polygon_bboxes):
                open_mask[row, col] = False

    return open_mask



def _build_polygon_bboxes(visible_polygons: List[np.ndarray]) -> List[PolygonBBox]:
    return [polygon_bbox(polygon) for polygon in visible_polygons]

def _find_nearest_open_cell(target: GridCell, open_mask: np.ndarray) -> Optional[GridCell]:
    """
    시작점/도착점이 막힌 칸으로 들어가면, 가장 가까운 열린 칸을 찾는다.
    """
    cols = open_mask.shape[1]
    rows = open_mask.shape[0]

    if open_mask[target[1], target[0]]:
        return target

    visited = set()
    queue = deque([target])
    visited.add(target)

    while queue:
        col, row = queue.popleft()

        for delta_col, delta_row in (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ):
            next_col = col + delta_col
            next_row = row + delta_row
            neighbor = (next_col, next_row)

            if not (0 <= next_col < cols and 0 <= next_row < rows):
                continue
            if neighbor in visited:
                continue
            if open_mask[next_row, next_col]:
                return neighbor

            visited.add(neighbor)
            queue.append(neighbor)

    return None


def _segment_hits_any_polygon(
    start: Point,
    end: Point,
    visible_polygons: List[np.ndarray],
    polygon_bboxes: Optional[List[PolygonBBox]] = None,
) -> bool:
    """
    선분이 얼음 다각형을 지나가는지 검사한다.
    """
    if polygon_bboxes is None:
        polygon_bboxes = _build_polygon_bboxes(visible_polygons)

    segment_bbox = (
        min(start[0], end[0]),
        min(start[1], end[1]),
        max(start[0], end[0]),
        max(start[1], end[1]),
    )

    for polygon, bbox in zip(visible_polygons, polygon_bboxes):
        min_x, min_y, max_x, max_y = bbox
        if segment_bbox[2] < min_x or max_x < segment_bbox[0] or segment_bbox[3] < min_y or max_y < segment_bbox[1]:
            continue
        if segment_intersects_polygon(start, end, polygon):
            return True
    return False


def _reconstruct_path(
    previous: Dict[GridCell, GridCell],
    start_cell: GridCell,
    goal_cell: GridCell,
) -> List[GridCell]:
    """
    다익스트라 탐색이 끝난 뒤 실제 격자 경로를 복원한다.
    """
    cells = [goal_cell]
    current = goal_cell

    while current != start_cell:
        current = previous[current]
        cells.append(current)

    cells.reverse()
    return cells


def _compute_polyline_length(path_points: List[Point]) -> float:
    """
    여러 점으로 이루어진 경로의 전체 길이를 계산한다.
    """
    if len(path_points) < 2:
        return 0.0

    total = 0.0
    for index in range(len(path_points) - 1):
        x1, y1 = path_points[index]
        x2, y2 = path_points[index + 1]
        total += math.hypot(x2 - x1, y2 - y1)
    return total


def _compute_turn_metrics(path_points: List[Point]) -> Tuple[int, float]:
    """
    경로의 굴곡 정도를 간단한 두 지표로 계산한다.

    - turn_count: 실제로 꺾인 횟수
    - total_turn_angle: 모든 방향 변화량의 합(도 단위)
    """
    if len(path_points) < 3:
        return 0, 0.0

    turn_count = 0
    total_turn_angle = 0.0

    for index in range(1, len(path_points) - 1):
        x1, y1 = path_points[index - 1]
        x2, y2 = path_points[index]
        x3, y3 = path_points[index + 1]

        angle_a = math.atan2(y2 - y1, x2 - x1)
        angle_b = math.atan2(y3 - y2, x3 - x2)
        turn_angle = abs(math.degrees(angle_b - angle_a))

        if turn_angle > 180.0:
            turn_angle = 360.0 - turn_angle

        if turn_angle >= 5.0:
            turn_count += 1
            total_turn_angle += turn_angle

    return turn_count, total_turn_angle


def _sample_path_points(path_points: List[Point]) -> List[Point]:
    """
    회피 양상 계산을 위해 경로 위의 점들을 조금 더 촘촘하게 샘플링한다.
    """
    if len(path_points) < 2:
        return path_points[:]

    sampled: List[Point] = []
    for index in range(len(path_points) - 1):
        start = path_points[index]
        end = path_points[index + 1]
        sampled.append(start)
        midpoint = ((start[0] + end[0]) * 0.5, (start[1] + end[1]) * 0.5)
        sampled.append(midpoint)

    sampled.append(path_points[-1])
    return sampled


def _compute_clearance_metrics(
    path_points: List[Point],
    visible_polygons: List[np.ndarray],
) -> Tuple[float, float]:
    """
    경로가 얼음에서 얼마나 떨어져 있는지 계산한다.

    - min_clearance: 경로 중 가장 위험하게 가까웠던 최소 거리
    - avg_clearance: 전체적으로 평균 얼마만큼 떨어져 이동했는지
    """
    sampled_points = _sample_path_points(path_points)

    if not sampled_points or not visible_polygons:
        return float("inf"), float("inf")

    clearances: List[float] = []
    for point in sampled_points:
        nearest_distance = min(
            point_to_polygon_distance(point, polygon) for polygon in visible_polygons
        )
        clearances.append(nearest_distance)

    return min(clearances), sum(clearances) / len(clearances)


def find_dijkstra_path(
    width: float,
    height: float,
    start_point: Point,
    goal_point: Point,
    visible_polygons: List[np.ndarray],
    cell_size: float = 0.5,
    progress_callback: Optional[Callable[[int], None]] = None,
    open_mask: Optional[np.ndarray] = None,
) -> Optional[PathResult]:
    """
    다익스트라 알고리즘으로 출발점에서 도착점까지의 경로를 계산한다.

    구현 순서:
    1. 해역을 일정 간격의 격자로 나눈다.
    2. 얼음이 있는 칸은 이동 불가로 처리한다.
    3. 열린 칸들만 이용해 다익스트라 탐색을 수행한다.
    4. 찾은 격자 경로를 실제 좌표 리스트로 바꿔서 반환한다.
    """
    clearance = cell_size * 0.42
    if open_mask is None:
        open_mask = _build_open_mask(width, height, cell_size, visible_polygons, clearance)
    polygon_bboxes = _build_polygon_bboxes(visible_polygons)

    start_cell = _point_to_cell(start_point, width, height, cell_size)
    goal_cell = _point_to_cell(goal_point, width, height, cell_size)

    start_cell = _find_nearest_open_cell(start_cell, open_mask)
    goal_cell = _find_nearest_open_cell(goal_cell, open_mask)

    if start_cell is None or goal_cell is None:
        return None

    if start_cell == goal_cell:
        direct_distance = math.hypot(goal_point[0] - start_point[0], goal_point[1] - start_point[1])
        path_points = [start_point, goal_point]
        turn_count, total_turn_angle = _compute_turn_metrics(path_points)
        min_clearance, avg_clearance = _compute_clearance_metrics(path_points, visible_polygons)
        return PathResult(
            path_points=path_points,
            distance=direct_distance,
            visited_nodes=1,
            cell_size=cell_size,
            direct_distance=direct_distance,
            detour_ratio=1.0,
            turn_count=turn_count,
            total_turn_angle=total_turn_angle,
            min_clearance=min_clearance,
            avg_clearance=avg_clearance,
        )

    queue: List[Tuple[float, GridCell]] = [(0.0, start_cell)]
    distances: Dict[GridCell, float] = {start_cell: 0.0}
    previous: Dict[GridCell, GridCell] = {}
    visited_nodes = 0

    neighbor_steps = [
        (1, 0, cell_size),
        (-1, 0, cell_size),
        (0, 1, cell_size),
        (0, -1, cell_size),
        (1, 1, cell_size * math.sqrt(2.0)),
        (1, -1, cell_size * math.sqrt(2.0)),
        (-1, 1, cell_size * math.sqrt(2.0)),
        (-1, -1, cell_size * math.sqrt(2.0)),
    ]

    rows, cols = open_mask.shape

    while queue:
        current_distance, current = heapq.heappop(queue)

        if current_distance > distances.get(current, float("inf")):
            continue

        visited_nodes += 1

        if progress_callback is not None and visited_nodes % PROGRESS_UPDATE_INTERVAL == 0:
            progress_callback(visited_nodes)

        if current == goal_cell:
            break

        current_col, current_row = current

        for delta_col, delta_row, step_cost in neighbor_steps:
            next_col = current_col + delta_col
            next_row = current_row + delta_row

            if not (0 <= next_col < cols and 0 <= next_row < rows):
                continue
            if not open_mask[next_row, next_col]:
                continue

            # 대각선 이동 시 옆 칸이 막혀 있으면 모서리를 비집고 통과하는 경로처럼 보일 수 있으므로 막는다.
            if delta_col != 0 and delta_row != 0:
                if not open_mask[current_row, next_col] or not open_mask[next_row, current_col]:
                    continue

            current_center = _cell_center(current, width, height, cell_size)
            neighbor_center = _cell_center((next_col, next_row), width, height, cell_size)
            if _segment_hits_any_polygon(current_center, neighbor_center, visible_polygons, polygon_bboxes):
                continue

            neighbor = (next_col, next_row)
            new_distance = current_distance + step_cost

            if new_distance < distances.get(neighbor, float("inf")):
                distances[neighbor] = new_distance
                previous[neighbor] = current
                heapq.heappush(queue, (new_distance, neighbor))

    if goal_cell not in previous and goal_cell != start_cell:
        return None

    path_cells = _reconstruct_path(previous, start_cell, goal_cell)
    if polygon_bboxes is None:
        polygon_bboxes = _build_polygon_bboxes(visible_polygons)

    path_points: List[Point] = []
    start_center = _cell_center(start_cell, width, height, cell_size)
    goal_center = _cell_center(goal_cell, width, height, cell_size)

    if not _segment_hits_any_polygon(start_point, start_center, visible_polygons, polygon_bboxes):
        path_points.append(start_point)
    else:
        path_points.append(start_center)

    for index, cell in enumerate(path_cells):
        center = _cell_center(cell, width, height, cell_size)
        if not path_points or center != path_points[-1]:
            path_points.append(center)

    if not _segment_hits_any_polygon(goal_center, goal_point, visible_polygons, polygon_bboxes):
        if goal_point != path_points[-1]:
            path_points.append(goal_point)

    actual_distance = _compute_polyline_length(path_points)
    direct_distance = math.hypot(goal_point[0] - start_point[0], goal_point[1] - start_point[1])
    detour_ratio = actual_distance / direct_distance if direct_distance > 0 else 1.0
    turn_count, total_turn_angle = _compute_turn_metrics(path_points)
    min_clearance, avg_clearance = _compute_clearance_metrics(path_points, visible_polygons)

    return PathResult(
        path_points=path_points,
        distance=actual_distance,
        visited_nodes=visited_nodes,
        cell_size=cell_size,
        direct_distance=direct_distance,
        detour_ratio=detour_ratio,
        turn_count=turn_count,
        total_turn_angle=total_turn_angle,
        min_clearance=min_clearance,
        avg_clearance=avg_clearance,
    )


def _build_single_point_result(
    start_point: Point,
    goal_point: Point,
    visible_polygons: List[np.ndarray],
    cell_size: float,
) -> PathResult:
    direct_distance = math.hypot(goal_point[0] - start_point[0], goal_point[1] - start_point[1])
    path_points = [start_point, goal_point]
    turn_count, total_turn_angle = _compute_turn_metrics(path_points)
    min_clearance, avg_clearance = _compute_clearance_metrics(path_points, visible_polygons)
    return PathResult(
        path_points=path_points,
        distance=direct_distance,
        visited_nodes=1,
        cell_size=cell_size,
        direct_distance=direct_distance,
        detour_ratio=1.0,
        turn_count=turn_count,
        total_turn_angle=total_turn_angle,
        min_clearance=min_clearance,
        avg_clearance=avg_clearance,
    )


def _cells_to_path_result(
    previous: Dict[GridCell, GridCell],
    start_cell: GridCell,
    goal_cell: GridCell,
    width: float,
    height: float,
    start_point: Point,
    goal_point: Point,
    visible_polygons: List[np.ndarray],
    cell_size: float,
    visited_nodes: int,
    polygon_bboxes: Optional[List[PolygonBBox]] = None,
) -> PathResult:
    path_cells = _reconstruct_path(previous, start_cell, goal_cell)

    path_points: List[Point] = []
    start_center = _cell_center(start_cell, width, height, cell_size)
    goal_center = _cell_center(goal_cell, width, height, cell_size)

    if not _segment_hits_any_polygon(start_point, start_center, visible_polygons, polygon_bboxes):
        path_points.append(start_point)
    else:
        path_points.append(start_center)

    for cell in path_cells:
        center = _cell_center(cell, width, height, cell_size)
        if not path_points or center != path_points[-1]:
            path_points.append(center)

    if not _segment_hits_any_polygon(goal_center, goal_point, visible_polygons, polygon_bboxes):
        if goal_point != path_points[-1]:
            path_points.append(goal_point)

    actual_distance = _compute_polyline_length(path_points)
    direct_distance = math.hypot(goal_point[0] - start_point[0], goal_point[1] - start_point[1])
    detour_ratio = actual_distance / direct_distance if direct_distance > 0 else 1.0
    turn_count, total_turn_angle = _compute_turn_metrics(path_points)
    min_clearance, avg_clearance = _compute_clearance_metrics(path_points, visible_polygons)

    return PathResult(
        path_points=path_points,
        distance=actual_distance,
        visited_nodes=visited_nodes,
        cell_size=cell_size,
        direct_distance=direct_distance,
        detour_ratio=detour_ratio,
        turn_count=turn_count,
        total_turn_angle=total_turn_angle,
        min_clearance=min_clearance,
        avg_clearance=avg_clearance,
    )


def _heuristic_cell_distance(cell: GridCell, goal_cell: GridCell, cell_size: float) -> float:
    delta_col = goal_cell[0] - cell[0]
    delta_row = goal_cell[1] - cell[1]
    return math.hypot(delta_col, delta_row) * cell_size


def _default_neighbor_steps(cell_size: float) -> List[Tuple[int, int, float]]:
    return [
        (1, 0, cell_size),
        (-1, 0, cell_size),
        (0, 1, cell_size),
        (0, -1, cell_size),
        (1, 1, cell_size * math.sqrt(2.0)),
        (1, -1, cell_size * math.sqrt(2.0)),
        (-1, 1, cell_size * math.sqrt(2.0)),
        (-1, -1, cell_size * math.sqrt(2.0)),
    ]


def _can_step_between_cells(
    current: GridCell,
    next_cell: GridCell,
    open_mask: np.ndarray,
    width: float,
    height: float,
    cell_size: float,
    visible_polygons: List[np.ndarray],
    polygon_bboxes: Optional[List[PolygonBBox]] = None,
) -> bool:
    rows, cols = open_mask.shape
    next_col, next_row = next_cell

    if not (0 <= next_col < cols and 0 <= next_row < rows):
        return False
    if not open_mask[next_row, next_col]:
        return False

    current_col, current_row = current
    delta_col = next_col - current_col
    delta_row = next_row - current_row

    if delta_col != 0 and delta_row != 0:
        if not open_mask[current_row, next_col] or not open_mask[next_row, current_col]:
            return False

    if polygon_bboxes is None:
        polygon_bboxes = _build_polygon_bboxes(visible_polygons)
    current_center = _cell_center(current, width, height, cell_size)
    next_center = _cell_center(next_cell, width, height, cell_size)
    return not _segment_hits_any_polygon(current_center, next_center, visible_polygons, polygon_bboxes)


def _nearest_polygon_distance(
    point: Point,
    visible_polygons: List[np.ndarray],
) -> float:
    if not visible_polygons:
        return float("inf")

    return min(point_to_polygon_distance(point, polygon) for polygon in visible_polygons)


def _safety_weight_for_cell(
    cell: GridCell,
    width: float,
    height: float,
    cell_size: float,
    visible_polygons: List[np.ndarray],
    safety_radius: float,
    penalty_strength: float,
    distance_cache: Dict[GridCell, float],
) -> float:
    if safety_radius <= 0.0:
        return 1.0

    if cell not in distance_cache:
        center = _cell_center(cell, width, height, cell_size)
        distance_cache[cell] = _nearest_polygon_distance(center, visible_polygons)

    ice_distance = distance_cache[cell]
    danger_distance = safety_radius * 0.5
    influence_distance = safety_radius * 3.0

    if ice_distance >= influence_distance:
        return 1.0

    risk_ratio = (influence_distance - ice_distance) / influence_distance
    safety_weight = 1.0 + (risk_ratio ** 2) * penalty_strength

    if ice_distance < danger_distance:
        safety_weight += penalty_strength * 2.0

    return safety_weight


def find_safety_weighted_dijkstra_path(
    width: float,
    height: float,
    start_point: Point,
    goal_point: Point,
    visible_polygons: List[np.ndarray],
    cell_size: float = 0.5,
    progress_callback: Optional[Callable[[int], None]] = None,
    open_mask: Optional[np.ndarray] = None,
    safety_radius: float = 1.0,
    penalty_strength: float = 8.0,
) -> Optional[PathResult]:
    """
    안전가중 다익스트라 알고리즘으로 경로를 계산한다.

    기본 다익스트라는 격자 간 이동 거리만 비용으로 사용하지만,
    이 함수는 해빙에 가까운 칸일수록 이동 비용을 크게 만들어
    해빙과 여유 거리를 두는 경로를 우선 선택하도록 한다.
    """
    clearance = cell_size * 0.42
    if open_mask is None:
        open_mask = _build_open_mask(width, height, cell_size, visible_polygons, clearance)
    polygon_bboxes = _build_polygon_bboxes(visible_polygons)

    start_cell = _find_nearest_open_cell(_point_to_cell(start_point, width, height, cell_size), open_mask)
    goal_cell = _find_nearest_open_cell(_point_to_cell(goal_point, width, height, cell_size), open_mask)

    if start_cell is None or goal_cell is None:
        return None
    if start_cell == goal_cell:
        return _build_single_point_result(start_point, goal_point, visible_polygons, cell_size)

    queue: List[Tuple[float, GridCell]] = [(0.0, start_cell)]
    distances: Dict[GridCell, float] = {start_cell: 0.0}
    previous: Dict[GridCell, GridCell] = {}
    visited_nodes = 0
    neighbor_steps = _default_neighbor_steps(cell_size)
    distance_cache: Dict[GridCell, float] = {}

    while queue:
        current_distance, current = heapq.heappop(queue)

        if current_distance > distances.get(current, float("inf")):
            continue

        visited_nodes += 1

        if progress_callback is not None and visited_nodes % PROGRESS_UPDATE_INTERVAL == 0:
            progress_callback(visited_nodes)

        if current == goal_cell:
            break

        current_col, current_row = current
        for delta_col, delta_row, step_cost in neighbor_steps:
            neighbor = (current_col + delta_col, current_row + delta_row)
            if not _can_step_between_cells(
                current, neighbor, open_mask, width, height, cell_size, visible_polygons, polygon_bboxes
            ):
                continue

            safety_weight = _safety_weight_for_cell(
                neighbor,
                width,
                height,
                cell_size,
                visible_polygons,
                safety_radius,
                penalty_strength,
                distance_cache,
            )
            new_distance = current_distance + step_cost * safety_weight

            if new_distance < distances.get(neighbor, float("inf")):
                distances[neighbor] = new_distance
                previous[neighbor] = current
                heapq.heappush(queue, (new_distance, neighbor))

    if goal_cell not in previous:
        return None

    return _cells_to_path_result(
        previous,
        start_cell,
        goal_cell,
        width,
        height,
        start_point,
        goal_point,
        visible_polygons,
        cell_size,
        visited_nodes,
    )


def _has_line_of_sight_between_cells(
    start_cell: GridCell,
    end_cell: GridCell,
    open_mask: np.ndarray,
    width: float,
    height: float,
    cell_size: float,
    visible_polygons: List[np.ndarray],
    polygon_bboxes: Optional[List[PolygonBBox]] = None,
) -> bool:
    if polygon_bboxes is None:
        polygon_bboxes = _build_polygon_bboxes(visible_polygons)
    start = _cell_center(start_cell, width, height, cell_size)
    end = _cell_center(end_cell, width, height, cell_size)

    if _segment_hits_any_polygon(start, end, visible_polygons, polygon_bboxes):
        return False

    distance = math.hypot(end[0] - start[0], end[1] - start[1])
    sample_count = max(1, int(math.ceil(distance / max(cell_size * 0.5, 1e-9))))

    for index in range(1, sample_count):
        ratio = index / sample_count
        sample = (
            start[0] + (end[0] - start[0]) * ratio,
            start[1] + (end[1] - start[1]) * ratio,
        )
        sample_cell = _point_to_cell(sample, width, height, cell_size)
        if not open_mask[sample_cell[1], sample_cell[0]]:
            return False

    return True


def find_astar_path(
    width: float,
    height: float,
    start_point: Point,
    goal_point: Point,
    visible_polygons: List[np.ndarray],
    cell_size: float = 0.5,
    progress_callback: Optional[Callable[[int], None]] = None,
    open_mask: Optional[np.ndarray] = None,
) -> Optional[PathResult]:
    clearance = cell_size * 0.42
    if open_mask is None:
        open_mask = _build_open_mask(width, height, cell_size, visible_polygons, clearance)
    polygon_bboxes = _build_polygon_bboxes(visible_polygons)

    start_cell = _find_nearest_open_cell(_point_to_cell(start_point, width, height, cell_size), open_mask)
    goal_cell = _find_nearest_open_cell(_point_to_cell(goal_point, width, height, cell_size), open_mask)

    if start_cell is None or goal_cell is None:
        return None
    if start_cell == goal_cell:
        return _build_single_point_result(start_point, goal_point, visible_polygons, cell_size)

    queue: List[Tuple[float, float, GridCell]] = [(0.0, 0.0, start_cell)]
    distances: Dict[GridCell, float] = {start_cell: 0.0}
    previous: Dict[GridCell, GridCell] = {}
    visited_nodes = 0
    neighbor_steps = _default_neighbor_steps(cell_size)

    while queue:
        _, current_distance, current = heapq.heappop(queue)

        if current_distance > distances.get(current, float("inf")):
            continue

        visited_nodes += 1

        if progress_callback is not None and visited_nodes % PROGRESS_UPDATE_INTERVAL == 0:
            progress_callback(visited_nodes)

        if current == goal_cell:
            break

        current_col, current_row = current
        for delta_col, delta_row, step_cost in neighbor_steps:
            neighbor = (current_col + delta_col, current_row + delta_row)
            if not _can_step_between_cells(
                current, neighbor, open_mask, width, height, cell_size, visible_polygons, polygon_bboxes
            ):
                continue

            new_distance = current_distance + step_cost
            if new_distance < distances.get(neighbor, float("inf")):
                distances[neighbor] = new_distance
                previous[neighbor] = current
                priority = new_distance + _heuristic_cell_distance(neighbor, goal_cell, cell_size)
                heapq.heappush(queue, (priority, new_distance, neighbor))

    if goal_cell not in previous:
        return None

    return _cells_to_path_result(
        previous,
        start_cell,
        goal_cell,
        width,
        height,
        start_point,
        goal_point,
        visible_polygons,
        cell_size,
        visited_nodes,
    )


def find_theta_star_path(
    width: float,
    height: float,
    start_point: Point,
    goal_point: Point,
    visible_polygons: List[np.ndarray],
    cell_size: float = 0.5,
    progress_callback: Optional[Callable[[int], None]] = None,
    open_mask: Optional[np.ndarray] = None,
) -> Optional[PathResult]:
    clearance = cell_size * 0.42
    if open_mask is None:
        open_mask = _build_open_mask(width, height, cell_size, visible_polygons, clearance)
    polygon_bboxes = _build_polygon_bboxes(visible_polygons)

    start_cell = _find_nearest_open_cell(_point_to_cell(start_point, width, height, cell_size), open_mask)
    goal_cell = _find_nearest_open_cell(_point_to_cell(goal_point, width, height, cell_size), open_mask)

    if start_cell is None or goal_cell is None:
        return None
    if start_cell == goal_cell:
        return _build_single_point_result(start_point, goal_point, visible_polygons, cell_size)

    queue: List[Tuple[float, float, GridCell]] = [(0.0, 0.0, start_cell)]
    distances: Dict[GridCell, float] = {start_cell: 0.0}
    parent: Dict[GridCell, GridCell] = {start_cell: start_cell}
    visited_nodes = 0
    neighbor_steps = _default_neighbor_steps(cell_size)

    while queue:
        _, current_distance, current = heapq.heappop(queue)

        if current_distance > distances.get(current, float("inf")):
            continue

        visited_nodes += 1

        if progress_callback is not None and visited_nodes % PROGRESS_UPDATE_INTERVAL == 0:
            progress_callback(visited_nodes)

        if current == goal_cell:
            break

        current_col, current_row = current
        current_parent = parent[current]

        for delta_col, delta_row, step_cost in neighbor_steps:
            neighbor = (current_col + delta_col, current_row + delta_row)
            if not _can_step_between_cells(
                current, neighbor, open_mask, width, height, cell_size, visible_polygons, polygon_bboxes
            ):
                continue

            if _has_line_of_sight_between_cells(
                current_parent, neighbor, open_mask, width, height, cell_size, visible_polygons, polygon_bboxes
            ):
                base_cell = current_parent
                base_center = _cell_center(base_cell, width, height, cell_size)
                neighbor_center = _cell_center(neighbor, width, height, cell_size)
                new_distance = distances[base_cell] + math.hypot(
                    neighbor_center[0] - base_center[0],
                    neighbor_center[1] - base_center[1],
                )
            else:
                base_cell = current
                new_distance = current_distance + step_cost

            if new_distance < distances.get(neighbor, float("inf")):
                distances[neighbor] = new_distance
                parent[neighbor] = base_cell
                priority = new_distance + _heuristic_cell_distance(neighbor, goal_cell, cell_size)
                heapq.heappush(queue, (priority, new_distance, neighbor))

    if goal_cell not in parent:
        return None

    return _cells_to_path_result(
        parent,
        start_cell,
        goal_cell,
        width,
        height,
        start_point,
        goal_point,
        visible_polygons,
        cell_size,
        visited_nodes,
    )

