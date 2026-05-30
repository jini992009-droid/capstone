import math
import random
from typing import List, Tuple

import numpy as np


# `Tuple[float, float]`는 "실수 2개로 이루어진 튜플"이라는 뜻이다.
# 여기서는 (x, y) 좌표 한 점을 뜻하도록 `Point`라는 이름을 붙여서 쓴다.
Point = Tuple[float, float]


def polygon_area(vertices: np.ndarray) -> float:
    """
    다각형의 면적을 계산한다.

    문법 메모:
    - `vertices: np.ndarray`
      `vertices` 매개변수는 numpy 배열이라는 뜻이다.
    - `-> float`
      이 함수의 반환값이 실수(float)라는 뜻이다.

    계산 방식:
    - 신발끈 공식(shoelace formula)을 사용한다.
    """
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def polygon_centroid(vertices: np.ndarray) -> np.ndarray:
    """
    다각형의 중심점(centroid)을 계산한다.

    왜 필요한가:
    - 생성한 다각형을 원하는 위치로 옮기려면,
      먼저 그 다각형의 중심이 어디인지 알아야 한다.
    """
    x = vertices[:, 0]
    y = vertices[:, 1]
    area = polygon_area(vertices)

    # 면적이 0이면 비정상적인 도형이므로 평균 좌표를 대신 사용한다.
    if area == 0:
        return np.mean(vertices, axis=0)

    cross = x * np.roll(y, 1) - np.roll(x, 1) * y
    centroid_x = np.dot(x + np.roll(x, 1), cross) / (6 * area)
    centroid_y = np.dot(y + np.roll(y, 1), cross) / (6 * area)
    return np.abs(np.array([centroid_x, centroid_y]))


def generate_convex_polygon(
    diameter: float,
    origin: Point,
    min_vertices: int,
    max_vertices: int,
) -> np.ndarray:
    """
    하나의 불규칙한 볼록 다각형을 생성한다.

    문법 메모:
    - `origin: Point`
      `origin`은 위에서 만든 Point 타입, 즉 `(x, y)` 좌표다.
    - `List[float]`
      실수 여러 개가 들어 있는 리스트라는 뜻이다.

    전체 흐름:
    1. 직경 범위 안에서 x, y 난수를 만든다.
    2. 각각 정렬한다.
    3. 가장자리 벡터를 만든다.
    4. 벡터를 각도순으로 정렬한다.
    5. 벡터를 이어서 볼록 다각형을 만든다.
    6. 마지막에 다각형 중심이 origin에 오도록 이동시킨다.
    """
    vertex_count = random.randint(min_vertices, max_vertices)

    x_values = sorted(random.uniform(0, diameter) for _ in range(vertex_count))
    y_values = sorted(random.uniform(0, diameter) for _ in range(vertex_count))

    x_min = x_values[0]
    x_max = x_values[-1]
    y_min = y_values[0]
    y_max = y_values[-1]

    # x 방향 벡터 만들기
    last_top = x_min
    last_bottom = x_min
    x_vectors: List[float] = []
    for value in x_values[1:-1]:
        if random.getrandbits(1):
            x_vectors.append(value - last_top)
            last_top = value
        else:
            x_vectors.append(last_bottom - value)
            last_bottom = value
    x_vectors.append(x_max - last_top)
    x_vectors.append(last_bottom - x_max)

    # y 방향 벡터 만들기
    last_left = y_min
    last_right = y_min
    y_vectors: List[float] = []
    for value in y_values[1:-1]:
        if random.getrandbits(1):
            y_vectors.append(value - last_left)
            last_left = value
        else:
            y_vectors.append(last_right - value)
            last_right = value
    y_vectors.append(y_max - last_left)
    y_vectors.append(last_right - y_max)

    # 너무 규칙적인 모양이 되지 않도록 y 벡터는 한 번 섞는다.
    random.shuffle(y_vectors)

    # `lambda pair: ...`는 "정렬 기준을 만드는 짧은 함수"라고 이해하면 된다.
    vector_pairs = sorted(
        zip(x_vectors, y_vectors),
        key=lambda pair: math.atan2(pair[1], pair[0]),
    )

    # `List[Point]`는 "(x, y) 점이 여러 개 들어 있는 리스트"라는 뜻이다.
    points: List[Point] = []
    x_pos = 0.0
    y_pos = 0.0
    min_polygon_x = 0.0
    min_polygon_y = 0.0

    for x_vec, y_vec in vector_pairs:
        points.append((x_pos, y_pos))
        x_pos += x_vec
        y_pos += y_vec
        min_polygon_x = min(min_polygon_x, x_pos)
        min_polygon_y = min(min_polygon_y, y_pos)

    shifted = np.asarray(points, dtype=float)

    # 다각형을 양수 좌표 쪽으로 이동
    shifted[:, 0] += x_min - min_polygon_x
    shifted[:, 1] += y_min - min_polygon_y

    # 중심이 origin 위치로 오도록 이동
    center = polygon_centroid(shifted)
    shifted -= center - np.asarray(origin, dtype=float)

    return shifted


def clip_polygon_to_rect(
    vertices: np.ndarray,
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
) -> np.ndarray:
    """
    다각형을 사각형 경계 안으로 잘라낸다.

    왜 필요한가:
    - 아주 큰 해빙은 실제로는 화면 바깥까지 이어져 있어도 된다.
    - 하지만 화면에는 해역 내부에 들어온 부분만 보여야 하므로,
      다각형을 사각형(해역) 기준으로 잘라낸 결과가 필요하다.
    """
    points = [tuple(point) for point in vertices]

    def clip_against_edge(points_list, inside_fn, intersect_fn):
        if not points_list:
            return []

        output = []
        previous = points_list[-1]
        previous_inside = inside_fn(previous)

        for current in points_list:
            current_inside = inside_fn(current)

            if current_inside:
                if not previous_inside:
                    output.append(intersect_fn(previous, current))
                output.append(current)
            elif previous_inside:
                output.append(intersect_fn(previous, current))

            previous = current
            previous_inside = current_inside

        return output

    def intersect_vertical(a, b, x_value):
        dx = b[0] - a[0]
        if dx == 0:
            return (x_value, a[1])
        t = (x_value - a[0]) / dx
        return (x_value, a[1] + t * (b[1] - a[1]))

    def intersect_horizontal(a, b, y_value):
        dy = b[1] - a[1]
        if dy == 0:
            return (a[0], y_value)
        t = (y_value - a[1]) / dy
        return (a[0] + t * (b[0] - a[0]), y_value)

    points = clip_against_edge(
        points,
        inside_fn=lambda p: p[0] >= min_x,
        intersect_fn=lambda a, b: intersect_vertical(a, b, min_x),
    )
    points = clip_against_edge(
        points,
        inside_fn=lambda p: p[0] <= max_x,
        intersect_fn=lambda a, b: intersect_vertical(a, b, max_x),
    )
    points = clip_against_edge(
        points,
        inside_fn=lambda p: p[1] >= min_y,
        intersect_fn=lambda a, b: intersect_horizontal(a, b, min_y),
    )
    points = clip_against_edge(
        points,
        inside_fn=lambda p: p[1] <= max_y,
        intersect_fn=lambda a, b: intersect_horizontal(a, b, max_y),
    )

    if len(points) < 3:
        return np.empty((0, 2), dtype=float)

    return np.asarray(points, dtype=float)


def polygon_bbox(vertices: np.ndarray) -> Tuple[float, float, float, float]:
    """
    다각형의 바운딩 박스(min_x, min_y, max_x, max_y)를 구한다.

    바운딩 박스는 자세한 충돌 검사 전에 빠르게 겹침 가능성을 걸러낼 때 유용하다.
    """
    return (
        float(np.min(vertices[:, 0])),
        float(np.min(vertices[:, 1])),
        float(np.max(vertices[:, 0])),
        float(np.max(vertices[:, 1])),
    )


def bboxes_overlap(
    bbox_a: Tuple[float, float, float, float],
    bbox_b: Tuple[float, float, float, float],
) -> bool:
    """
    두 바운딩 박스가 겹치는지 검사한다.
    """
    min_x_a, min_y_a, max_x_a, max_y_a = bbox_a
    min_x_b, min_y_b, max_x_b, max_y_b = bbox_b

    return not (
        max_x_a < min_x_b
        or max_x_b < min_x_a
        or max_y_a < min_y_b
        or max_y_b < min_y_a
    )


def _orientation(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: Point, b: Point, c: Point) -> bool:
    return (
        min(a[0], c[0]) <= b[0] <= max(a[0], c[0])
        and min(a[1], c[1]) <= b[1] <= max(a[1], c[1])
    )


def _segments_intersect(a1: Point, a2: Point, b1: Point, b2: Point) -> bool:
    eps = 1e-9
    o1 = _orientation(a1, a2, b1)
    o2 = _orientation(a1, a2, b2)
    o3 = _orientation(b1, b2, a1)
    o4 = _orientation(b1, b2, a2)

    if ((o1 > eps and o2 < -eps) or (o1 < -eps and o2 > eps)) and (
        (o3 > eps and o4 < -eps) or (o3 < -eps and o4 > eps)
    ):
        return True

    if abs(o1) <= eps and _on_segment(a1, b1, a2):
        return True
    if abs(o2) <= eps and _on_segment(a1, b2, a2):
        return True
    if abs(o3) <= eps and _on_segment(b1, a1, b2):
        return True
    if abs(o4) <= eps and _on_segment(b1, a2, b2):
        return True

    return False


def point_in_polygon(point: Point, polygon: np.ndarray) -> bool:
    """
    점 하나가 다각형 내부에 있는지 검사한다.
    """
    x, y = point
    inside = False
    count = len(polygon)

    for i in range(count):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % count]
        crosses = (y1 > y) != (y2 > y)
        if crosses:
            x_at_y = (x2 - x1) * (y - y1) / ((y2 - y1) + 1e-12) + x1
            if x < x_at_y:
                inside = not inside

    return inside


def polygons_overlap(poly_a: np.ndarray, poly_b: np.ndarray) -> bool:
    """
    두 다각형이 서로 겹치는지 검사한다.

    검사 순서:
    1. 바운딩 박스로 빠르게 1차 필터링
    2. 선분 교차 검사
    3. 한 다각형이 다른 다각형 내부에 포함되는지 검사
    """
    if len(poly_a) < 3 or len(poly_b) < 3:
        return False

    if not bboxes_overlap(polygon_bbox(poly_a), polygon_bbox(poly_b)):
        return False

    for i in range(len(poly_a)):
        a1 = tuple(poly_a[i])
        a2 = tuple(poly_a[(i + 1) % len(poly_a)])
        for j in range(len(poly_b)):
            b1 = tuple(poly_b[j])
            b2 = tuple(poly_b[(j + 1) % len(poly_b)])
            if _segments_intersect(a1, a2, b1, b2):
                return True

    if point_in_polygon(tuple(poly_a[0]), poly_b):
        return True
    if point_in_polygon(tuple(poly_b[0]), poly_a):
        return True

    return False


def segment_intersects_polygon(start: Point, end: Point, polygon: np.ndarray) -> bool:
    """
    선분 하나가 다각형과 겹치거나 통과하는지 검사한다.

    왜 필요한가:
    - 격자 중심점끼리 이동하는 경로라도,
      선분 자체가 얼음 다각형을 가로지르면 실제로는 통과한 것으로 보일 수 있다.
    - 그래서 경로 선분이 다각형 경계와 교차하는지 추가 검사한다.
    """
    if len(polygon) < 3:
        return False

    segment_bbox = (
        min(start[0], end[0]),
        min(start[1], end[1]),
        max(start[0], end[0]),
        max(start[1], end[1]),
    )
    if not bboxes_overlap(segment_bbox, polygon_bbox(polygon)):
        return False

    if point_in_polygon(start, polygon) or point_in_polygon(end, polygon):
        return True

    for index in range(len(polygon)):
        edge_start = tuple(polygon[index])
        edge_end = tuple(polygon[(index + 1) % len(polygon)])
        if _segments_intersect(start, end, edge_start, edge_end):
            return True

    return False


def point_to_segment_distance(point: Point, start: Point, end: Point) -> float:
    """
    점 하나와 선분 하나 사이의 최단 거리를 구한다.
    """
    px, py = point
    sx, sy = start
    ex, ey = end

    dx = ex - sx
    dy = ey - sy

    if dx == 0 and dy == 0:
        return math.hypot(px - sx, py - sy)

    projection = ((px - sx) * dx + (py - sy) * dy) / (dx * dx + dy * dy)
    projection = max(0.0, min(1.0, projection))

    nearest_x = sx + projection * dx
    nearest_y = sy + projection * dy
    return math.hypot(px - nearest_x, py - nearest_y)


def point_to_polygon_distance(point: Point, polygon: np.ndarray) -> float:
    """
    점 하나와 다각형 사이의 최단 거리를 구한다.

    - 점이 다각형 내부에 있으면 거리는 0이다.
    - 외부에 있으면 각 변까지의 최소 거리를 반환한다.
    """
    if len(polygon) < 3:
        return float("inf")

    if point_in_polygon(point, polygon):
        return 0.0

    min_distance = float("inf")
    for index in range(len(polygon)):
        edge_start = tuple(polygon[index])
        edge_end = tuple(polygon[(index + 1) % len(polygon)])
        distance = point_to_segment_distance(point, edge_start, edge_end)
        min_distance = min(min_distance, distance)

    return min_distance
