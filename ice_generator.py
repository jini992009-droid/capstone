from dataclasses import dataclass
import math
import random
from typing import List

import numpy as np

from geometry_utils import (
    Point,
    clip_polygon_to_rect,
    generate_convex_polygon,
    polygon_area,
    polygons_overlap,
)


@dataclass
class IceChunk:
    """
    얼음 조각 하나를 저장하는 자료형이다.

    문법 메모:
    - `@dataclass`
      여러 값을 묶는 "데이터 전용 클래스"를 간단하게 만들게 도와준다.
      `__init__` 같은 코드를 자동으로 만들어 준다고 생각하면 된다.

    필드 설명:
    - points: 실제 얼음 다각형의 꼭짓점들
    - center: 배치에 사용한 원형 후보의 중심
    - radius: 배치에 사용한 원형 후보의 반지름
    """
    points: np.ndarray
    center: Point
    radius: float


def circles_overlap(
    center_a: Point,
    radius_a: float,
    center_b: Point,
    radius_b: float,
    gap: float,
) -> bool:
    """
    두 원이 서로 겹치는지 검사한다.

    왜 원으로 검사하는가:
    - 다각형끼리 직접 비교하는 것보다 빠르고 설명도 쉽다.
    - 먼저 원 기준으로 걸러낸 뒤, 통과한 후보만 다각형으로 만든다.
    """
    dx = center_a[0] - center_b[0]
    dy = center_a[1] - center_b[1]
    return math.hypot(dx, dy) < (radius_a + radius_b + gap)


def create_corner_ice_chunk(
    width: float,
    height: float,
    min_vertices: int,
    max_vertices: int,
    edge_margin: float,
    min_gap: float,
) -> IceChunk:
    """
    해역 바깥 코너에 있는 초대형 해빙 1개를 만든다.

    핵심 아이디어:
    - 큰 해빙의 중심을 해역 바깥 코너 쪽에 둔다.
    - 따라서 실제로는 매우 큰 해빙이지만, 화면에서는 해역 안으로 들어온 일부만 보인다.
    - 네가 그린 예시처럼 "테두리에 붙은 커다란 한 덩어리"로 보이게 만드는 방식이다.
    """
    corner = random.choice(("top_left", "top_right", "bottom_left", "bottom_right"))

    # 화면 안에 보이는 큰 해빙이 해역의 약 1/3 정도를 차지하도록 목표 면적을 더 크게 잡는다.
    target_area_min = width * height * 0.31
    target_area_max = width * height * 0.37
    edge_eps = 1e-6

    for _ in range(400):
        # 실제 해빙은 해역보다 훨씬 큰 덩어리라고 가정한다.
        diameter = random.uniform(min(width, height) * 1.35, min(width, height) * 1.75)
        radius = diameter / 2.0

        # 중심을 해역 밖에 두되, 이전보다 조금 덜 바깥에 둬서 화면 안에 들어오는 비율을 키운다.
        outside_ratio_x = random.uniform(0.06, 0.16)
        outside_ratio_y = random.uniform(0.06, 0.16)
        outside_x = width * outside_ratio_x
        outside_y = height * outside_ratio_y

        if corner == "top_left":
            center = (-outside_x, height + outside_y)
        elif corner == "top_right":
            center = (width + outside_x, height + outside_y)
        elif corner == "bottom_left":
            center = (-outside_x, -outside_y)
        else:
            center = (width + outside_x, -outside_y)

        polygon = generate_convex_polygon(
            diameter=diameter,
            origin=center,
            min_vertices=max(min_vertices + 6, 16),
            max_vertices=max(max_vertices + 10, 26),
        )

        visible_polygon = clip_polygon_to_rect(polygon, 0.0, 0.0, width, height)
        if len(visible_polygon) < 3:
            continue

        visible_area = polygon_area(visible_polygon)
        if visible_area < target_area_min or visible_area > target_area_max:
            continue

        visible_width = float(np.max(visible_polygon[:, 0]) - np.min(visible_polygon[:, 0]))
        visible_height = float(np.max(visible_polygon[:, 1]) - np.min(visible_polygon[:, 1]))

        if corner == "top_left":
            touches_corner_edges = (
                np.min(visible_polygon[:, 0]) <= edge_eps
                and np.max(visible_polygon[:, 1]) >= height - edge_eps
            )
        elif corner == "top_right":
            touches_corner_edges = (
                np.max(visible_polygon[:, 0]) >= width - edge_eps
                and np.max(visible_polygon[:, 1]) >= height - edge_eps
            )
        elif corner == "bottom_left":
            touches_corner_edges = (
                np.min(visible_polygon[:, 0]) <= edge_eps
                and np.min(visible_polygon[:, 1]) <= edge_eps
            )
        else:
            touches_corner_edges = (
                np.max(visible_polygon[:, 0]) >= width - edge_eps
                and np.min(visible_polygon[:, 1]) <= edge_eps
            )

        # 너무 얇은 삼각형 조각처럼 보이지 않도록, 화면 안에서도 충분한 폭과 높이를 갖게 한다.
        has_large_presence = (
            visible_width >= width * 0.42
            and visible_height >= height * 0.42
        )

        if touches_corner_edges and has_large_presence:
            return IceChunk(points=polygon, center=center, radius=radius)

    # 반복 시도 후에도 조건에 안 맞으면 기본 배치를 사용한다.
    fallback_diameter = min(width, height) * 1.6
    fallback_radius = fallback_diameter / 2.0
    fallback_offset_x = width * 0.1
    fallback_offset_y = height * 0.1

    if corner == "top_left":
        fallback_center = (-fallback_offset_x, height + fallback_offset_y)
    elif corner == "top_right":
        fallback_center = (width + fallback_offset_x, height + fallback_offset_y)
    elif corner == "bottom_left":
        fallback_center = (-fallback_offset_x, -fallback_offset_y)
    else:
        fallback_center = (width + fallback_offset_x, -fallback_offset_y)

    fallback_polygon = generate_convex_polygon(
        diameter=fallback_diameter,
        origin=fallback_center,
        min_vertices=max(min_vertices + 6, 16),
        max_vertices=max(max_vertices + 10, 26),
    )

    return IceChunk(points=fallback_polygon, center=fallback_center, radius=fallback_radius)


def generate_ice_chunks(
    width: float,
    height: float,
    count: int,
    min_diameter: float,
    max_diameter: float,
    min_vertices: int,
    max_vertices: int,
    edge_margin: float,
    min_gap: float,
    max_attempts: int,
) -> List[IceChunk]:
    """
    서로 겹치지 않는 얼음 다각형 여러 개를 생성한다.

    문법 메모:
    - `List[IceChunk]`
      `IceChunk` 객체가 여러 개 들어 있는 리스트라는 뜻이다.

    전체 순서:
    1. 후보 원의 크기를 고른다.
    2. 후보 중심 위치를 고른다.
    3. 기존 얼음과 겹치면 버린다.
    4. 통과하면 다각형을 만든다.
    5. 바다 경계를 넘으면 버린다.
    6. 통과하면 저장한다.
    """
    chunks: List[IceChunk] = []

    # 큰 해빙 1개를 먼저 배치해서 전체 분포가 더 실제 해빙처럼 보이게 만든다.
    # count가 1 이상이면 전체 개수 안에서 큰 해빙 1개를 포함한다.
    if count > 0:
        large_chunk = create_corner_ice_chunk(
            width=width,
            height=height,
            min_vertices=min_vertices,
            max_vertices=max_vertices,
            edge_margin=edge_margin,
            min_gap=min_gap,
        )
        chunks.append(large_chunk)

    remaining_count = max(0, count - len(chunks))

    for _ in range(remaining_count):
        placed = False

        for _ in range(max_attempts):
            diameter = random.uniform(min_diameter, max_diameter)
            radius = diameter / 2.0

            center_x = random.uniform(edge_margin + radius, width - edge_margin - radius)
            center_y = random.uniform(edge_margin + radius, height - edge_margin - radius)
            center = (center_x, center_y)

            overlaps_existing = any(
                circles_overlap(center, radius, chunk.center, chunk.radius, min_gap)
                for chunk in chunks
            )
            if overlaps_existing:
                continue

            polygon = generate_convex_polygon(
                diameter=diameter,
                origin=center,
                min_vertices=min_vertices,
                max_vertices=max_vertices,
            )

            # 생성된 다각형이 바다 경계 안에 완전히 들어가는지 확인
            min_x = float(np.min(polygon[:, 0]))
            max_x = float(np.max(polygon[:, 0]))
            min_y = float(np.min(polygon[:, 1]))
            max_y = float(np.max(polygon[:, 1]))
            if (
                min_x < edge_margin
                or max_x > width - edge_margin
                or min_y < edge_margin
                or max_y > height - edge_margin
            ):
                continue

            candidate_visible = clip_polygon_to_rect(polygon, 0.0, 0.0, width, height)
            if len(candidate_visible) < 3:
                continue

            overlaps_polygon = False
            for chunk in chunks:
                existing_visible = clip_polygon_to_rect(chunk.points, 0.0, 0.0, width, height)
                if polygons_overlap(candidate_visible, existing_visible):
                    overlaps_polygon = True
                    break

            if overlaps_polygon:
                continue

            chunks.append(IceChunk(points=polygon, center=center, radius=radius))
            placed = True
            break

        # 너무 빽빽하면 요청 개수를 다 채우지 못할 수 있다.
        if not placed:
            break

    return chunks
