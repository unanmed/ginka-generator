from typing import List, Dict
import math
import numpy as np

# 视觉相似度，由 ChatGPT-4o 从 ts 转译而来

class VisualSimilarityConfig:
    def __init__(self):
        self.type_weights: Dict[int, float] = {
            0: 0.2, 1: 0.3, 2: 0.6, 3: 0.7, 4: 0.7, 5: 0.5,
            6: 0.4, 7: 0.5, 8: 0.6, 9: 0.6, 10: 0.4, 11: 0.4, 12: 0.7
        }
        self.enable_visual_focus: bool = True
        self.enable_density_awareness: bool = True

def generate_focus_weights(rows: int, cols: int) -> List[List[float]]:
    weights = []
    center_x = cols / 2
    center_y = rows / 2
    for i in range(rows):
        row_weights = []
        for j in range(cols):
            dx = (j - center_x) / cols
            dy = (i - center_y) / rows
            distance = math.sqrt(dx ** 2 + dy ** 2)
            gaussian = math.exp(-(distance ** 2) / (2 * 0.3 ** 2))
            row_weights.append(1.0 + 0.6 * gaussian)
        weights.append(row_weights)
    return weights

def calculate_density_impact(map1: List[List[int]], map2: List[List[int]], type_weights: Dict[int, float]) -> List[List[float]]:
    rows, cols = len(map1), len(map1[0])
    density_map = [[0.0 for _ in range(cols)] for _ in range(rows)]
    window_size = 3
    half_window = window_size // 2

    for i in range(rows):
        for j in range(cols):
            density = 0
            for di in range(-half_window, half_window + 1):
                for dj in range(-half_window, half_window + 1):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        weight1 = type_weights.get(map1[ni][nj], 0.5)
                        weight2 = type_weights.get(map2[ni][nj], 0.5)
                        density += (weight1 + weight2) / 2
            density_map[i][j] = 1.0 + 0.4 * (density / (window_size ** 2))
    return density_map

def calculate_visual_similarity(map1: List[List[int]], map2: List[List[int]], config: VisualSimilarityConfig = None) -> float:
    if config is None:
        config = VisualSimilarityConfig()

    if len(map1) != len(map2) or len(map1[0]) != len(map2[0]):
        return 0.0

    rows, cols = len(map1), len(map1[0])
    total_score = 0.0
    max_possible_score = 0.0

    focus_weights = generate_focus_weights(rows, cols) if config.enable_visual_focus else [[1.0 for _ in range(cols)] for _ in range(rows)]
    density_map = calculate_density_impact(map1, map2, config.type_weights) if config.enable_density_awareness else [[1.0 for _ in range(cols)] for _ in range(rows)]

    for i in range(rows):
        for j in range(cols):
            type1 = map1[i][j]
            type2 = map2[i][j]
            base_weight = max(config.type_weights.get(type1, 0.5), config.type_weights.get(type2, 0.5))
            spatial_weight = focus_weights[i][j] * density_map[i][j]
            type_score = 1.0 if type1 == type2 else 0.0

            total_score += type_score * base_weight * spatial_weight
            max_possible_score += base_weight * spatial_weight

    return total_score / max_possible_score if max_possible_score > 0 else 0.0
