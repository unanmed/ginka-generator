import numpy as np

def blend_alpha(bg, fg, alpha):
    """ 使用 alpha 通道混合前景图块和背景图 """
    for c in range(3):  # 只混合 RGB 三个通道
        bg[:, :, c] = (1 - alpha) * bg[:, :, c] + alpha * fg[:, :, c]
    return bg

def matrix_to_image_cv(map_matrix, tile_set, tile_size=32):
    """
    使用OpenCV加速的版本（适合大尺寸地图）
    :param map_matrix: [H, W] 的numpy数组
    :param tile_set: 字典 {tile_id: cv2图像（BGR格式）}
    :param tile_size: 图块边长（像素）
    """
    H, W = map_matrix.shape  # 获取地图尺寸
    canvas = np.zeros((H * tile_size, W * tile_size, 3), dtype=np.uint8)  # 画布（黑色背景）
    
    # 遍历地图矩阵
    for row in range(H):
        for col in range(W):
            tile_index = str(map_matrix[row, col])  # 获取当前坐标的图块类型
            x, y = col * tile_size, row * tile_size  # 计算像素位置

            # 先绘制地面（0）
            if '0' in tile_set:
                canvas[y:y+tile_size, x:x+tile_size] = tile_set['0'][:, :, :3]  # 仅填充 RGB

            if tile_index == '30':
                if row == 0:
                    tile_index = '30_1'
                elif row == W - 1:
                    tile_index = '30_3'
                elif col == 0:
                    tile_index = '30_2'
                elif col == H - 1:
                    tile_index = '30_4'

            # 叠加其他透明图块
            if tile_index in tile_set and tile_index != 0:
                tile_rgba = tile_set[tile_index]
                tile_rgb = tile_rgba[:, :, :3]  # 提取 RGB
                alpha = tile_rgba[:, :, 3] / 255.0  # 归一化 alpha

                # 混合当前图块到背景
                canvas[y:y+tile_size, x:x+tile_size] = blend_alpha(
                    canvas[y:y+tile_size, x:x+tile_size], tile_rgb, alpha
                )

    return canvas