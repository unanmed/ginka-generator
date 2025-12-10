import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from .image import matrix_to_image_cv

# -------------------------
# 加载 tile 图块
# -------------------------
def load_tiles(tile_folder):
    tile_dict = {}
    for file in os.listdir(tile_folder):
        name, _ = os.path.splitext(file)
        img = cv2.imread(os.path.join(tile_folder, file), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[WARN] Tile image {file} 读取失败，跳过")
            continue
        tile_dict[name] = img
    print(f"加载了 {len(tile_dict)} 个图块")
    return tile_dict


# -------------------------
# 主处理逻辑
# -------------------------
def convert_dataset_to_images(
    json_path,
    tile_folder,
    output_folder,
    tile_size=32
):
    # 输出路径
    os.makedirs(output_folder, exist_ok=True)

    # 加载 tiles
    tile_dict = load_tiles(tile_folder)

    # 读取 json
    with open(json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    data = dataset["data"]

    for map_id, train_data in tqdm(data.items()):
        map_matrix = np.array(train_data["map"])

        try:
            img = matrix_to_image_cv(map_matrix, tile_dict, tile_size)
        except Exception as e:
            print(f"[ERROR] 地图 {map_id} 转换失败: {e}")
            continue

        out_path = os.path.join(output_folder, f"{map_id.replace('::', '-')}.png")
        cv2.imwrite(out_path, img)
        
    print('地图处理完毕！')


# -------------------------
# 执行
# -------------------------
if __name__ == "__main__":
    convert_dataset_to_images(
        json_path="data/result.json",     # 数据集文件
        tile_folder="tiles",          # 贴图文件夹
        output_folder="map_images",  # 输出文件夹
        tile_size=32                  # tile 尺寸
    )