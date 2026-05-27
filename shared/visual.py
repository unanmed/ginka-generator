import argparse
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
def normalize_filter_reasons(train_data):
    reasons = train_data.get("filterReasons", [])
    if isinstance(reasons, str):
        return [reasons]
    if not isinstance(reasons, list):
        return []
    return [str(reason) for reason in reasons if reason]


def draw_filter_reasons(img, reasons):
    if not reasons:
        return img

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    padding = 8
    line_gap = 6
    text_sizes = [cv2.getTextSize(text, font, font_scale, thickness)[0] for text in reasons]
    line_height = max(height for _, height in text_sizes)
    box_width = min(max(width for width, _ in text_sizes) + padding * 2, img.shape[1])
    box_height = min(
        padding * 2 + len(reasons) * line_height + max(0, len(reasons) - 1) * line_gap,
        img.shape[0]
    )

    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (box_width, box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.72, img, 0.28, 0, img)

    y = padding + line_height
    for text in reasons:
        cv2.putText(
            img,
            text,
            (padding, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )
        y += line_height + line_gap

    return img


def render_dataset_images(json_path, tile_dict, output_folder, tile_size=32):
    if not json_path:
        return
    if not os.path.exists(json_path):
        print(f"[WARN] 数据集 {json_path} 不存在，跳过")
        return

    os.makedirs(output_folder, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    data = dataset.get("data", {})

    for map_id, train_data in tqdm(data.items(), desc=os.path.basename(json_path)):
        map_matrix = np.array(train_data["map"])

        try:
            img = matrix_to_image_cv(map_matrix, tile_dict, tile_size)
        except Exception as e:
            print(f"[ERROR] 地图 {map_id} 转换失败: {e}")
            continue

        reasons = normalize_filter_reasons(train_data)
        if reasons:
            img = draw_filter_reasons(img, reasons)

        out_path = os.path.join(output_folder, f"{map_id.replace('::', '-')}.png")
        cv2.imwrite(out_path, img)

    print(f"{json_path} 地图处理完毕！")


def convert_dataset_to_images(
    json_path,
    tile_folder,
    output_folder,
    tile_size=32,
    filtered_json_path=None
):
    # 加载 tiles
    tile_dict = load_tiles(tile_folder)

    render_dataset_images(json_path, tile_dict, output_folder, tile_size)

    if filtered_json_path:
        filtered_output_folder = os.path.join(output_folder, "filtered")
        render_dataset_images(
            filtered_json_path,
            tile_dict,
            filtered_output_folder,
            tile_size
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Convert dataset maps to preview images")
    parser.add_argument("--json-path", default="data/result.json")
    parser.add_argument("--tile-folder", default="tiles")
    parser.add_argument("--output-folder", default="map_images")
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument("--filtered-json-path", default="data/result.filtered.json")
    return parser.parse_args()


# -------------------------
# 执行
# -------------------------
if __name__ == "__main__":
    args = parse_args()
    filtered_json_path = args.filtered_json_path
    default_filtered_path = os.path.join("data", "result.filtered.json")
    if filtered_json_path is None and os.path.exists(default_filtered_path):
        filtered_json_path = default_filtered_path

    convert_dataset_to_images(
        json_path=args.json_path,
        tile_folder=args.tile_folder,
        output_folder=args.output_folder,
        tile_size=args.tile_size,
        filtered_json_path=filtered_json_path
    )