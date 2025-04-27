import os
import torch
from flask import Flask, request, jsonify, send_from_directory

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

generator = None

# ====== Flask部分 ======
app = Flask(__name__, static_folder='../frontend/dist', static_url_path='')

@app.route('/')
def serve_index():
    # 返回 Vue 打包后的 index.html
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/generate', methods=['POST'])
def generate_map():
    """
    接收请求，调用模型生成地图，返回JSON
    """
    try:
        # 你可以根据需要从request.json里读取参数
        params = request.json or {}

        # 假设你的模型输入是随机噪声或固定向量
        # 下面是示例代码，根据你真实情况修改
        noise = torch.randn(1, 1024, device=device)
        with torch.no_grad():
            output = generator(noise)

        # 假设output是 [B, H, W] 分类结果
        # 转为CPU，转成Python list
        map_data = output.argmax(dim=1).squeeze(0).cpu().tolist()

        return jsonify({
            'status': 'success',
            'map': map_data
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.errorhandler(404)
def not_found(e):
    # 如果找不到文件，返回前端index.html（前端路由支持）
    return send_from_directory(app.static_folder, 'index.html')

# ====== 启动 ======
if __name__ == '__main__':
    # 检查模型
    if os.path.exists("../model/ginka"):
        from ..model.ginka.model import GinkaModel
        generator = GinkaModel()
        generator.to(device)
        generator.eval()
        state = torch.load("../model/ginka.pth", map_location=device)
        generator.load_state_dict(state["model_state"])
        app.run(host='0.0.0.0', port=3444, debug=True)
    else:
        print("未找到模型定义，请先下载模型并命名为 ginka，放置在 model 文件夹中！")
