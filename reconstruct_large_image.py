import torch
from torchvision import transforms
from PIL import Image
import os
import math
from models.self_model import DivideNet_V3


def reconstruct():
    # --- 1. 配置参数 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "/home/wenhao/bishe_code/checkpoints/V3_Final/best_model_v3.pth"
    # 这里放你分割后的那些小图的文件夹
    input_dir = "/home/wenhao/bishe_code/bishe_DivideNet_photoes_Preprocessing"
    save_dir = "/home/wenhao/bishe_code/test_results/reconstructed"

    tile_size = 256
    stride = 256  # 必须与你切图脚本中的 stride 一致
    target_group = "48"  # 假设我们要还原第 48 组大图

    os.makedirs(save_dir, exist_ok=True)

    # --- 2. 加载模型 ---
    model = DivideNet_V3().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ 模型加载成功，准备还原第 {target_group} 组大图...")

    # --- 3. 筛选并排序该组的所有 blended 小图 ---
    # 匹配规则: {group_id}_{tile_count}_blended.png
    all_tiles = [f for f in os.listdir(input_dir) if f.startswith(f"{target_group}_") and "blended" in f]
    # 按照位置号排序，确保顺序正确
    all_tiles.sort(key=lambda x: int(x.split('_')[1]))

    if not all_tiles:
        print(f"❌ 未找到第 {target_group} 组的图片，请检查路径或组号。")
        return

    # --- 4. 自动计算大图尺寸 ---
    # 假设原图是根据 Camera1 的 BMP 尺寸切分的。
    # 我们需要通过最大的位置号来推算行列数（你的切图脚本是从左到右、从上到下）
    # 这里我们采用一个简单的方法：先模拟切图逻辑获取行列。
    # 提示：由于你是 50 组固定尺寸，建议直接查看原图宽 w 和高 h。
    # 如果不知道原图宽、高，我们从最大的 tile 索引反推：

    # 预处理转换
    transform = transforms.Compose([
        transforms.Resize((tile_size, tile_size)),
        transforms.ToTensor()
    ])
    to_pil = transforms.ToPILImage()

    # 临时打开第一张获取原图信息（这里假设你已知原图尺寸，或者通过脚本逻辑反推）
    # 针对你的切图脚本：原图 w, h
    # 我们先做一次推理循环，并直接根据位置拼接

    # 【注意】为了精准拼接，需要知道每行有多少个 tile。
    # 你的脚本逻辑：for top... for left...
    # 意味着 tile_count 是先变 left(行内)，再变 top(跨行)

    # 这里我们手动设定原图尺寸，或者从文件名推断。
    # 假设原图尺寸 2048x2048 (请根据你 BMP 的实际尺寸修改)
    W_LARGE, H_LARGE = 2048, 2048

    canvas_clear = Image.new('L', (W_LARGE, H_LARGE))
    canvas_blur = Image.new('L', (W_LARGE, H_LARGE))

    tile_idx = 0
    # 重新模拟切图时的坐标循环
    for top in range(0, H_LARGE - tile_size + 1, stride):
        for left in range(0, W_LARGE - tile_size + 1, stride):
            if tile_idx >= len(all_tiles):
                break

            # 获取对应的文件名
            tile_name = f"{target_group}_{tile_idx + 1}_blended.png"
            tile_path = os.path.join(input_dir, tile_name)

            if not os.path.exists(tile_path):
                tile_idx += 1
                continue

            # 推理
            img_pil = Image.open(tile_path).convert('L')
            img_tensor = transform(img_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                pred_clear, pred_blur = model(img_tensor)

            # 转回 PIL
            out_c_tile = to_pil(pred_clear.squeeze(0).cpu())
            out_b_tile = to_pil(pred_blur.squeeze(0).cpu())

            # 粘贴到大图
            canvas_clear.paste(out_c_tile, (left, top))
            canvas_blur.paste(out_b_tile, (left, top))

            tile_idx += 1
        print(f"🧱 正在拼接: 行坐标 {top} 已完成")

    # --- 5. 保存大图 ---
    canvas_clear.save(os.path.join(save_dir, f"Group_{target_group}_Full_Clear.png"))
    canvas_blur.save(os.path.join(save_dir, f"Group_{target_group}_Full_Blur.png"))
    print(f"✨ 大图还原完成！保存至: {save_dir}")


if __name__ == "__main__":
    reconstruct()