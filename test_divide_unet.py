import torch
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from models.self_model import DivideNet_V3


def test_v4_final():
    # --- 1. 路径与设备配置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "/home/wenhao/bishe_code/checkpoints/V4_UNet_Final/best_model_v4.pth"
    input_image = "/home/wenhao/bishe_code/bishe_DivideNet_photoes_Preprocessing/1_1_blended.png"

    # 结果保存目录
    save_dir = "/home/wenhao/bishe_code/test_results/14"
    os.makedirs(save_dir, exist_ok=True)

    # --- 2. 加载 V4 模型 ---
    model = DivideNet_V4().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"✅ V4 Weights Loaded Successfully.")

    # --- 3. 预处理 ---
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    img_pil = Image.open(input_image).convert('L')
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    # --- 4. 推理 (Inference) ---
    with torch.no_grad():
        # 假设 V4 返回两个 Tensor: (清晰, 模糊)
        pred_clear, pred_blur = model(img_tensor)

    # --- 5. 结果转换与保存 ---
    def save_tensor_as_img(tensor, name):
        # 降维、限制范围、转为 0-255 uint8
        img_np = tensor.squeeze().cpu().clamp(0, 1).numpy()
        img_uint8 = (img_np * 255).astype(np.uint8)
        img_obj = Image.fromarray(img_uint8)

        full_path = os.path.join(save_dir, name)
        img_obj.save(full_path)
        return img_np

    # 分别保存三张独立的图片
    # 1. 原始输入图 (已缩放至 256x256)
    input_np = save_tensor_as_img(img_tensor, "0_original_input.png")
    # 2. 预测的清晰图 (供后续 StrainNet 使用)
    clear_np = save_tensor_as_img(pred_clear, "1_pred_clear.png")
    # 3. 预测的模糊图
    blur_np = save_tensor_as_img(pred_blur, "2_pred_blur.png")

    # --- 6. (可选) 生成一张水平拼接的对比图，方便直接查看效果 ---
    combined = np.hstack((input_np, clear_np, blur_np))
    combined_img = Image.fromarray((combined * 255).astype(np.uint8))
    combined_img.save(os.path.join(save_dir, "3_combined_comparison.png"))

    print(f"🚀 测试完成！")
    print(f"📁 结果文件已保存在: {save_dir}")
    print(f"   - 原始图: 0_original_input.png")
    print(f"   - 清晰图: 1_pred_clear.png")
    print(f"   - 模糊图: 2_pred_blur.png")


if __name__ == "__main__":
    test_v4_final()