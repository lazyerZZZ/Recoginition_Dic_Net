import torch
from torchvision import transforms
from PIL import Image
import os
from models.self_model import DivideNet_V3  # 确保 self_model.py 在同级目录


def test():
    # --- 1. 配置参数 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "/home/wenhao/bishe_code/checkpoints/V3_Final/best_model_v3.pth"
    input_path = "/home/wenhao/bishe_code/bishe_DivideNet_photoes_Preprocessing/48_34_blended.png"  # 换成你想测试的图
    save_dir = "/home/wenhao/bishe_code/test_results/21"
    os.makedirs(save_dir, exist_ok=True)

    # --- 2. 加载模型 ---
    model = DivideNet_V3().to(device)
    # 加载保存的权重
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()  # 切换到预测模式
    print(f"✅ 权重加载成功: {model_path}")

    # --- 3. 预处理图片 ---
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    img_pil = Image.open(input_path).convert('L')
    img_tensor = transform(img_pil).unsqueeze(0).to(device)  # 增加 batch 维度 [1, 1, 256, 256]

    # --- 4. 推理过程 ---
    with torch.no_grad():
        # 注意：这里的返回顺序必须和你的 model.forward 保持一致 (out_clear, out_blur)
        pred_clear, pred_blur = model(img_tensor)

    # --- 5. 保存结果 ---
    # 将 Tensor 转回 PIL Image
    to_pil = transforms.ToPILImage()

    # .squeeze(0) 去掉 batch 维度，.cpu() 移回内存
    out_c_img = to_pil(pred_clear.squeeze(0).cpu())
    out_b_img = to_pil(pred_blur.squeeze(0).cpu())

    # 保存图片
    base_name = os.path.basename(input_path).replace(".png", "")
    out_c_img.save(os.path.join(save_dir, f"{base_name}_pred_clear.png"))
    out_b_img.save(os.path.join(save_dir, f"{base_name}_pred_blur.png"))

    print(f"🚀 测试完成！结果已保存至: {save_dir}")


if __name__ == "__main__":
    test()