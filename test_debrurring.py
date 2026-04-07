import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os


# 确保从你的模型文件中导入 DeblurUNet 和依赖的 DoubleConv
from model import DeblurUNet

def test_single_image():
    # --- 1. 配置路径 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 精确到单张图片的绝对路径
    input_image_path = '/home/wenhao/bishe_code/test_results/6/48_34_blended_pred_blur.png'
    model_path = '/home/wenhao/bishe_code/checkpoints/Deblur_V1/best_deblur_model.pth'
    save_dir = '/home/wenhao/bishe_code/results'

    os.makedirs(save_dir, exist_ok=True)

    # --- 2. 检查文件是否存在 ---
    if not os.path.exists(input_image_path):
        print(f"错误: 找不到输入图片 {input_image_path}")
        return

    # --- 3. 加载模型 ---
    model = DeblurUNet().to(device)

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        # 兼容处理不同的保存格式
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            model.load_state_dict(state_dict['model_state_dict'])
        else:
            model.load_state_dict(state_dict)
        print(f"模型权重已加载: {model_path}")
    else:
        print("警告: 未找到预训练权重，将使用随机初始化的模型进行测试。")

    model.eval()

    # --- 4. 预处理 (适配你的单通道 U-Net) ---
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 必须是 8 的倍数
        transforms.ToTensor(),
    ])

    # --- 5. 推理 ---
    with torch.no_grad():
        # 加载并转为灰度图 ('L')，因为你的 enc1 输入通道是 1
        img_pil = Image.open(input_image_path).convert('L')
        orig_size = img_pil.size  # 记录原始尺寸用于还原 (可选)

        input_tensor = transform(img_pil).unsqueeze(0).to(device)  # [1, 1, 256, 256]

        # 模型前向传播
        output = model(input_tensor)

        # 后处理
        output = output.squeeze(0).cpu()  # [1, 256, 256]

        # 如果你想把结果还原回图片原始大小，取消下面这一行的注释：
        # output = nn.functional.interpolate(output.unsqueeze(0), size=orig_size[::-1], mode='bilinear').squeeze(0)

        output_img = transforms.ToPILImage()(output)

        # 保存结果
        file_name = os.path.basename(input_image_path)
        save_path = os.path.join(save_dir, f"restored_{file_name}")
        output_img.save(save_path)

    print(f"\n处理完成！")
    print(f"输入路径: {input_image_path}")
    print(f"保存路径: {save_path}")


if __name__ == '__main__':
    test_single_image()