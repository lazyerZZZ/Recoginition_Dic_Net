import torch
import torch.nn.functional as F
import numpy as np
from imageio import imread
from path import Path
import models  # 确保你的 StrainNet 代码库在当前目录下


# --- 1. 核心配置 ---
class Config:
    arch = 'StrainNet_f'  # 你的模型类型：StrainNet_l, _f, 或 _h
    pretrained = '/home/wenhao/bishe_code/checkpoints/StrainNet-f.pth.tar'  # 权重路径
    div_flow = 2.0  # 官方默认缩放因子
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def run_strainnet():
    # --- 2. 加载模型 ---
    print(f"=> 正在初始化网络: {Config.arch}")
    network_data = torch.load(Config.pretrained, map_location=Config.device)
    # models.__dict__ 会根据字符串 'StrainNet_l' 自动寻找对应的类
    model = models.__dict__[Config.arch](network_data).to(Config.device)
    model.eval()

    # --- 3. 指定输入图片 (你分离网络输出的两张图) ---
    # 请修改为你实际的图片路径
    ref_img_path = '/home/wenhao/bishe_code/test_results/5/45_33_blended_pred_blur.png'
    def_img_path = '/home/wenhao/bishe_code/test_results/5/45_33_blended_pred_clear.png'

    print(f"=> 读取图片:\n   Ref: {ref_img_path}\n   Def: {def_img_path}")
    img1 = imread(ref_img_path).astype(np.float32) / 255.0
    img2 = imread(def_img_path).astype(np.float32) / 255.0

    # --- 4. 预处理 (适配 StrainNet 的输入维度) ---
    # 如果是灰度图 (H, W)，增加通道维和 Batch 维
    if img1.ndim == 2:
        t1 = torch.from_numpy(img1).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        t2 = torch.from_numpy(img2).unsqueeze(0).unsqueeze(0)
        # StrainNet_f 和 _h 强制要求 3 通道输入
        if Config.arch in ['StrainNet_f', 'StrainNet_h']:
            t1 = torch.cat([t1, t1, t1], dim=1)
            t2 = torch.cat([t2, t2, t2], dim=1)
        input_var = torch.cat([t1, t2], dim=1).to(Config.device)  # [1, 2或6, H, W]
    else:
        # 如果是彩色图，调整维度为 [1, 6, H, W]
        t1 = torch.from_numpy(img1.transpose(2, 0, 1)).unsqueeze(0)
        t2 = torch.from_numpy(img2.transpose(2, 0, 1)).unsqueeze(0)
        input_var = torch.cat([t1, t2], dim=1).to(Config.device)

    # --- 5. 执行推理 ---
    output = model(input_var)

    # 如果是 _h 或 _l 版本，输出是原图的一半，需要上采样还原尺寸
    if Config.arch in ['StrainNet_h', 'StrainNet_l']:
        output = F.interpolate(output, scale_factor=2, mode='bilinear', align_corners=False)

    # --- 6. 转换结果并保存 ---
    disp = output.cpu().numpy()[0]
    # 官方公式转换：将网络输出转换为像素位移
    u = - disp[0, :, :] * Config.div_flow + 1
    v = - disp[1, :, :] * Config.div_flow + 1

    save_dir = Path('/home/wenhao/bishe_code/results/StrainNet_final_output')
    save_dir.makedirs_p()

    np.savetxt(save_dir / 'displacement_u.csv', u, delimiter=',')
    np.savetxt(save_dir / 'displacement_v.csv', v, delimiter=',')

    print(f"完成！全场位移场已保存至: {save_dir}")


if __name__ == '__main__':
    run_strainnet()