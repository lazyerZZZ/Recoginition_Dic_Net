import torch
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from torchvision import transforms
import os

# 假设模型类定义在 models 文件夹中
from models.self_model import DivideNet_V4
# 如果你的 StrainNet 结构不同，请替换为对应的类
from models.StrainNetF import StrainNet_f


class StereoReconstructor:
    def __init__(self, div_path, strain_path, K_raw, orig_size=(1280, 720), device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 1. 加载模型
        self.div_model = DivideNet_V4().to(self.device)
        self.div_model.load_state_dict(torch.load(div_path, map_location=self.device))
        self.div_model.eval()

        self.strain_model = StrainNet_f().to(self.device)
        self.strain_model.load_state_dict(torch.load(strain_path, map_location=self.device))
        self.strain_model.eval()

        # 2. 缩放内参 K (从原始分辨率缩放到 256x256)
        self.K = self._scale_camera_intrinsic(K_raw, orig_size, (256, 256))

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def _scale_camera_intrinsic(self, K, orig_size, new_size):
        """将标定时得到的内参缩放到模型输入尺寸"""
        sw = new_size[0] / orig_size[0]
        sh = new_size[1] / orig_size[1]
        K_new = K.copy()
        K_new[0, 0] *= sw  # fx
        K_new[1, 1] *= sh  # fy
        K_new[0, 2] *= sw  # cx
        K_new[1, 2] *= sh  # cy
        return K_new

    def compute_relative_pose(self, R1, T1, R2, T2):
        """通过两个独立标定位姿计算相对位姿 (以左镜为原点)"""
        # R_rel = R2 * R1^T
        R_rel = R2 @ R1.T
        # T_rel = T2 - R_rel * T1
        T_rel = T2 - R_rel @ T1
        return R_rel, T_rel

    def run(self, img_path, R1, T1, R2, T2):
        # 1. 读取并分离图像
        img = Image.open(img_path).convert('L')
        input_tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # 分离出左(Clear)和右(Blur)
            c_img, b_img = self.div_model(input_tensor)

            # 计算视差场 (u, v)
            disp = self.strain_model(c_img, b_img)
            u = disp[0, 0].cpu().numpy()
            v = disp[0, 1].cpu().numpy()

        # 2. 获取相对位姿并构建投影矩阵
        R_rel, T_rel = self.compute_relative_pose(R1, T1, R2, T2)

        P1 = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = self.K @ np.hstack((R_rel, T_rel))

        # 3. 准备三角重构点对
        H, W = u.shape
        x, y = np.meshgrid(np.arange(W), np.arange(H))
        pts1 = np.stack([x, y], axis=-1).reshape(-1, 2).astype(np.float32).T
        pts2 = np.stack([x + u, y + v], axis=-1).reshape(-1, 2).astype(np.float32).T

        # 4. 执行三角重构
        pts4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
        pts3d = (pts4d[:3] / pts4d[3]).T

        # 5. 过滤掉无效点 (深度限制)
        mask = (pts3d[:, 2] > 0) & (pts3d[:, 2] < 2000)
        return pts3d[mask]


# --- 运行示例 ---
if __name__ == "__main__":
    # --- 填入你的标定数据 ---
    # 原始 1280x720 标定得到的内参
    K_orig = np.array([
        [1250.5, 0, 640.2],
        [0, 1250.8, 360.5],
        [0, 0, 1]
    ])

    # 遮住右镜标定左镜得到的位姿 (示例)
    R1 = np.eye(3)
    T1 = np.array([[0], [0], [500.0]])  # 假设距离标定板 500mm

    # 遮住左镜标定右镜得到的位姿 (示例)
    R2 = np.eye(3)
    T2 = np.array([[-65.0], [0], [505.0]])  # 假设水平偏移了 65mm

    # 初始化逻辑
    pipeline = StereoReconstructor(
        div_path='checkpoints/V4_UNet_Final/best_model_v4.pth',
        strain_path='/home/wenhao/bishe_code/checkpoints/StrainNet-f.pth.tar',
        K_raw=K_orig,
        orig_size=(1280, 720)
    )

    # 执行并保存
    points = pipeline.run("data/test_overlap.png", R1, T1, R2, T2)
    np.savetxt("output_cloud.obj", points, fmt='v %.4f %.4f %.4f')
    print(f"🎉 重建完成！生成了 {len(points)} 个空间点，已保存至 output_cloud.obj")