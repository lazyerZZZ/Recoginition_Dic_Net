import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import os, glob, time
from tqdm import tqdm

# 导入刚才为你设计的 V4 模型
# 假设你把 DivideNet_V4 保存在 models/self_model.py 中
from models.self_model import DivideNet_V4


# --- 数据集定义 ---
class SpeckleDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        # 获取所有混合图路径
        mix_images = sorted(glob.glob(os.path.join(root_dir, "*_blended.png")))
        for m_path in mix_images:
            base = os.path.basename(m_path).replace("_blended.png", "")
            # 兼容你之前的命名习惯：_clear 和 _blurred
            self.samples.append({
                "mix": m_path,
                "clear": os.path.join(root_dir, f"{base}_clear.png"),
                "blur": os.path.join(root_dir, f"{base}_blurred.png")
            })
        self.transform = transform

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        m = self.transform(Image.open(s["mix"]).convert('L'))
        c = self.transform(Image.open(s["clear"]).convert('L'))
        b = self.transform(Image.open(s["blur"]).convert('L'))
        return m, c, b


def train():
    # --- 1. 环境与路径配置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 建议换个文件夹保存，防止覆盖 V3 的权重
    save_dir = "/home/wenhao/bishe_code/checkpoints/V4_UNet_Final"
    os.makedirs(save_dir, exist_ok=True)

    # 预处理：DIC 图像通常需要保持严格的分辨率
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # --- 2. 加载数据集 ---
    data_path = "/home/wenhao/bishe_code/bishe_DivideNet_photoes_Preprocessing"
    full_dataset = SpeckleDataset(data_path, transform=transform)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    # --- 3. 模型、损失函数与优化器 ---
    model = DivideNet_V4(in_channels=1, out_channels=1).to(device)

    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    # 初始学习率 2e-4 是 Adam 的黄金配置
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    # 建议增加学习率衰减，让后期收敛更平稳
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    best_val_loss = float('inf')

    # --- 4. 训练循环 ---
    for epoch in range(50):
        start_time = time.time()
        model.train()
        train_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]")
        for m, c, b in pbar:
            m, c, b = m.to(device), c.to(device), b.to(device)

            # 前向传播
            pc, pb = model(m)

            # --- 损失计算逻辑（承袭你的 V3 经验） ---
            # 1. 像素级重建损失 (L1 + MSE 结合，保持对比度和边缘)
            l_pix_c = criterion_mse(pc, c) + 1.0 * criterion_l1(pc, c)
            l_pix_b = criterion_mse(pb, b) + 1.0 * criterion_l1(pb, b)
            l_pix = l_pix_c + l_pix_b

            # 2. 物理一致性损失 (保证 pc + pb = m)
            l_sum = criterion_mse(pc + pb, m) * 0.5

            # 3. 掩码互斥损失 (利用你的经验解决阴影残留)
            # 逻辑：在清晰图该黑的地方，模糊图如果不白，就罚它。反之亦然。
            mask_c_only = (c < 0.4) & (b > 0.8)
            mask_b_only = (b < 0.4) & (c > 0.8)
            l_excl = 0
            if mask_c_only.any(): l_excl += torch.mean(torch.abs(1 - pb[mask_c_only]))
            if mask_b_only.any(): l_excl += torch.mean(torch.abs(1 - pc[mask_b_only]))

            # 最终总损失
            loss = l_pix + l_sum + 5.0 * l_excl

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # --- 5. 验证阶段 ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for m, c, b in val_loader:
                m, c, b = m.to(device), c.to(device), b.to(device)
                pc, pb = model(m)
                # 验证集使用 MSE 作为核心评估指标
                v_l = criterion_mse(pc, c) + criterion_mse(pb, b)
                val_loss += v_l.item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)

        # 更新学习率策略
        scheduler.step(avg_val)

        # 打印日志
        duration = time.time() - start_time
        print(f"✅ Epoch {epoch + 1} | Time: {duration:.1f}s | Train: {avg_train:.6f} | Val: {avg_val:.6f}")

        # --- 6. 保存逻辑 ---
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model_v4.pth"))
            print("🌟 验证集表现提升，已保存 V4 最佳权重。")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"v4_epoch_{epoch + 1}.pth"))


if __name__ == "__main__":
    train()