import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import os, glob, time
from tqdm import tqdm
from model import DivideNet_V3


# --- 数据集定义 ---
class SpeckleDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        # 获取所有混合图路径
        mix_images = sorted(glob.glob(os.path.join(root_dir, "*_blended.png")))
        for m_path in mix_images:
            base = os.path.basename(m_path).replace("_blended.png", "")
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = "/home/wenhao/bishe_code/checkpoints/V3_Final"
    os.makedirs(save_dir, exist_ok=True)

    # 基础预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # --- 核心修改：划分数据集 ---
    full_dataset = SpeckleDataset("/home/wenhao/bishe_code/bishe_DivideNet_photoes_Preprocessing", transform=transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4)

    model = DivideNet_V3().to(device)
    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=2e-4)

    best_val_loss = float('inf')

    for epoch in range(50):
        start_time = time.time()

        # --- 训练阶段 ---
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]")
        for m, c, b in pbar:
            m, c, b = m.to(device), c.to(device), b.to(device)
            pc, pb = model(m)

            # 1. 重建损失 (L1权重加大，解决灰度不够黑的问题)
            l_pix = (criterion_mse(pc, c) + 1.0 * criterion_l1(pc, c)) + \
                    (criterion_mse(pb, b) + 1.0 * criterion_l1(pb, b))

            # 2. 物理一致性 (降低权重，防止模型通过合体来偷懒)
            l_sum = criterion_mse(pc + pb, m) * 0.5

            # 3. 掩码互斥损失 (解决阴影残留)
            mask_c_only = (c < 0.4) & (b > 0.8)  # 只有清晰斑点
            mask_b_only = (b < 0.4) & (c > 0.8)  # 只有模糊斑点
            l_excl = 0
            if mask_c_only.any(): l_excl += torch.mean(torch.abs(1 - pb[mask_c_only]))
            if mask_b_only.any(): l_excl += torch.mean(torch.abs(1 - pc[mask_b_only]))

            loss = l_pix + l_sum + 5.0 * l_excl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # --- 验证阶段 ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for m, c, b in val_loader:
                m, c, b = m.to(device), c.to(device), b.to(device)
                pc, pb = model(m)
                # 验证集只计算基础的重建误差，作为性能指标
                v_l = criterion_mse(pc, c) + criterion_mse(pb, b)
                val_loss += v_l.item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)

        # 打印耗时和双端 Loss
        duration = time.time() - start_time
        print(f"✅ Epoch {epoch + 1} | Time: {duration:.1f}s | TrainLoss: {avg_train:.6f} | ValLoss: {avg_val:.6f}")

        # 保存最优模型
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model_v3.pth"))
            print("🌟 验证集表现提升，已保存最佳权重。")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"v3_epoch_{epoch + 1}.pth"))


if __name__ == "__main__":
    train()