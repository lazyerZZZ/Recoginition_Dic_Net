import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import os, glob, time
from tqdm import tqdm

# --- 1. 直接从你的 self_model.py 调用模型 ---
from models.self_model import DeblurUNet


# --- 2. 数据集定义 (精准匹配你的命名规则) ---
class DeblurDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        # 匹配 deblur_*_blurred.png
        blur_pattern = os.path.join(root_dir, "*_blurred.png")
        blur_imgs = sorted(glob.glob(blur_pattern))

        for b_path in blur_imgs:
            # 自动推导对应的 _sharp.png
            s_path = b_path.replace("_blurred.png", "_sharp.png")
            if os.path.exists(s_path):
                self.samples.append((b_path, s_path))

        print(f"📦 数据准备就绪: 已找到 {len(self.samples)} 对去模糊样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        b_p, s_p = self.samples[idx]
        # DIC 任务建议转为灰度 'L'
        b_img = self.transform(Image.open(b_p).convert('L'))
        s_img = self.transform(Image.open(s_p).convert('L'))
        return b_img, s_img


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 配置区域 ---
    # 路径请根据实际修改
    data_path = "/home/wenhao/bishe_code/bishe_DeblurringNet_Preprocessing"
    save_dir = "/home/wenhao/bishe_code/checkpoints/Deblur_V1"
    os.makedirs(save_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # --- 数据划分 ---
    full_ds = DeblurDataset(data_path, transform)
    train_size = int(0.8 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4)

    # --- 模型初始化 ---
    model = DeblurUNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_val_loss = float('inf')

    # --- 训练循环 ---
    for epoch in range(50):
        start_time = time.time()

        # 1. 训练
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1} [Train]")
        for b, s in pbar:
            b, s = b.to(device), s.to(device)
            pred = model(b)
            loss = criterion(pred, s)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 2. 验证 (Val_loader 启动！)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for b, s in val_loader:
                b, s = b.to(device), s.to(device)
                pred = model(b)
                v_l = criterion(pred, s)
                val_loss += v_l.item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        duration = time.time() - start_time

        print(f"✨ Epoch {epoch + 1} | Time: {duration:.1f}s | TrainLoss: {avg_train:.6f} | ValLoss: {avg_val:.6f}")

        # 3. 最佳模型保存
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), os.path.join(save_dir, "best_deblur_model.pth"))
            print("🌟 验证集表现更优，已更新最优权重。")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"deblur_epoch_{epoch + 1}.pth"))


if __name__ == "__main__":
    train()