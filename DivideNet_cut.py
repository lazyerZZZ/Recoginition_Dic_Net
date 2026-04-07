import os
from PIL import Image


def generate_deep_dic_dataset(input_dir, output_dir, tile_size=256, stride=256):
    """
    针对 50 组 BMP 原图生成 256x256 训练集的脚本
    """
    # 1. 准备文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✅ 已创建输出文件夹: {output_dir}")

    # 2. 获取并排序所有 .bmp 文件
    all_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.bmp')])

    print(f"🔍 扫描完成：共检测到 {len(all_files)} 张 .bmp 大图")
    if len(all_files) % 3 != 0:
        print("⚠️ 警告：文件总数不是 3 的倍数，请检查图片是否成组（重叠/清晰/模糊）！")

    # 定义子图后缀名，方便 PyTorch 读取
    types = {0: "blended", 1: "clear", 2: "blurred"}
    total_samples = 0

    # 3. 三张一组循环处理
    for i in range(0, len(all_files), 3):
        group_files = all_files[i: i + 3]
        if len(group_files) < 3: break

        group_id = (i // 3) + 1  # 对应第 1-50 组

        # 打开三张大图
        try:
            imgs = [Image.open(os.path.join(input_dir, f)) for f in group_files]
            w, h = imgs[0].size

            tile_count = 1
            # 4. 滑动窗口剪切
            # range(start, stop, step) 确保了不会超出边界
            for top in range(0, h - tile_size + 1, stride):
                for left in range(0, w - tile_size + 1, stride):

                    # 针对每一组采样点，同时切出三张小图
                    for idx, img in enumerate(imgs):
                        box = (left, top, left + tile_size, top + tile_size)
                        tile = img.crop(box)

                        # 保存为 .png 格式（无损压缩，节省空间）
                        # 命名逻辑：组号_位置号_类别.png
                        save_name = f"{group_id}_{tile_count}_{types[idx]}.png"
                        tile.save(os.path.join(output_dir, save_name))

                    tile_count += 1

            total_samples += (tile_count - 1)
            print(f"📈 进度: {group_id}/50 | 本组生成: {tile_count - 1} 组 | 累计: {total_samples}")

        except Exception as e:
            print(f"❌ 处理第 {group_id} 组时出错: {e}")

    print("-" * 30)
    print(f"✨ 任务成功完成！")
    print(f"📁 数据集保存至: {output_dir}")
    print(f"📊 最终训练样本组数: {total_samples}")


# ================= 配置路径 =================
# r"" 表示原始字符串，防止 Windows 路径反斜杠转义
input_path = r"E:\bishe_Divide_photoes\Camera1"
output_path = r"E:\bishe_DivideNet_photoes_Preprocessing"

# ================= 执行 =================
generate_deep_dic_dataset(input_path, output_path, tile_size=256, stride=256)