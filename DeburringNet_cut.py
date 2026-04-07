import os
from PIL import Image


def generate_deblur_dataset(input_dir, output_dir, tile_size=256, stride=256):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取所有 .bmp 文件并排序
    all_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.bmp')])

    print(f"🔍 检测到 {len(all_files)} 张 .bmp 图片（预计 84 组）")

    # 定义去模糊任务的类型：第一张清晰，第二张模糊
    types = {0: "sharp", 1: "blurred"}
    total_samples = 0

    # 按照 2 张一组进行迭代
    for i in range(0, len(all_files), 2):
        group_files = all_files[i: i + 2]
        if len(group_files) < 2: break

        group_id = (i // 2) + 1
        imgs = [Image.open(os.path.join(input_dir, f)) for f in group_files]
        w, h = imgs[0].size

        tile_count = 1
        for top in range(0, h - tile_size + 1, stride):
            for left in range(0, w - tile_size + 1, stride):

                for idx, img in enumerate(imgs):
                    box = (left, top, left + tile_size, top + tile_size)
                    tile = img.crop(box)

                    # 命名格式：deblur_组号_位置_类型.png
                    save_name = f"deblur_{group_id}_{tile_count}_{types[idx]}.png"
                    tile.save(os.path.join(output_dir, save_name))

                tile_count += 1

        total_samples += (tile_count - 1)
        if group_id % 10 == 0:
            print(f"📈 去模糊数据集进度: {group_id}/84 | 累计生成: {total_samples}")

    print(f"\n✨ 去模糊数据集制作完成！共计 {total_samples} 组训练样本。")


# ================= 配置路径 =================
# 建议为去模糊任务单独建一个输出文件夹
input_path_deblur = r"E:\bishe_Deblur_photoes\Camera1"
output_path_deblur = r"E:\bishe_Deblur_Preprocessing"

generate_deblur_dataset(input_path_deblur, output_path_deblur, tile_size=256, stride=256)