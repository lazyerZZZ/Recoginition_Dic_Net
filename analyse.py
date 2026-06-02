import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def evaluate_images(original_path, generated_path):
    # 1. 读取图像
    # 注意：确保两张图尺寸完全一致
    img_orig = cv2.imread(original_path)
    img_gen = cv2.imread(generated_path)

    if img_orig is None or img_gen is None:
        raise ValueError("图片路径有误，请检查路径。")

    # 2. 预处理：统一尺寸（如果由于网络输出导致尺寸微差异，需Resize）
    if img_orig.shape != img_gen.shape:
        img_gen = cv2.resize(img_gen, (img_orig.shape[1], img_orig.shape[0]))

    # 3. 计算 PSNR
    # data_range=255 表示像素值范围是 0-255
    psnr_val = psnr(img_orig, img_gen, data_range=255)

    # 4. 计算 SSIM
    # multichannel=True 用于处理彩色图像 (RGB)
    # 对于较新版本的 skimage，参数名可能改为 channel_axis=2
    ssim_val = ssim(img_orig, img_gen, channel_axis=2, data_range=255)

    return psnr_val, ssim_val


# --- 使用示例 ---
original = "/home/wenhao/bishe_code/bishe_DivideNet_photoes_Preprocessing/48_34_clear.png"
generated = "/home/wenhao/bishe_code/test_results/21/48_34_blended_pred_clear.png"
p_score, s_score = evaluate_images(original, generated)

print(f"PSNR: {p_score:.4f} dB")
print(f"SSIM: {s_score:.4f}")
