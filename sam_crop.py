import os
# import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


# def show_anns(anns):
#     if len(anns) < 2:
#         return
#     second_largest_ann = sorted(anns, key=(lambda x: x['area']), reverse=True)[1]
#     ax = plt.gca()
#     ax.set_autoscale_on(False)

#     img = np.ones((second_largest_ann['segmentation'].shape[0], second_largest_ann['segmentation'].shape[1], 4))
#     img[:, :, 3] = 0

#     m = second_largest_ann['segmentation']
#     color_mask = np.concatenate([np.random.random(3), [0.35]])
#     img[m] = color_mask
#     ax.imshow(img)


if __name__ == "__main__":
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda:1"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    # # image = cv2.imread('./carotid/13.jpg')
    # # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = Image.open('./carotid/13.jpg')
    # image_np = np.array(image)
    # masks = mask_generator.generate(image_np)
    # print(len(masks))
    # print(masks[0].keys())
    # print(masks[0].values())

    # 创建存储裁剪图片的文件夹
    output_folder = './carotid_crop'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取./carotid文件夹中的所有图像文件
    image_files = [f for f in os.listdir('./carotid') if f.lower().endswith('.jpg')]

    # 使用tqdm显示进度条
    for filename in tqdm(image_files, desc="Processing images"):
        # 读取和处理图像
        image_path = os.path.join('./carotid', filename)
        image = Image.open(image_path)
        image_np = np.array(image)

        # 生成掩码并找到第二大的注释
        masks = mask_generator.generate(image_np)
        second_largest_ann = sorted(masks, key=(lambda x: x['area']), reverse=True)[1]

        # 裁剪图像
        # x, y, w, h = second_largest_ann['bbox']
        x, y, w, h = map(int, second_largest_ann['bbox'])
        cropped_image = image_np[y:y+h, x:x+w]

        # 构建新的文件名和保存路径
        base_filename = os.path.splitext(filename)[0]
        cropped_image_path = os.path.join(output_folder, f'{base_filename}_crop.jpg')
        # plt.imsave(cropped_image_path, cropped_image)
        pil_image = Image.fromarray(np.uint8(cropped_image)).convert('RGB')
        pil_image.save(cropped_image_path)

    print("所有图像裁剪完成！")
