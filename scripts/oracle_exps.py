import sys
sys.path.append(sys.path[0]+'/..')
import glob
import os
import cv2
import mmengine
import numpy as np
import torch
from segment_anything import build_sam, SamPredictor, SamAutomaticMaskGenerator

folder = '/Users/kyanchen/datasets/Building/3. The cropped image tiles and raster labels/test'
checkpoint = '../pretrain/sam_vit_h_4b8939.pth'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

n_points = 1
mode = 'box'  # 'point' or 'box' or 'mask'

img_folder = os.path.join(folder, 'image')
mask_folder = os.path.join(folder, 'label')
save_folder = os.path.join(folder, f'result_{mode}_{n_points}')
mmengine.mkdir_or_exist(save_folder)
img_files = glob.glob(os.path.join(img_folder, '*.tif'))

predictor = SamPredictor(build_sam(checkpoint=checkpoint).to(device))

def generate_points_boxes(contours, mask):
    points = []
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 5 or h < 5:
            continue
        boxes.append([x, y, x + w, y + h])
    for box in boxes:
        x1, y1, x2, y2 = box
        tmp_points = []
        while len(tmp_points) < n_points:
            x = np.random.randint(x1, x2)
            y = np.random.randint(y1, y2)
            if mask[y, x] == 255:
                tmp_points.append([x, y])
        points.append(tmp_points)
    return boxes, points

for img_file in mmengine.track_iter_progress(img_files):
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    predictor.set_image(img)

    mask_file = os.path.join(mask_folder, os.path.basename(img_file))
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    contours, h = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 在二值化图像上搜索轮廓

    draw_img = img.copy()
    ret = cv2.drawContours(draw_img, contours, -1, (0, 0, 255), 2)
    # 第六步：画出带有轮廓的原始图片
    # cv2.imshow('ret', ret)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    gen_boxes, gen_points = generate_points_boxes(contours, mask)

    point_coords = None
    point_labels = None
    boxes = None
    mask_input = None
    multimask_output = False
    if mode == 'point':
        point_coords = gen_points
        point_labels = [[1] * len(x) for x in point_coords]
    elif mode == 'box':
        boxes = torch.tensor(gen_boxes, device=predictor.device)
        boxes = predictor.transform.apply_boxes_torch(boxes, img.shape[:2])
    elif mode == 'mask':
        pass

    if len(boxes) == 0:
        continue
    masks, scores, logits = predictor.predict_torch(
        point_coords=point_coords,
        point_labels=point_labels,
        mask_input=mask_input,
        boxes=boxes,
        multimask_output=False
    )
    save_mask = torch.any(masks, dim=0).squeeze().cpu().numpy().astype(np.uint8) * 255
    cv2.imwrite(os.path.join(save_folder, os.path.splitext(os.path.basename(img_file))[0] + '.png'), save_mask)
    save_mask = cv2.cvtColor(save_mask, cv2.COLOR_GRAY2BGR)
    save_mask = cv2.addWeighted(draw_img, 0.5, save_mask, 0.5, 0)
    cv2.imwrite(os.path.join(save_folder, os.path.splitext(os.path.basename(img_file))[0]+'.jpg'), save_mask)

