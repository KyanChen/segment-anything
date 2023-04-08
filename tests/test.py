import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

from segment_anything import build_sam, SamPredictor, SamAutomaticMaskGenerator

checkpoint = '../pretrain/sam_vit_h_4b8939.pth'
img_file = '/Users/kyanchen/datasets/Building/3. The cropped image tiles and raster labels/train/image/94.tif'

img = cv2.imread(img_file)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#
# def show_anns(anns):
#     if len(anns) == 0:
#         return
#     sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
#     ax = plt.gca()
#     ax.set_autoscale_on(False)
#     polygons = []
#     color = []
#     for ann in sorted_anns:
#         m = ann['segmentation']
#         img = np.ones((m.shape[0], m.shape[1], 3))
#         color_mask = np.random.random((1, 3)).tolist()[0]
#         for i in range(3):
#             img[:,:,i] = color_mask[i]
#         ax.imshow(np.dstack((img, m*0.35)))
#
# mask_generator = SamAutomaticMaskGenerator(build_sam(checkpoint=checkpoint))
# masks = mask_generator.generate(img)
#
# plt.figure(figsize=(20,20))
# plt.imshow(img)
# show_anns(masks)
# plt.axis('off')
# plt.show()


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


predictor = SamPredictor(build_sam(checkpoint=checkpoint))
predictor.set_image(img)
# input_point = np.array([[200, 400]])
# input_label = np.array([1])
input_point = np.array([[200, 400], [210, 410]])
input_label = np.array([1, 1])

plt.figure(figsize=(10, 10))
plt.imshow(img)
show_points(input_point, input_label, plt.gca())
plt.axis('on')
plt.show()


masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=False,
)

for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()