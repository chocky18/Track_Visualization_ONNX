import numpy as np
import cv2
import os

dir_path = os.path.dirname(os.path.abspath(__file__))
rail_sem_colors_info_path ="rail_sem_colors.txt"

def read_rail_sem_info(info_path=rail_sem_colors_info_path):
	with open(info_path) as fp:
		lines = fp.readlines()

		labels = [line[:-1].replace(';', ',').split(',')[0] for line in lines]
		colors = np.array([line[:-1].replace(';', ',').split(',')[-3:] for line in lines]).astype(np.int)

	return colors, labels

colors, labels = read_rail_sem_info()
	
def util_draw_seg(seg_map, image, alpha = 0.5):

	# Convert segmentation prediction to colors
	color_segmap = cv2.resize(image, (seg_map.shape[1], seg_map.shape[0]))
	color_segmap[seg_map>0] = colors[seg_map[seg_map>0]]

	# Resize to match the image shape
	color_segmap = cv2.resize(color_segmap, (image.shape[1],image.shape[0]))

	# Fuse both images
	if(alpha == 0):
		combined_img = np.hstack((image, color_segmap))
	else:
		combined_img = cv2.addWeighted(image, alpha, color_segmap, (1-alpha),0)

	return combined_img