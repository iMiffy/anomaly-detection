import os
import cv2 
from imgaug import augmenters as iaa
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True,
	help="path to input dataset")
ap.add_argument("-b", "--copies", type=int, default=2,
	help="copies of images to be created via augmentation")
args = vars(ap.parse_args())


def augment(image_np, copies, image_name, save_path):
    aug = iaa.SomeOf((1,3), [iaa.Fliplr(0.5), iaa.Flipud(0,5), iaa.Affine(rotate=(-90,90)), 
                      iaa.GammaContrast((0.5, 2.0), per_channel=True)], random_order=True)  
    image_aug = aug(images=[image_np]*copies)
    i = 1
    for each_im in image_aug:
        cv2.imwrite(os.path.join(save_path, image_name[:-4] + '_augment_' + str(i) + '.jpg'), each_im)
        i += 1
  
parent_dir = os.path.abspath(os.path.join(args['path'], os.pardir))
HSV_path = os.path.join(parent_dir, 'HSV_version')
LAB_path = os.path.join(parent_dir, 'LAB_version')

os.makedirs(HSV_path)
os.makedirs(LAB_path)
os.chdir(args['path'])


for each_image in os.listdir():
    
    image = cv2.imread(each_image)
    resized = cv2.resize(image, (512,512), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(each_image, resized)
    
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
    cv2.imwrite(os.path.join(HSV_path, each_image[:-4] + '_HSV.jpg'), hsv)
    cv2.imwrite(os.path.join(LAB_path, each_image[:-4] + '_LAB.jpg'), lab)

    augment(hsv, args['copies'], each_image, HSV_path)
    augment(lab, args['copies'], each_image, LAB_path)
    augment(resized, args['copies'], each_image, args['path'])
