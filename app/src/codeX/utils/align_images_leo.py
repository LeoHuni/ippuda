import os
import sys
import bz2
import argparse
# from face_alignment import image_align
# from landmarks_detector import LandmarksDetector
import multiprocessing
from streamlit_app import *

def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path

import numpy as np
import scipy.ndimage
import os
import PIL.Image


def image_align(img_name,src_file, dst_file, face_landmarks, output_size=1024, transform_size=4096, enable_padding=True, x_scale=1, y_scale=1, em_scale=0.1, alpha=False):
        # Align function from FFHQ dataset pre-processing step
        # https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py

        lm = np.array(face_landmarks)
        lm_chin          = lm[0  : 17]  # left-right
        lm_eyebrow_left  = lm[17 : 22]  # left-right
        lm_eyebrow_right = lm[22 : 27]  # left-right
        lm_nose          = lm[27 : 31]  # top-down
        lm_nostrils      = lm[31 : 36]  # top-down
        lm_eye_left      = lm[36 : 42]  # left-clockwise
        lm_eye_right     = lm[42 : 48]  # left-clockwise
        lm_mouth_outer   = lm[48 : 60]  # left-clockwise
        lm_mouth_inner   = lm[60 : 68]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left     = np.mean(lm_eye_left, axis=0)
        eye_right    = np.mean(lm_eye_right, axis=0)
        eye_avg      = (eye_left + eye_right) * 0.5
        eye_to_eye   = eye_right - eye_left
        mouth_left   = lm_mouth_outer[0]
        mouth_right  = lm_mouth_outer[6]
        mouth_avg    = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        x *= x_scale
        y = np.flipud(x) * [-y_scale, y_scale]
        c = eye_avg + eye_to_mouth * em_scale
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        # # Load in-the-wild image.
        # if not os.path.isfile(src_file):
        #     print('\nCannot find source image. Please run "--wilds" before "--align".')
        #     return
        # img = PIL.Image.open(src_file).convert('RGBA').convert('RGB')
        img = src_file
        # Shrink.
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, PIL.Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Pad.
        pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
            blur = qsize * 0.02
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
            img = np.uint8(np.clip(np.rint(img), 0, 255))
            if alpha:
                mask = 1-np.clip(3.0 * mask, 0.0, 1.0)
                mask = np.uint8(np.clip(np.rint(mask*255), 0, 255))
                img = np.concatenate((img, mask), axis=2)
                img = PIL.Image.fromarray(img, 'RGBA')
            else:
                img = PIL.Image.fromarray(img, 'RGB')
            quad += pad[:2]

        # Transform.
        img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
        if output_size < transform_size:
            img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

        # Save aligned image.
        # img.save(dst_file, 'PNG')
        alli_img_buffer = BytesIO()
        img.save(alli_img_buffer, format='JPEG')
        img_alli_binary = Binary(alli_img_buffer.getvalue())

        # Create a dictionary object to hold the metadata and the binary data of the image
        alli_image_metadata = {'filename': img_name +'_alligned.jpg', 'format': 'JPEG'}
        alli_image_data = {'metadata': alli_image_metadata, 'image': img_alli_binary}
        # Insert the image data into the MongoDB collection
        dst_file.insert_one(alli_image_data)
        print('Aligned image uploaded successfully for :', img)


import dlib
import os
path = os.getcwd()
# print(path)

class LandmarksDetector:
    def __init__(self, predictor_model_path='app/src/codeX/utils/shape_predictor_68_face_landmarks.dat'):
        """
        :param predictor_model_path: path to shape_predictor_68_face_landmarks.dat file
        """
        self.detector = dlib.get_frontal_face_detector() # cnn_face_detection_model_v1 also can be used
        self.shape_predictor = dlib.shape_predictor(predictor_model_path)

    def get_landmarks(self, image):
        # img = dlib.load_rgb_image(image)
        img = image
        dets = self.detector(img, 1)
        for detection in dets:
            try:
                face_landmarks = [(item.x, item.y) for item in self.shape_predictor(img, detection).parts()]
                yield face_landmarks
            except:
                print("Exception in get_landmarks()!")
# if __name__ == "__main__":
"""
Extracts and aligns all faces from images using DLib and a function from original FFHQ dataset preparation step
python align_images.py /raw_images /aligned_images
"""
parser = argparse.ArgumentParser(description='Align faces from input images', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('raw_dir', default = r'code\utils\images', help='Directory with raw images for face alignment')
# parser.add_argument('aligned_dir', default = r'code\utils\images\aligned_images',help='Directory for storing aligned images')
parser.add_argument('--output_size', default=512, help='The dimension of images for input to the model', type=int)
parser.add_argument('--x_scale', default=1, help='Scaling factor for x dimension', type=float)
parser.add_argument('--y_scale', default=1, help='Scaling factor for y dimension', type=float)
parser.add_argument('--em_scale', default=0.1, help='Scaling factor for eye-mouth distance', type=float)
parser.add_argument('--use_alpha', default=False, help='Add an alpha channel for masking', type=bool)

args, other_args = parser.parse_known_args()
# print("align path: %s"%os.getcwd())

# RAW_IMAGES_DIR = r'code\utils\images\raw' #args.raw_dir
# ALIGNED_IMAGES_DIR = r'code\utils\images\aligned_images' #args.aligned_dir
# RAW_IMAGES_DIR = RAW_IMAGES_DIR
# print(RAW_IMAGES_DIR,ALIGNED_IMAGES_DIR)
# ALIGNED_IMAGES_DIR = r'images\aligned_images' #args.aligned_dir
# RAW_IMAGES_DIR = r'images' #args.raw_dir
# ALIGNED_IMAGES_DIR = r'images\aligned_images' #args.aligned_dir
landmarks_detector = LandmarksDetector()
collection = db["src"]
for img_name in ('my_image.jpg','target.jpg'):
    print('Aligning %s ...' % img_name)
    # isFile = os.path.isfile(RAW_IMAGES_DIR+'/'+img_name)
    # if isFile:
    if collection.find_one({'metadata.filename': img_name}):    
        try:
            # raw_img_path = os.path.join(RAW_IMAGES_DIR, img_name)
            # fn = face_img_name = '%s_%02d.png' % (os.path.splitext(img_name)[0], 1)
            # print(raw_img_path,face_img_name)
            # if os.path.isfile(fn):
            #     continue
                    # Extract the binary data of the image from the image data
            image_data = collection.find_one({'metadata.filename': img_name})['image']
            # Load binary image data as PIL image
            pil_image = Image.open(BytesIO(image_data))
            print('Image shape:', pil_image.size)
            rgb_image = pil_image.convert('RGB')
            np_image = np.array(rgb_image)

            print('Getting landmarks...')
            for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(np_image), start=1):
                try:
                    print('Starting face alignment...')
                    # face_img_name = '%s_%02d.png' % (os.path.splitext(img_name)[0], i)
                    aligned_face_path =  db["alligned"]  #os.path.join(ALIGNED_IMAGES_DIR, face_img_name)
                    image_align(img_name,rgb_image, aligned_face_path, face_landmarks, output_size=args.output_size, x_scale=args.x_scale, y_scale=args.y_scale, em_scale=args.em_scale, alpha=args.use_alpha)
                    # print('Wrote result %s' % aligned_face_path)
                except:
                    print("Exception in face alignment!")
        except:
            print("Exception in landmark detection!")
