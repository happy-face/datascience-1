import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import cv2
import csv
import traceback

from imutils import face_utils
import imutils
import dlib

from scipy.signal import convolve2d
from scipy.spatial import distance as dist

from PIL import Image, ImageFilter, ImageOps


# construct the argument parser and parse the arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ip", "--im-path", required=True, help="Path to folder with images")
    parser.add_argument("-do", "--debug-output", required=False, help="Path to output folder with debug immages")
    parser.add_argument("-o", "--output", required=True, help="Output CSV file")
    parser.add_argument("-hfp", "--hog-face-predictor", action="store_true", help="Use hog (dlib) face predictor")
    parser.add_argument("-dfp", "--dnn-face-predictor", action="store_true", help="Use dnn (opencv) face predictor")
    parser.add_argument("--force", action="store_true", help="Overwrites output CSV and debug output folder if it already exists")
    return parser.parse_args()


# create the facial landmark predictor
python_script_dir = os.path.dirname(os.path.realpath(__file__))
face_features_predictor = dlib.shape_predictor(os.path.join(python_script_dir, 'shape_predictor_68_face_landmarks.dat'))

# initialize dlib's face detector (HOG-based)
face_detector = dlib.get_frontal_face_detector()

#initialize OpenCV's dnn face detector
dnn_prototxt = os.path.join(python_script_dir, 'deploy.prototxt.txt')
dnn_model = os.path.join(python_script_dir, 'res10_300x300_ssd_iter_140000.caffemodel')
dnn_face_detector = cv2.dnn.readNetFromCaffe(dnn_prototxt, dnn_model)

# convert from BGR to RGB for plotting purposes
def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# input image MUST BE GRAYSCALE
def estimate_general_quality(img):

    def estimate_sharpness(img):
        img = Image.fromarray(img) #convert to PIL image format
        img_blur = np.array(img.filter(ImageFilter.BLUR))
        diffs = np.asfarray(img) - np.asfarray(img_blur)
        return np.percentile(np.abs(diffs), 98)

    def estimate_noise(img):
        img = Image.fromarray(img)
        # like sharpness, but median filter then subtract instead of blur to reduce impact of edges
        img_blur = np.array(img.filter(ImageFilter.MedianFilter(3)))
        diffs = np.asfarray(img) - np.asfarray(img_blur)
        noise = np.percentile(np.abs(diffs), 98)
        return noise

    def estimate_motion_blur(img):
        img = Image.fromarray(img)
        # try several difference kernels at different angles, find the one that blurs the image the most
        kernels = [[0, 0, 0, 1, 0, -1, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0, 0, -1, 0],
                   [1, 0, 0, 0, 0, 0, 0, 0, -1],
                   [0, 0, 1, 0, 0, 0, -1, 0, 0]]
        min_sharpness = 1000000
        for ker in kernels:
            im_edges = img.filter(ImageFilter.Kernel((3, 3), ker, 1, 128))
            im_edges = np.abs(np.asfarray(im_edges) - 128)
            sharpness = np.percentile(im_edges, 98)
            if sharpness < min_sharpness:
                min_sharpness = sharpness
        return min_sharpness


    resized_img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)


    sharpness_score = estimate_sharpness(resized_img)
    noise_score = estimate_noise(resized_img)
    #print("Noise in the image: ", noise_score)
    motion_blur_score = estimate_motion_blur(resized_img)
    #print("Brightness of the image: ", bright_score)
    return sharpness_score, noise_score, motion_blur_score

#input has to be a colored image
def estimate_color_quality(img):

    def estimate_contrast(img):
        img = np.array(img, dtype='f') * 1. / 255
        img_hsv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)
        h, s, v = np.dsplit(img_hsv, 3)
        # return mean, stddev, and 80th percentile range of brightness as estimates of contrast
        return [np.mean(v), np.std(v), np.percentile(v, 90) - np.percentile(v, 10)]

    def estimate_saturation(img):
        img = np.array(img, dtype='f') * 1. / 255
        img_hsv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)
        h, s, v = np.dsplit(img_hsv, 3)
        return np.mean(s)

    resized_img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)

    contrast_score = estimate_contrast(resized_img)
    saturation_score = estimate_saturation(resized_img)
    return contrast_score, saturation_score

def estimate_composition_quality(img):
    # create the facial landmark predictor
    python_script_dir = os.path.dirname(os.path.realpath(__file__))
    predictor = dlib.shape_predictor(os.path.join(python_script_dir, 'shape_predictor_68_face_landmarks.dat'))
    def estimate_lines(img):
        img = Image.fromarray(img)  # convert to PIL image format
        # Run Hough transform to find lines in image, compute average angle and spread
        img_small = img.resize((int(img.size[0] / 2), int(img.size[1] / 2)))
        edges_bin = cv2.Canny(np.array(img), 100, 300)
        lines = cv2.HoughLinesP(edges_bin, 1, .02, 10, minLineLength=50, maxLineGap=15)
        # circular mean
        if lines is not None and len(lines) > 0 and len(lines[0]) > 0:
            sum_x = 0
            sum_y = 0
            for line in lines[0]:
                dx, dy = line[2] - line[0], line[3] - line[1]
                sum_x += dx
                sum_y += dy

            sum_x /= len(lines[0])
            sum_y /= len(lines[0])
            return [np.arctan2(sum_y, sum_x), np.sqrt(sum_x * sum_x + sum_y * sum_y)]  # average angle, concentration
        else:
            return [0, 0]

    def estimate_symmetry(img):
        img = Image.fromarray(img)  # convert to PIL image format
        # flip image h+v around center and thirds, measure differences
        img_arr = cv2.equalizeHist(np.array(img))  # normalize first

        horiz_diff = np.sqrt(np.mean(np.power(img_arr - img_arr[::-1, :], 2)))
        vert_diff = np.sqrt(np.mean(np.power(img_arr - img_arr[:, ::-1], 2)))

        # check thirds
        # some crops around the third lines
        img_left = img.crop((0, 0, 2 * img.size[0] / 3, img.size[1]))
        img_right = img.crop((img.size[0] / 3, 0, img.size[0], img.size[1]))
        img_top = img.crop((0, 0, img.size[0], 2 * img.size[1] / 3))
        img_bot = img.crop((0, img.size[1] / 3, img.size[0], img.size[1]))

        max_thirds_symm = 0
        for im in [img_left, img_right, img_top, img_bot]:
            im = cv2.equalizeHist(np.array(im))
            h_diff = np.sqrt(np.mean(np.power(im - im[::-1, :], 2)))
            v_diff = np.sqrt(np.mean(np.power(im - im[:, ::-1], 2)))
            max_thirds_symm = max(max_thirds_symm, h_diff, v_diff)

        return [max(horiz_diff, vert_diff), max_thirds_symm]

    resized_img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)

    lines = estimate_lines(resized_img)
    symmetry = estimate_symmetry(resized_img)
    return lines, symmetry


#must have gray image as input
def face_detection(img):
    # detect faces in the grayscale image
    faces = face_detector(img, 1)
    if len(faces) < 1:
        for angle in np.arange(90, 360, 90):
            print("rot angle: ", angle)
            rotated = imutils.rotate_bound(img, angle)
            faces = face_detector(rotated, 1)
            if len(faces) > 0:
                return faces, angle, rotated

    return faces, 0, img

#must have gray image as input
def face_detection_dnn(img):
    # detect faces in the grayscale image
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    dnn_face_detector.setInput(blob)
    detections = dnn_face_detector.forward()

    faces = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.60:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = dlib.rectangle(left=startX, top=startY, right=endX, bottom=endY)
            faces.append(face)

    if len(faces) < 1:
        for angle in np.arange(90, 360, 90):
            print("rot angle: ", angle)
            rotated = imutils.rotate_bound(img, angle)
            (h, w) = rotated.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(rotated, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            dnn_face_detector.setInput(blob)
            detections = dnn_face_detector.forward()

            faces = []
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.60:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    face = dlib.rectangle(left=startX, top=startY, right=endX, bottom=endY)
                    faces.append(face)
            if len(faces) > 0:
                return faces, angle, rotated

    return faces, 0, img


#gray image as input!!!
def face_landmarks_detection(img, faces, im_name):
    face_landmarks = []

    for (i,rect) in enumerate(faces):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = face_features_predictor(img, rect)
        shape = face_utils.shape_to_np(shape)

        face_landmarks.append(shape)

    return face_landmarks


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear


def eye_ear(face_shapes):
    eye_ear_list = []

    # grab the indexes of the facial landmarks for the left and
    # right eye
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    for (i,shape) in enumerate(face_shapes):
        # extract the left and right eye coordinates
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # compute the eye aspect ratio for both eyes
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        eye_ear_list.append((leftEAR, rightEAR))

    return eye_ear_list


def closed_eyes_detector(face_shapes):

    EYE_AR_THRESH = 0.1
    # grab the indexes of the facial landmarks for the left and
    # right eye
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    closed_eyes_in_image = []
    for (i,shape) in enumerate(face_shapes):
        # extract the left and right eye coordinates
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # compute the eye aspect ratio for both eyes
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        closed_eye = []

        if leftEAR > EYE_AR_THRESH:
            eye_closed = 0
        else:
            eye_closed = 1

        closed_eye.append(eye_closed)

        if rightEAR > EYE_AR_THRESH:
            eye_closed = 0
        else:
            eye_closed = 1

        closed_eye.append(eye_closed)
        closed_eyes_in_image.append(closed_eye)

    return closed_eyes_in_image


def get_face_roi(img, face):

    (x, y, w, h) = face_utils.rect_to_bb(face)
    face_roi = img[y:(y + h), x:(x + w)]
    return face_roi


def get_path_recursive(input_folder, file_extensions, paths):
    for root, subdirs, files in os.walk(input_folder):
        for file in files:
            if os.path.splitext(file)[1] in file_extensions:
                paths.append(os.path.join(root, file))
#        for subdir in subdirs:
#            print(os.path.join(root, subdir))
#            get_path_recursive(os.path.join(root, subdir), file_extensions, paths)



#gray image as input!!!
def debug_image(img, angle, faces, face_landmarks, debug_str, im_name, out_path):
    if angle != 0:
        img = imutils.rotate_bound(img, angle)

    img_height, img_width, colors = img.shape
    tickness = int(round(max(img_height, img_width) / 512));
    font_scale = 0.5 * tickness

    # plot rectangle and face number
    for (i,rect) in enumerate(faces):
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), tickness)
        cv2.putText(img, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), tickness)

    # plot face landmarks
    for one_face_landmarks in face_landmarks:
        for (x, y) in one_face_landmarks:
            cv2.circle(img, (x, y), 1, (0, 0, 255), tickness)

    # output debug string in top left corner
    debug_str_y = 100
    for one_debug_str in debug_str:
        cv2.putText(img, one_debug_str, (50, debug_str_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.75, (0, 255, 0), int(tickness * 0.75))
        debug_str_y += int(28 * font_scale)

    output_path = os.path.join(out_path, im_name + '_debug.png')
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if out_path:
        cv2.imwrite(os.path.join(out_path, im_name + '_debug.png'),img)


def debug_list_str(x):
    debug_str = "[ "
    for i in range(0, len(x)):
        debug_str += "%.1f" % x[i]
        if i != len(x) - 1:
            debug_str += ", "
    debug_str += " ]"
    return debug_str

def debug_list_list_str(x):
    debug_str = "[ "
    for i in range(0, len(x)):
        debug_str += "%s" % debug_list_str(x[i])
        if i != len(x) - 1:
            debug_str += ", "
    debug_str += " ]"
    return debug_str


def img_list_to_features(im_paths, debug_output=None):
    table = []
    for im_path in im_paths:
        try:
            print("Processing: %d/%d %s" % (len(table) + 1, len(im_paths), im_path))
            image = cv2.imread(im_path)
            assert image is not None

            #convert to gray
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            sharpness, noise, motion_blur = estimate_general_quality(gray_img)
            contrast, saturation = estimate_color_quality(image)
            lines, symmetry = estimate_composition_quality(gray_img)

            #face region detection
            if args.hog_face_predictor:
                faces, angle, face_img = face_detection(gray_img)

            if args.dnn_face_predictor:
                faces, angle, face_img = face_detection_dnn(image)

            number_of_faces = len(faces)

            #EXTRACT FACE FEATURES

            faces_sharpness_all = []
            faces_noise_all = []
            faces_motionb_all = []

            for face in faces:
                # extract face ROI
                face_roi = get_face_roi(face_img, face)
                face_sharp, face_noise, face_motion_blur = estimate_general_quality(face_roi)

                faces_sharpness_all.append(face_sharp)
                faces_noise_all.append(face_noise)
                faces_motionb_all.append(face_motion_blur)

            face_landmarks = face_landmarks_detection(face_img, faces, im_path)
            eye_ear_list = eye_ear(face_landmarks)

            im_set = os.path.split(os.path.dirname(im_path))[-1]
            im_path_csv = os.path.join(im_set, os.path.basename(im_path))
            table_entry = [im_path_csv, im_set, sharpness, noise, motion_blur, contrast, saturation, lines, symmetry, faces, number_of_faces, faces_sharpness_all, faces_noise_all, faces_motionb_all, eye_ear_list]
            table.append(table_entry)

            if debug_output is not None:
                debug_str = []
                debug_str.append("shr: %.1f" % sharpness)
                debug_str.append("nse: %.1f" % noise)
                debug_str.append("mblr: %.1f" % motion_blur)
                debug_str.append("cnt: %s" % debug_list_str(contrast))
                debug_str.append("str: %.1f" % saturation)
                debug_str.append("ln: %s" % debug_list_str(lines))
                debug_str.append("sym: %s" % debug_list_str(symmetry))
                debug_str.append("nfac: %.1f" % number_of_faces)
                debug_str.append("fshr: %s" % debug_list_str(faces_sharpness_all))
                debug_str.append("fmblr: %s" % debug_list_str(faces_motionb_all))
                debug_str.append("cleye: %s" % debug_list_list_str(eye_ear_list))
                debug_image(image, angle, faces, face_landmarks, debug_str, im_path_csv, debug_output)

        except KeyboardInterrupt:
            exit(-1)

        except:
            print("Failed to process: ", im_path)
            traceback.print_exc()
            print # -*- coding: utf-8 -*-

    df_output = pd.DataFrame(table, columns = ['im_path', 'set_name', 'sharpness', 'noise', 'motion_blur', 'contrast', 'saturation', 'lines', 'symmetry', 'faces', 'number_of_faces', 'faces_sharp_all', 'faces_noise_all',
                   'faces_motion_blur_all', 'eye_ear_list'])
    return df_output


if __name__ == "__main__":
    args = parse_args()

    # check if output file already exists
    if os.path.exists(args.output) and not args.force:
        print("Output folder %s already exists! Use --force to override this check." % (args.debug_output))
        exit()


    # ensure that output folder exists
    if args.debug_output:
        if os.path.exists(args.debug_output) and not args.force:
            print("Output folder %s already exists! Use --force to override this check." % (args.debug_output))
            exit()
        if not os.path.exists(args.debug_output):
            os.makedirs(args.debug_output)

    #import image list
    file_extensions = ['.JPG', '.jpg', '.png', '.PNG']
    im_paths = []
    get_path_recursive(args.im_path, file_extensions, im_paths)
    im_paths = sorted(im_paths)

    df_output = img_list_to_features(im_paths, args.debug_output)
    df_output.to_csv(os.path.join(args.output))
