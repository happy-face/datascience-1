import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import cv2
import csv

from imutils import face_utils
import imutils
import dlib

from scipy.signal import convolve2d
from scipy.spatial import distance as dist



# construct the argument parser and parse the arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ip", "--im-path", required=True, help="Path to folder with images")
    parser.add_argument("-o", "--output", required=True, help="Output folder")
    parser.add_argument("--force", action="store_true", help="Overwrites output folder if it already exists")
    return parser.parse_args()


#convert from BGR to RGB for plotting purposes
def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#input image MUST BE GRAYSCALE
def estimate_general_quality(img):


    #blur detection (higher number = less blurry)
    def estimate_blur(img):
        # compute the Laplacian of the image and then return the focus
        # measure, which is simply the variance of the Laplacian
        return cv2.Laplacian(img, cv2.CV_64F).var()

    #noise estimation (lower number == less noise)
    def estimate_noise(img):
        H, W = img.shape
        M = [[1, -2, 1],
             [-2, 4, -2],
             [1, -2, 1]]

        sigma = np.sum(np.sum(np.absolute(convolve2d(img, M))))
        sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2))
        return sigma

    def estimate_brightness(img):
        histogram = cv2.calcHist([img], [0], None, [256], [0, 256])
        pixels = sum(histogram)
        brightness = scale = len(histogram)

        for index in range(0, scale):
            ratio = histogram[index] / pixels
            brightness += ratio * (-scale + index)

        return 1 if brightness == 255 else brightness / scale

    blur_score = estimate_blur(img)
    print("Blurrines of the image: ", blur_score)
    noise_score = estimate_noise(img)
    print("Noise in the image: ", noise_score)
    bright_score = estimate_brightness(img)
    print("Brightness of the image: ", bright_score)
    return blur_score, noise_score, bright_score

#must have gray image as input
def face_detector(img):

    # initialize dlib's face detector (HOG-based)
    detector = dlib.get_frontal_face_detector()

    # detect faces in the grayscale image
    faces = detector(img, 1)
    print('Faces found: ', len(faces))

    if len(faces) == 0:
        print('No faces detected')

    return faces

#gray image as input!!!
def face_landmarks_detector(img, faces, im_name, out_path):

    # create the facial landmark predictor
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    face_landmarks = []

    for (i,rect) in enumerate(faces):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(img, rect)
        shape = face_utils.shape_to_np(shape)

        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # show the face number
        cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

        face_landmarks.append(shape)

    # show the output image with the face detections
    cv2.imshow("Output", image)
    cv2.waitKey(0)
    output_path = os.path.join(out_path, im_name + '_face_landmarks.png')
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cv2.imwrite(os.path.join(out_path, im_name + '_face_landmarks.png'),image)

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
        for subdir in subdirs:
            get_path_recursive(os.path.join(root, subdir), file_extensions, paths)


if __name__ == "__main__":
    args = parse_args()

    #create output folder
    if os.path.exists(args.output) and not args.force:
        print("Output folder %s already exists! Use --force to override this check." % (args.output))
        exit()
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    #import image list
    file_extensions = ['.JPG', '.jpg', '.png', '.PNG']
    im_paths = []
    get_path_recursive(args.im_path, file_extensions, im_paths)

    df = pd.DataFrame(im_paths, columns=['file_name'])


    table = []
    for im_path in im_paths:

        image = cv2.imread(im_path)
        assert image is not None

        #convert to gray
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blur, noise, brightness = estimate_general_quality(gray_img)

        #face region detection
        faces = face_detector(gray_img)
        number_of_faces = len(faces)

        #EXTRACT FACE FEATURES

        faces_blur_all = []
        faces_noise_all = []
        faces_brightness_all = []

        for face in faces:
            # extract face ROI
            face_roi = get_face_roi(gray_img, face)
            face_blur, face_noise, face_brightness = estimate_general_quality(face_roi)

            faces_blur_all.append(face_blur)
            faces_noise_all.append(face_noise)
            faces_brightness_all.append(face_brightness)

        landmarks = face_landmarks_detector(gray_img, faces, im_path, args.output)

        closed_eyes = closed_eyes_detector(landmarks)

        im_set = os.path.split(os.path.dirname(im_path))[-1]
        im_path_csv = os.path.join(im_set, os.path.basename(im_path))
        table_entry = [im_path, im_path_csv, blur, noise, brightness, faces, number_of_faces, faces_blur_all, faces_noise_all, faces_brightness_all, closed_eyes]
        table.append(table_entry)

    df_output = pd.DataFrame(table, columns = ['im_file', 'set', 'blur', 'noise', 'brightness', 'faces', 'number_of_faces', 'faces_blur_all', 'faces_noise_all',
                   'faces_brightness_all', 'closed_eyes'])

    df_output.to_csv(os.path.join(args.output, 'results_processed.csv'), sep='\t')
