import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import cv2

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


    blur_score = estimate_blur(img)
    print("Blurrines of the image: ", blur_score)
    noise_score = estimate_noise(img)
    print("Noise in the image: ", noise_score)
    return blur_score, noise_score

#must have gray image as input
def face_detector(img):
    #using HAAR Cascade Classifier
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
    print('Faces found: ', len(faces))

    if len(faces) == 0:
        print('No faces detected')

    return faces

def eyes_detector(img):

    eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
    eyes = eye_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
    print('Eyes found: ', len(eyes))

    if len(eyes) == 0:
        print('No eyes detected')

    return eyes

def get_face_roi(img, im_name, faces):
    face_roi = []
    if faces.any():
        for (x, y, w, h) in faces:
            face_roi.append(img[y:(y + h), x:(x + w)])
    return face_roi

#requires grayscale image
def calculate_brightness(img):

    histogram = cv2.calcHist([img], [0], None, [256], [0,256])
    pixels = sum(histogram)
    brightness = scale = len(histogram)

    for index in range(0, scale):
        ratio = histogram[index] / pixels
        brightness += ratio * (-scale + index)

    return 1 if brightness == 255 else brightness / scale


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



#requires grayscale image
def detect_facial_landmarks(img):
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # detect faces in the grayscale image
    rects = detector(img, 1)

    EYE_AR_THRESH = 0.1
    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    closed_eyes_image = []
    for (i,rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(img, rect)
        shape = face_utils.shape_to_np(shape)

        # convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # show the face number
        cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
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

        closed_eyes_image.append(closed_eye)

    # show the output image with the face detections + facial landmarks
    cv2.imshow("Output", image)
    cv2.waitKey(0)

    return closed_eyes_image



if __name__ == "__main__":
    args = parse_args()

    #create output folder
    if os.path.exists(args.output) and not args.force:
        print("Output folder %s already exists! Use --force to override this check." % (args.output))
        exit()
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    #import image list
    im_files = os.listdir(args.im_path)
    df = pd.DataFrame(im_files, columns=['file_name'])


    blur_all = []
    noise_all = []
    faces_all = []
    eyes_all = []
    face_brightness_all = []
    closed_eyes_all = []

    for im_name in im_files:

        image = cv2.imread(os.path.join('images/', im_name))

        #convert to gray (HAAR requirement)
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


        blur, noise = estimate_general_quality(gray_img)

        blur_all.append(blur)
        noise_all.append(noise)

        #face region detection
        faces = face_detector(gray_img)
        faces_all.append(faces)
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        plt.figure()
        plt.imshow(convertToRGB(image))
        plt.savefig(os.path.join(args.output, im_name + '_faces_detected.png'))

        #eyes detection
        eyes = eyes_detector(gray_img)
        eyes_all.append(eyes)
        for (x, y, w, h) in eyes:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        plt.figure()
        plt.imshow(convertToRGB(image))
        plt.savefig(os.path.join(args.output, im_name + '_eyes_detected.png'))

        face_roi = get_face_roi(gray_img, im_name, faces)
        '''#save faces ROI
        for i, face in enumerate(face_roi):
            cv2.imwrite(os.path.join(args.output, im_name + '_face' + str(i) + '_eyes_detected.png'), face)'''

        face_brightness_image= []
        for f in face_roi:
            face_brightness = calculate_brightness(f)
            face_brightness_image.append(face_brightness)
        face_brightness_all.append(face_brightness_image)
        print("Face brightness: ", face_brightness_image)

        closed_eyes = detect_facial_landmarks(gray_img)
        closed_eyes_all.append(closed_eyes)

    df['blur'] = blur_all
    df['noise'] = noise_all
    df['faces'] = faces_all
    df['eyes'] = eyes_all
    df['face_brightness'] = face_brightness_all
    df['closed_eyes'] = closed_eyes_all

    df.to_csv(os.path.join(args.output, 'results_processed.csv'))
