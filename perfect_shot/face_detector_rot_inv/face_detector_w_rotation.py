# USAGE
# python rotate_simple.py --image images/saratoga.jpg

# import the necessary packages
import numpy as np
import argparse
import imutils
import cv2
import dlib
from imutils import face_utils
import math



# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the image file")
args = vars(ap.parse_args())


#must have gray image as input
def face_detector(img):

    # initialize dlib's face detector (HOG-based)
    detector = dlib.get_frontal_face_detector()

    # detect faces in the grayscale image
    faces = detector(img, 1)

    return faces

#gray image as input!!!
def face_landmarks_detector(img, faces):

    # create the facial landmark predictor
    predictor = dlib.shape_predictor('/home/ivana/gitrepo/datascience/perfect_shot/shape_predictor_68_face_landmarks.dat')

    img_height, img_width, img_depth = img.shape
    tickness = int(round(max(img_height, img_width) / 512));
    font_scale = 0.5 * tickness

    face_landmarks = []

    for (i,rect) in enumerate(faces):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(img, rect)
        shape = face_utils.shape_to_np(shape)

        (x, y, w, h) = face_utils.rect_to_bb(rect)

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), tickness)
        print(x)


        # show the face number
        cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), tickness)

        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), tickness)

        face_landmarks.append(shape)

#    # show the output image with the face detections
#    if img_width > 1024:
#        disp_img_width = 1024
#        disp_img_heigh = int(disp_img_width * (float(img_height) / img_width))
#    elif img_height > 768:
#        disp_img_heigh = 768
#        disp_img_width = int(disp_img_heigh * (float(img_width) / img_height))
#    else:
#        disp_img_width = img_width
#        disp_img_heigh = img_height
#    disp_img = cv2.resize(image, (disp_img_width, disp_img_heigh))
#    cv2.imshow("Output", disp_img)
#    cv2.waitKey(0)

    # output_path = os.path.join(out_path, im_name + '_face_landmarks.png')
    # output_dir = os.path.dirname(output_path)
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    #
    # if out_path:
    #     cv2.imwrite(os.path.join(out_path, im_name + '_face_landmarks.png'),image)

    return face_landmarks



img_max_width = 1700
img_max_height = 900

# load the image from disk
image = cv2.imread(args["image"])

height, width, depth = image.shape
if width > img_max_width:
	new_width = img_max_width;
	new_height = int(float(new_width) * (float(height) / width))
	width = new_width
	height = new_height

if height > img_max_height:
	new_height = img_max_height
	new_width = int(float(new_height) * (float(width) / float(height)))
	width = new_width
	height = new_height

image = cv2.resize(image, (width, height))
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# loop over the rotation angles again
# rotate bound ensures no part of the image is cut off
for angle in np.arange(0, 360, 30):
	rotated = imutils.rotate_bound(image, angle)
	cv2.imshow("Rotated (Correct)", rotated)
	faces = face_detector(rotated)
	face_landmarks = face_landmarks_detector(rotated, faces)
	number_of_faces = len(faces)
	print(number_of_faces)
	cv2.waitKey(0)