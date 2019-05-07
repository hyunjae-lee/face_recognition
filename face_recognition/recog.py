
# HYUNJAE LEE 20133096

from PIL import Image, ImageDraw
import face_recognition
import os
import cv2
import glob
from matplotlib import pyplot as plt

# Load images
path = glob.glob("*.jpg")

# 3-step [Face recognition]
unknown_list = []
fail_list = []

# 3.1 Unknown images encoding
for eachImage in path:
    unknown_image = face_recognition.load_image_file(eachImage)
    try:
        unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]
    except IndexError:
        fail_list.append(eachImage)
        continue

    unknown_list.append([unknown_face_encoding, eachImage])

print(" Detected Faces lists : ")
for detected_face in unknown_list:
    print(detected_face[1])

print(" Failed Faces lists : ")
for failed_face in fail_list:
    print(failed_face)

# 3.2 Known images encoding
for eachImage in path:
    hyunjae_image = face_recognition.load_image_file("hyunjae6.jpg")
    bada_image = face_recognition.load_image_file("bada.jpg")

    try:
        hyunjae_face_encoding = face_recognition.face_encodings(hyunjae_image)[0]
        bada_face_encoding = face_recognition.face_encodings(bada_image)[0]
    except IndexError:
        print(" Known lists error")
        print("I wasn't able to locate any faces in at least one of the images")
        quit()

    known_faces = [
        hyunjae_face_encoding,
        bada_face_encoding
    ]

# 3.3 Comparing between unknown images and known images with tolearnce = 0.5
print("==== Comparing unknown faces with known images ====")
print("")

for unknown_face in unknown_list:
    results = face_recognition.compare_faces(known_faces, unknown_face[0], tolerance=0.5)

    print(" === Check {} with known images".format(unknown_face[1]))
    print(" Is the unknown face a picture of HYUNJAE ? {}".format(results[0]))
    print(" Is the unknown face a picture of BADA ? {}".format(results[1]))
    print(" Is the unknown face a new person that we've never seen before? {}".format(not True in results))
    print("")
