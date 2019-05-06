# -*- coding:utf-8 -*-
'''
 'Requirement'
 끝번호가 짝수인 학생(얼굴인식) 본인의 얼굴을 셀카로 10매 촬영하시오.
 이때 조명 배경, 포즈 등을 다르게 하시오.
 그리고 오픈소스를 이용하여 얼굴을 검출하고,
 특징을 찾아내고 이들을 얼굴영상 위에 겹쳐 표시하시오.

 그 이후에 인식을 수행하여 동일한 얼굴임을 입증해 보이시오.
'''
# HYUNJAE LEE 20133096

from PIL import Image, ImageDraw
import face_recognition
import os
import cv2
import os
import glob
from matplotlib import pyplot as plt

# Load images
path = glob.glob("*.jpg")
cv_img = []
for img in path:
	print(img)
	n = cv2.imread(img, cv2.IMREAD_COLOR)
	#cv2.imshow('temp',n)
	#cv2.waitKey(0)
	#cv2.destoyAllWindows()
	'''
	b,g,r = cv2.split(n)
	img2 = cv2.merge([r,g,b])
	plt.imshow(img2)
	plt.xticks([])
	plt.yticks([])
	plt.show()
	'''
	cv_img.append(n)

# 1-step [Face Detection]
print("===========================================")
print("========= 1 STEP : Face Detection =========")
print("===========================================")

# Lists
Detected_list = []
Failed_list = []

for eachImage in path:
	image = face_recognition.load_image_file(eachImage)
	face_locations = face_recognition.face_locations(image)
	#face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")

	# If no face is detected
	if not face_locations:
		print("")
		print(" ------------------------------- ")
		print("# {} image has No face detected".format(eachImage))
		print(" ------------------------------- ")
		print("")
		Failed_list.append(eachImage)
		continue 
	print("")	
	print(" ------------------------------- ")
	print("# {} image has face detected".format(eachImage))
	Detected_list.append(eachImage)

	# If multiple faces are detected
	iteration = 1

	for face_location in face_locations:
		top, right, bottom, left = face_location
		print("A face is located at pixel location Top: {}, Left: {}, Right: {}, Bottom: {}".format(top, left, right, bottom))
		rename = "detected_#"+str(iteration)+"_face:"+eachImage
		face_image = image[top:bottom, left:right]
		pill_image = Image.fromarray(face_image)
		pill_image.save(os.path.curdir + "/detected/" +rename)
		print(" ------------------------------- ")
		print("")
		iteration += 1

print(" Detected Lists : ")
print(Detected_list)
print(" Failed Lists : ")
print(Failed_list)

# 2-step [Face Feature Extraction]
print("====================================================")
print("========= 2 STEP : Face Feature Extraction =========")
print("====================================================")
for eachImage in path:
    image = face_recognition.load_image_file(eachImage)
    face_landmarks_list = face_recognition.face_landmarks(image)
    print("")
    print(" ------------------------------- ")
    print("I found {} face(s) in {} photograph.".format(len(face_landmarks_list), eachImage))

    # Create a PIL imagedraw object so we can draw on the picture
    pil_image = Image.fromarray(image)

    for face_landmarks in face_landmarks_list:
	    d = ImageDraw.Draw(pil_image, 'RGBA')

    	# Make the eyebrows into a nightmare
	    d.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 128))
	    d.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 128))
	    d.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150), width=5)
	    d.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150), width=5)

	    # Gloss the lips
	    d.polygon(face_landmarks['top_lip'], fill=(150, 0, 0, 128))
	    d.polygon(face_landmarks['bottom_lip'], fill=(150, 0, 0, 128))
	    d.line(face_landmarks['top_lip'], fill=(150, 0, 0, 64), width=8)
	    d.line(face_landmarks['bottom_lip'], fill=(150, 0, 0, 64), width=8)

	    # Sparkle the eyes
	    d.polygon(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
	    d.polygon(face_landmarks['right_eye'], fill=(255, 255, 255, 30))

	    # Apply some eyeliner
	    d.line(face_landmarks['left_eye'] +
           [face_landmarks['left_eye'][0]], fill=(0, 0, 0, 110), width=6)
	    d.line(face_landmarks['right_eye'] +
           [face_landmarks['right_eye'][0]], fill=(0, 0, 0, 110), width=6)

	    # Print the loaction of each facial feature in this image
	    for facial_feature in face_landmarks.keys():
	        print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))
		
		# Trace out each facial feature in the image with a line!
	    for facial_feature in face_landmarks.keys():
	        d.line(face_landmarks[facial_feature], width = 5)
		
	    rename = "featured_" + eachImage
	    pil_image.save(os.path.curdir + "/featured/" + rename)
	    print(" ------------------------------- ")
	    print("")

    
