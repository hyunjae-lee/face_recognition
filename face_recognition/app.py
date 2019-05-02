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

# Load images
img_dir = ""  # Enter Directory of all images
data_path = os.path.join(img_dir, '*g')
files = glob.glob(data_path)
data = []  # list of images
for f1 in files:
    img = cv2.imread(f1)
    data.append(img)

# 1-step [Face Detection]

for eachImage in data:
    image = face_recognition.load_image_file(eachImage)
    face_locations = face_recognition.face_locations(image)
    print(face_locations)
    crop(image, face_locations, 'detection_cropped.jpg')

# 2-step [Face Feature Extraction]
for eachImage in data:
    image = face_recognition.load_image_file(eachImage)
    face_landmarks_list = face_recognition.face_landmarks(image)

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

    pil_image.show()

    print(face_landmarks_list)
    crop(image, face_landmarks_list, 'featured_cropped.jpg')

# 3-step [Face recognition]
known_image = face_recognition.load_image_file("hyunjae.jpg")
unknown_image = face_recognition.load_image_file("unknown.jpg")

hyunjae_encoding = face_recognition.face_encodings(known_image)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

results = face_recognition.compare_faces([hyunjae_encoding], unknown_encoding)


def crop(image_path, coords, saved_location):
    """
    @param image_path: The path to the image to edit
    @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
    @param saved_location: Path to save the cropped image
    """
    image_obj = Image.open(image_path)
    cropped_image = image_obj.crop(coords)
    cropped_image.save(saved_location)
    cropped_image.show()


if __name__ == '__main__':
    main()
