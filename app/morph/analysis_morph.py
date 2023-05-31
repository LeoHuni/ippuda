import cv2
import dlib
import numpy as np
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r'app/src/codeX/utils/shape_predictor_68_face_landmarks.dat')
from stream_main import *

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def shape_to_np(shape, dtype="float"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

def cos_sim(A, B):
	cos_sim = dot(A, B)/(norm(A)*norm(B))
	cos_sim = round(cos_sim,5)
	perc_dist = (math.pi - math.acos(cos_sim)) * 100 / math.pi
	return perc_dist

def vectorize(A):
	vec=[]
	for i in range(len(A)):
		if i == len(A)-1:
			vec.append(A[i]-A[0])
		else:
			vec.append(A[i+1]-A[i])
	return vec

def normalize(A):
	n = np.sum(A)
	return A/n

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
from numpy import dot
from numpy.linalg import norm
import math
# image1 = cv2.imread(r'E:\side_job\Korean_Consulting_project\sequence_0_.jpg')
# ana_image = cv2.imread(r'E:\side_job\Korean_Consulting_project\sequence_40_.jpg')

def analysis(image1,ana_image) :
	# image = cv2.imread(r'E:\side_job\Korean_Consulting_project\sequence_0_.jpg')
	image=  np.array(image1)
	# image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# detect faces in the grayscale image
	rects = detector(gray, 1)

	# image2 = cv2.imread(r'E:\side_job\Korean_Consulting_project\sequence_40_.jpg')
	image2 = np.array(ana_image)
	# image2 = imutils.resize(image2, width=500)
	gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
	# detect faces in the grayscale image
	rects2 = detector(gray2, 1)

	# loop over the face detections
	for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		# convert dlib's rectangle to a OpenCV-style bounding box
		# [i.e., (x, y, w, h)], then draw the face bounding box
		(x, y, w, h) = face_utils.rect_to_bb(rect)
		# cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
		# # show the face number
		# cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
		# 	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		for (x, y) in shape:
			cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
	# show the output image with the face detections + facial landmarks
	# cv2.imshow("Output", image)
	# cv2.waitKey(0)  
	# cv2.destroyAllWindows()     

	for (i2, rect2) in enumerate(rects2):
		
		shape2 = predictor(gray2, rect2)
		shape2 = face_utils.shape_to_np(shape2)

		# convert dlib's rectangle to a OpenCV-style bounding box
		# [i.e., (x, y, w, h)], then draw the face bounding box
		(x, y, w, h) = face_utils.rect_to_bb(rect2)
		# cv2.rectangle(image2, (x, y), (x + w, y + h), (0, 255, 0), 2)
		# # show the face number
		# cv2.putText(image2, "Face #{}".format(i2 + 1), (x - 10, y - 10),
		# 	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		# # loop over the (x, y)-coordinates for the facial landmarks
		# # and draw them on the image
		# for (x, y) in shape2:
		# 	cv2.circle(image2, (x, y), 1, (0, 0, 255), -1)
	# show the output image with the face detections + facial landmarks
	# cv2.imshow("Output", image2)
	# cv2.waitKey(0)  
	# cv2.destroyAllWindows()     

	################################################################
	Left_eyes = shape[36:42]
	Right_eyes = shape[42:48]

	nose_bridge_tmp = shape[33]
	nose_bridge = np.concatenate((shape[27:31],nose_bridge_tmp[None,:]),axis=0)

	nose = shape[31:36]
	face_outline = shape[0:17]
	Left_eyebrow = shape[22:27]
	Right_eyebrow = shape[17:22]
	u_mouth_tmp = shape[61:65]
	Upper_mouth = np.concatenate((shape[49:55],u_mouth_tmp[::-1]),axis=0)
	l_mouth_tmp = shape[66:68]
	lower_mouth = np.concatenate((shape[56:60], l_mouth_tmp[::-1]),axis =0)

	################################################################
	Left_eyes2 = shape2[36:42]
	Right_eyes2 = shape2[42:48]

	nose_bridge2_tmp = shape2[33]
	nose_bridge2 = np.concatenate((shape2[27:31],nose_bridge2_tmp[None,:]),axis=0)

	nose2 = shape2[31:36]
	face_outline2 = shape2[0:17]
	Left_eyebrow2 = shape2[22:27]
	Right_eyebrow2 = shape2[17:22]
	u_mouth2_tmp = shape2[61:65]
	Upper_mouth2 = np.concatenate((shape2[49:55],u_mouth2_tmp[::-1]),axis=0)
	l_mouth2_tmp = shape2[66:68]
	lower_mouth2 = np.concatenate((shape2[56:60], l_mouth2_tmp[::-1]),axis =0)
	################################################################

	L_eye_list = []
	R_eye_list = []
	nose_bridge_list = []
	nose_list = []
	face_outline_list = []
	L_eyebrow_list = []
	R_eyebrow_list = []
	u_mouth_list = []
	l_mouth_list = []

	u_L_eye_list = []
	u_R_eye_list = []
	u_nose_bridge_list = []
	u_nose_list = []
	u_face_outline_list = []
	u_L_eyebrow_list = []
	u_R_eyebrow_list = []
	u_u_mouth_list = []
	u_l_mouth_list = []

	################################################################
	for i in range(len(Left_eyes)):
		L_eye_list.append(cos_sim(vectorize(Left_eyes)[i], vectorize(Left_eyes2)[i]))
		u_L_eye_list.append(np.linalg.norm(vectorize(Left_eyes)[i] - vectorize(Left_eyes2)[i]))
		R_eye_list.append(cos_sim(vectorize(Right_eyes)[i], vectorize(Right_eyes2)[i]))
		u_R_eye_list.append(np.linalg.norm(vectorize(Left_eyes)[i] - vectorize(Left_eyes2)[i]))
		
	for i in range(len(nose_bridge)):
		nose_bridge_list.append(cos_sim(vectorize(nose_bridge)[i], vectorize(nose_bridge2)[i]))
		u_nose_bridge_list.append(np.linalg.norm(vectorize(nose_bridge)[i] - vectorize(nose_bridge2)[i]))

	for i in range(len(nose)):
		nose_list.append(cos_sim(vectorize(nose)[i], vectorize(nose2)[i]))
		u_nose_list.append(np.linalg.norm(vectorize(nose_bridge)[i] - vectorize(nose_bridge2)[i]))

	for i in range(len(face_outline)):
		face_outline_list.append(cos_sim(vectorize(face_outline)[i], vectorize(face_outline2)[i]))
		u_face_outline_list.append(np.linalg.norm(vectorize(face_outline)[i] - vectorize(face_outline2)[i]))

	for i in range(len(Left_eyebrow)):
		L_eyebrow_list.append(cos_sim(vectorize(Left_eyebrow)[i], vectorize(Left_eyebrow2)[i]))
		u_L_eyebrow_list.append(np.linalg.norm(vectorize(Left_eyebrow)[i] - vectorize(Left_eyebrow2)[i]))
		R_eyebrow_list.append(cos_sim(vectorize(Right_eyebrow)[i], vectorize(Right_eyebrow2)[i]))
		u_R_eyebrow_list.append(np.linalg.norm(vectorize(Right_eyebrow)[i] - vectorize(Right_eyebrow2)[i]))

	for i in range(len(Upper_mouth)):
		u_mouth_list.append(cos_sim(vectorize(Upper_mouth)[i], vectorize(Upper_mouth2)[i]))
		u_u_mouth_list.append(np.linalg.norm(vectorize(Upper_mouth)[i] - vectorize(Upper_mouth2)[i]))

	for i in range(len(lower_mouth)):
		l_mouth_list.append(cos_sim(vectorize(lower_mouth)[i], vectorize(lower_mouth2)[i]))
		u_l_mouth_list.append(np.linalg.norm(vectorize(lower_mouth)[i] - vectorize(lower_mouth2)[i]))
	################################################################

	L_eye_l2_mean = np.mean(u_L_eye_list)
	R_eye_l2_mean = np.mean(u_R_eye_list)
	nose_bridge_l2_mean = np.mean(u_nose_bridge_list)
	nose_l2_mean = np.mean(u_nose_list)
	face_outline_l2_mean = np.mean(u_face_outline_list)
	L_eyebrow_l2_mean = np.mean(u_L_eyebrow_list)
	R_eyebrow_l2_mean = np.mean(u_R_eyebrow_list)
	u_mouth_l2_mean = np.mean(u_u_mouth_list)
	l_mouth_l2_mean = np.mean(u_l_mouth_list)
	################################################################
	if L_eye_l2_mean > 1:
		L_eye_l2_mean = abs(L_eye_l2_mean-2)
	else:
		L_eye_l2_mean = 1

	if R_eye_l2_mean > 1:
		R_eye_l2_mean = abs(R_eye_l2_mean-2)
	else:
		R_eye_l2_mean = 1


	if round(nose_bridge_l2_mean) > 2:
		nose_bridge_l2_mean = abs(nose_bridge_l2_mean-3)
	elif round(nose_bridge_l2_mean) > 1:
		nose_bridge_l2_mean = abs(nose_bridge_l2_mean-2)
	else:
		nose_bridge_l2_mean = 1

	if round(nose_l2_mean) > 2:
		nose_l2_mean = abs(nose_l2_mean-3)
	elif round(nose_l2_mean) > 1:
		nose_l2_mean = abs(nose_l2_mean-2)
	else:
		nose_l2_mean = 1

	if round(face_outline_l2_mean) > 2:
		face_outline_l2_mean = abs(face_outline_l2_mean-3)
	elif round(face_outline_l2_mean) > 1 and face_outline_l2_mean < 1.9:
		face_outline_l2_mean = abs(face_outline_l2_mean-2)
	elif round(face_outline_l2_mean) > 1 and face_outline_l2_mean > 1.9:
		face_outline_l2_mean = abs(face_outline_l2_mean-1)
	else:
		face_outline_l2_mean = 1

	if L_eyebrow_l2_mean > 1:
		L_eyebrow_l2_mean = abs(L_eyebrow_l2_mean-2)
	else:
		L_eyebrow_l2_mean = 1

	if R_eyebrow_l2_mean > 1:
		R_eyebrow_l2_mean = abs(R_eyebrow_l2_mean-2)
	else:
		R_eyebrow_l2_mean = 1

	if u_mouth_l2_mean > 1:
		u_mouth_l2_mean = abs(u_mouth_l2_mean-2)
	else:
		u_mouth_l2_mean = 1	

	if l_mouth_l2_mean > 1:
		l_mouth_l2_mean = abs(l_mouth_l2_mean-2)
	else:
		l_mouth_l2_mean = 1	
	################################################################

	L_eye_listMean =  np.mean(L_eye_list) 
	R_eye_listMean =  np.mean(R_eye_list)
	nose_bridge_mean =  np.mean(nose_bridge_list)
	nose_mean = np.mean(nose_list)
	face_mean = np.mean(face_outline_list)
	L_eyebrow_mean = np.mean(L_eyebrow_list)
	R_eyebrow_mean = np.mean(R_eyebrow_list)
	U_mouth_mean = np.mean(u_mouth_list)
	L_mouth_mean = np.mean(l_mouth_list)

	################################################################

	from shapely.geometry import Polygon

	L_eye_poly = Polygon(Left_eyes)
	L_eye_poly2 = Polygon(Left_eyes2)
	L_eye_area = L_eye_poly.area
	L_eye_area2 = L_eye_poly2.area

	R_eye_poly = Polygon(Right_eyes)
	R_eye_poly2 = Polygon(Right_eyes2)
	R_eye_area = R_eye_poly.area
	R_eye_area2 = R_eye_poly2.area

	nose_bridge_poly = Polygon(nose_bridge)
	nose_bridge_poly2 = Polygon(nose_bridge2)
	nose_bridge_area = nose_bridge_poly.area
	nose_bridge_area2 = nose_bridge_poly2.area

	nose_poly = Polygon(nose)
	nose_poly2 = Polygon(nose2)
	nose_area = nose_poly.area
	nose_area2 = nose_poly2.area

	face_outline_poly = Polygon(face_outline)
	face_outline_poly2 = Polygon(face_outline2)
	face_outline_area = face_outline_poly.area
	face_outline_area2 = face_outline_poly2.area

	Left_eyebrow_poly = Polygon(Left_eyebrow)
	Left_eyebrow_poly2 = Polygon(Left_eyebrow2)
	Left_eyebrow_area = Left_eyebrow_poly.area
	Left_eyebrow_area2 = Left_eyebrow_poly2.area

	Right_eyebrow_poly = Polygon(Right_eyebrow)
	Right_eyebrow_poly2 = Polygon(Right_eyebrow2)
	Right_eyebrow_area = Right_eyebrow_poly.area
	Right_eyebrow_area2 = Right_eyebrow_poly2.area

	Upper_mouth_poly = Polygon(Upper_mouth)
	Upper_mouth_poly2 = Polygon(Upper_mouth2)
	Upper_mouth_area = Upper_mouth_poly.area
	Upper_mouth_area2 = Upper_mouth_poly2.area

	lower_mouth_poly = Polygon(lower_mouth)
	lower_mouth_poly2 = Polygon(lower_mouth2)
	lower_mouth_area = lower_mouth_poly.area
	lower_mouth_area2 = lower_mouth_poly2.area

	################################################################
	if L_eye_area < L_eye_area2:
		L_eye_coeff = L_eye_area/L_eye_area2
	else:
		L_eye_coeff = L_eye_area2/L_eye_area

	if R_eye_area < R_eye_area2:
		R_eye_coeff = R_eye_area/R_eye_area2
	else:
		R_eye_coeff = R_eye_area2/R_eye_area

	if (nose_bridge_area == 0) or (nose_bridge_area2 == 0):
		nose_bridge_coeff = 1
	else:
		if nose_bridge_area < nose_bridge_area2:
			nose_bridge_coeff = nose_bridge_area/nose_bridge_area2
		elif nose_bridge_area2 < nose_bridge_area:
			nose_bridge_coeff = nose_bridge_area2/nose_bridge_area
		else:
			nose_bridge_coeff = 1

	if nose_area < nose_area2:
		nose_coeff = nose_area/nose_area2
	else:
		nose_coeff = nose_area2/nose_area

	if face_outline_area < face_outline_area2:
		face_outline_coeff = face_outline_area/face_outline_area2
	else:
		face_outline_coeff = face_outline_area2/face_outline_area

	if Left_eyebrow_area < Left_eyebrow_area2:
		Left_eyebrow_coeff = Left_eyebrow_area/Left_eyebrow_area2
	else:
		Left_eyebrow_coeff = Left_eyebrow_area2/Left_eyebrow_area

	if Right_eyebrow_area < Right_eyebrow_area2:
		Right_eyebrow_coeff = Right_eyebrow_area/Right_eyebrow_area2
	else:
		Right_eyebrow_coeff = Right_eyebrow_area2/Right_eyebrow_area

	if Upper_mouth_area < Upper_mouth_area2:
		Upper_mouth_coeff = Upper_mouth_area/Upper_mouth_area2
	else:
		Upper_mouth_coeff = Upper_mouth_area2/Upper_mouth_area

	if lower_mouth_area < lower_mouth_area2:
		lower_mouth_coeff = lower_mouth_area/lower_mouth_area2
	else:
		lower_mouth_coeff = lower_mouth_area2/lower_mouth_area
	################################################################
	L_eye_listMean =  np.mean(L_eye_list) 
	R_eye_listMean =  np.mean(R_eye_list)
	nose_bridge_mean =  np.mean(nose_bridge_list)
	nose_mean = np.mean(nose_list)
	face_mean = np.mean(face_outline_list)
	L_eyebrow_mean = np.mean(L_eyebrow_list)
	R_eyebrow_mean = np.mean(R_eyebrow_list)
	U_mouth_mean = np.mean(u_mouth_list)
	L_mouth_mean = np.mean(l_mouth_list)

	L_eye_res = L_eye_listMean * L_eye_coeff * L_eye_l2_mean
	R_eye_res = R_eye_listMean * R_eye_coeff * R_eye_l2_mean
	nose_bridge_res = nose_bridge_mean * nose_bridge_coeff * nose_bridge_l2_mean
	nos_res = nose_mean * nose_coeff * nose_l2_mean
	face_res = face_mean * face_outline_coeff * face_outline_l2_mean
	L_eyebrow_res = L_eyebrow_mean * Left_eyebrow_coeff * L_eyebrow_l2_mean
	R_eyebrow_res = R_eyebrow_mean * Right_eyebrow_coeff * R_eyebrow_l2_mean
	U_mouth_res = U_mouth_mean * Upper_mouth_coeff * u_mouth_l2_mean
	L_mouth_res = L_mouth_mean * lower_mouth_coeff * l_mouth_l2_mean

	print('왼쪽 눈 유사도 : %3.2f' %L_eye_res)
	print('오른쪽 눈 유사도 : %3.2f' %R_eye_res)
	print('콧등 유사도 : %3.2f' %nose_bridge_res)
	print('코 유사도 : %3.2f' %nos_res)
	print('얼굴 아웃라인 유사도 : %3.2f' %face_res)
	print('왼쪽 눈썹 유사도 : %3.2f' %L_eyebrow_res)
	print('오른쪽 눈썹 유사도 : %3.2f' %R_eyebrow_res)
	print('윗 입술 유사도 : %3.2f' %U_mouth_res)
	print('아랫 입술 유사도 : %3.2f' %L_mouth_res)

	res_tot= [L_eye_res, R_eye_res, nose_bridge_res, nos_res, face_res, L_eyebrow_res, R_eyebrow_res, U_mouth_res, L_mouth_res]
	# import matplotlib.pyplot as plt
	# print(res_tot)
	# plt.imshow(image[:,:,::-1])
	# plt.show()
	# plt.imshow(image2[:,:,::-1])
	# plt.show()

	# print(distance)
	return res_tot#L_eye_res, R_eye_res, nose_bridge_res, nos_res, face_res, L_eyebrow_res, R_eyebrow_res, U_mouth_res, L_mouth_res

# print('start')
# res_tot = analysis(image1,ana_image)

