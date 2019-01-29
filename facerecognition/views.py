from django.shortcuts import render, redirect
import cv2
import os
from PIL import Image
import numpy as np 
from .settings import BASE_DIR
def index(request):
	return render(request,'index.html')

def create_dataset(request):

	enrollment = request.POST['enrollment']
	detect_face = cv2.CascadeClassifier(BASE+DIR+'/util/haarcascade_frontalface_default.xml')
	camera  = cv2.VideoCapture(0)
	
	image_id = enrollment

	sampleNum = 0

	while True:

		ret,img = camera.read()

		gray = 	cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

		faces = faceDetect.detectMultiScale(gray,1.3,5)

		for x,y,w,h in faces:
			
			samepleNum = sampleNum+1

			cv2.imwrite(BASE_DIR+'/util/dataset/user.'+str(enrollment)+'.'+str(sampleNum)+'.jpg',gray[y:y+h,x:x+w])

			cv2.rectangle(img,x,y,(x+w,y+h),(0,255,0),2)

			cv2.waitKey(250)

		cv2.imshow("Face",img)
		cv2.waitKey(1)

		if samepleNum>35:
			break
	camera.release()

	cv2.destroyAllWindows()

	return redirect('/')

def trainer(request):
	recognizer = cv2.face.LBPHFaceRecognizer_create()

	path = BASE_DIR+'util/dataset'

	def getImagesWithId(path):
		imagesPaths = [os.path.join(path,f) for f in os.listdir(path)]

		faces = []
		Ids = []

		for imagePath in imagePaths:

			faceImg = Image.open(imagePath).convert('L')

			faceNp = np.array(faceImg,'unit8')

			ID = int(os.path.split(imagePath)[-1].split('.')[1])

			faces.append(faceNp)

			Ids.append(ID)

			cv2.imshow("training",faceNp)
			cv2.waitKey(10)

		return np.array(Ids),np.array(faces)
	id,faces = getImagesWithID(path)

	recognizer.train(faces,ids)

	recognizer.save(BASE_DIR+'utils/recognizer/trainingData.yml')
	cv2.destroyAllWindows()

	return redirec('/')




