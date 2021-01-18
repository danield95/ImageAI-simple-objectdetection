#Import ObjectDetection class from the ImageAI library.
from imageai.Detection import ObjectDetection
import glob
import cv2

#Create an instance of the class ObjectDetection.
detector = ObjectDetection()

#Specify the path from our input image, output image, and model.
model_path = "/Users/dumitrescucristian/Desktop/OD/models/yolo-tiny.h5"
input_path = "/Users/dumitrescucristian/Desktop/OD/input/img1.jpg"
output_path = "/Users/dumitrescucristian/Desktop/OD/output/newimage.jpg"

#Use the setModelTypeAsTinyYOLOv3() function to load our model.
#Call the function setModelPath(). This function accepts a string which contains
#the path to the pre-trained model.
#Call the function loadModel() from the detector instance. It loads the model
#from the path specified above using the setModelPath() class method.
detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath(model_path)
detector.loadModel()

#detect object in picture
detections = detector.detectObjectsFromImage(input_image=input_path, output_image_path=output_path)

for eachItem in detections:
    print(eachItem["name"] , " : ", eachItem["percentage_probability"])

# #detect object in picture and extract each object from the input image and save it independently.
# detections, objects_path= detector.detectObjectsFromImage(input_image=input_path, output_image_path=output_path, extract_detected_objects=True)
#
# for eachObject, eachObjectPath in zip(detections, objects_path):
#     print(eachObject["name"] , " : " , eachObject["percentage_probability"], " : ", eachObject["box_points"] )
#     print("Object's image saved in " + eachObjectPath)
#     print("--------------------------------")
