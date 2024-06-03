import cv2
from deepface import DeepFace

class ImageBlur():
    def __init__(self) -> None:
        pass
        

    def blur(self, img_path):
        image = cv2.imread(img_path)
        face_locations = DeepFace.extract_faces(img_path=img_path, detector_backend='mtcnn')
        for face_location in face_locations:
            x, y, w, h = face_location['facial_area']['x'], face_location['facial_area']['y'], face_location['facial_area']['w'], face_location['facial_area']['h'] 
            face_image = image[y:y+h, x:x+w]
            face_image = cv2.blur(face_image, (50, 50))
            image[y:y+h, x:x+w] = face_image

        return image