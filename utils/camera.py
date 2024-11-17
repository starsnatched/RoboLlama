import cv2

class Camera:
    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)
        
    def read_frame(self):
        ret, frame = self.cap.read()
        return ret, frame
    
    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()