# File: face_recognition_app.py
import matplotlib.pyplot as plt
import cv2
import face_recognition
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label

# Load images of known faces and get their encodings
image1 = face_recognition.load_image_file("hydra2.jpg")
encoding1 = face_recognition.face_encodings(image1)[0]

image2 = face_recognition.load_image_file("obama.jpg")
encoding2 = face_recognition.face_encodings(image2)[0]

known_face_encodings = [encoding1, encoding2]
known_face_names = ["Hydra", "Obamna"]

class FaceRecognitionApp(App):
    def build(self):
        # Initialize main layout
        self.layout = BoxLayout(orientation='vertical')

        # Webcam feed widget
        self.img_widget = Image()
        self.layout.add_widget(self.img_widget)

        # Label for displaying recognized faces
        self.label = Label(text="Recognized faces will appear here")
        self.layout.add_widget(self.label)

        # Load known face encodings and names
        self.known_face_encodings = []
        self.known_face_names = []

        # Initialize webcam
        self.capture = cv2.VideoCapture(0)

        # Start a scheduled event for updating the webcam feed and face recognition
        Clock.schedule_interval(self.update, 1.0 / 5.0)  # 5 FPS

        return self.layout

    def update(self, dt):
        # Read a frame from the webcam
        ret, frame = self.capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=50)
        
        if not ret:
            return

        # Find all face locations and face encodings in the current frame
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame)
        
        face_names = []
        for face_encoding in face_encodings:
            # Check if the face matches any known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # If a match is found, use the first one
            if True in matches:
                match_index = matches.index(True)
                name = known_face_names[match_index]
                print("Match found: ", name)    

            face_names.append(name)

        # Display recognized face names in the label
        self.label.text = "Recognized faces: " + ", ".join(face_names)

        # Draw rectangles around faces and names on the frame
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Convert frame to texture
        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        self.img_widget.texture = texture

    def on_stop(self):
        # Release the webcam when the app closes
        self.capture.release()

if __name__ == '__main__':
    # Load known faces before starting the app
    app = FaceRecognitionApp()
    app.run()
