import os
import os.path
import cv2
import base64
import json
import numpy as np
import dlib
import face_recognition_models
from PIL import Image
from pprint import pprint
from os.path import basename
from datetime import datetime
from argparse import ArgumentParser
from base64 import b64encode
from json import dumps


face_detector = dlib.get_frontal_face_detector()

predictor_68_point_model = face_recognition_models.pose_predictor_model_location()
pose_predictor_68_point = dlib.shape_predictor(predictor_68_point_model)

predictor_5_point_model = face_recognition_models.pose_predictor_five_point_model_location()
pose_predictor_5_point = dlib.shape_predictor(predictor_5_point_model)

cnn_face_detection_model = face_recognition_models.cnn_face_detector_model_location()
cnn_face_detector = dlib.cnn_face_detection_model_v1(cnn_face_detection_model)

face_recognition_model = face_recognition_models.face_recognition_model_location()
face_encoder = dlib.face_recognition_model_v1(face_recognition_model)


class FaceDetector(object):
    def __init__(self, video_reference, image_folder_path, attendance_folder_path, new_images_path=None):
        self.map_name_encode = {}
        self.list_images_faces = []
        self.folder_path = image_folder_path
        self.attendance_folder_path = attendance_folder_path
        self.folder_new_images = new_images_path # FOR TECNOFACENS
        
        self.video_capture = cv2.VideoCapture(video_reference)
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        

        # Initialize some variables
        self.face_locations_list = []
        self.face_encodings_list = []
        self.face_landmarks_list = []
        self.face_names = []
        self.process_this_frame = True

    def find_students_images(self):
        if os.path.isfile("all_photos_processed.json"):
            print("Arquivo de modelo encontrado")
            self.load_map_from_json()
        else:
            print("Arquivo de modelo nao encontrado")
            path = self.folder_path
            valid_images = [".jpg",".gif",".png"]
            i = 1
            for f in os.listdir(path):
                ext = os.path.splitext(f)[1]
                if ext.lower() not in valid_images:
                    continue
                current_image = Image.open(os.path.join(path,f))

                if current_image:
                    name = os.path.splitext(basename(os.path.join(path,f)))[0]

                    current_image = current_image.convert('RGB')
                    face = self.face_encodings(np.array(current_image))
                    print("%s - %s" % (i, len(os.listdir(path)) ) )
                    if face:
                        k_face = face[0]
                        if k_face.any():
                            self.map_name_encode[name] = k_face
                        else:
                            print(name)
                    else:
                        print(name)
                    i += 1
                else:
                    print(os.path.join(path,f))

            self.save_map_to_json()

    def find_new_students_images(self):
        path = self.folder_new_images
        if path:
            valid_images = [".jpg",".gif",".png"]
            has_change = False
            for f in os.listdir(self.folder_new_images):
                ext = os.path.splitext(f)[1]
                if ext.lower() not in valid_images:
                    continue
                current_image = Image.open(os.path.join(path,f))
                if current_image:
                    name = os.path.splitext(basename(os.path.join(path,f)))[0]

                    if name not in self.map_name_encode.keys():
                        print("Appending new students image = %s" % name)
                        has_change = True
                        current_image = current_image.convert('RGB')
                        face = self.face_encodings(np.array(current_image))
                        if face:
                            k_face = face[0]
                            if k_face.any():
                                self.map_name_encode[name] = k_face
            if has_change:
                self.save_map_to_json()

    def load_map_from_json(self):
        with open("all_photos_processed.json", "r") as f:
            self.map_name_encode = json.loads(f.read())
            for k in self.map_name_encode.keys():
                self.map_name_encode[k] = np.array(self.map_name_encode[k])

    def save_map_to_json(self):
        map_copy = self.map_name_encode.copy()
        for k in map_copy.keys():
            map_copy[k] = map_copy[k].tolist()

        output_json = json.dumps(map_copy)
        with open("all_photos_processed.json","w") as output:
            output.write(output_json)

    def process(self):
        while True:
            ret, frame = self.video_capture.read()

            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            rgb_small_frame = small_frame[:, :, ::-1]

            if self.process_this_frame:
                self.face_locations_list = self.face_locations(rgb_small_frame)

                self.face_encodings_list = self.face_encodings(rgb_small_frame, self.face_locations_list)

                self.face_landmarks_list = self.face_landmarks(rgb_small_frame, self.face_locations_list)

                self.face_names = []
                for face_encoding in self.face_encodings_list:
                    matches = self.compare_faces(list(self.map_name_encode.values()), face_encoding, tolerance=0.5)
                    name = "DESCONHECIDO"
                    
                    if True in matches:
                        first_match_index = matches.index(True)
                        name = list(self.map_name_encode.keys())[first_match_index]

                        self.save_attend(name)

                    self.face_names.append(name)

            self.process_this_frame = not self.process_this_frame

            for i in range(0,len(self.face_landmarks_list)):
                if len(self.face_landmarks_list[i]) > 0:
                    for j in range(1,68):
                        cv2.circle(frame, (self.face_landmarks_list[i][j][0]*4, self.face_landmarks_list[i][j][1]*4), 1, (255, 255, 255), thickness=-1)
            
            # Display the results
            for (top, right, bottom, left), name in zip(self.face_locations_list, self.face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, bottom + 80), (right, bottom + 45), (0, 0, 0), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                red_value = 255
                green_value = 255
                blue_value = 255

                if name == "DESCONHECIDO":
                    green_value = 0
                    blue_value = 0

                cv2.putText(frame, name.upper(), (left + 10, bottom + 73 ), font, 1.0, (blue_value, green_value, red_value), 1)

            cv2.imshow('FaceDetector 1.0', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):          
                self.video_capture.release()
                cv2.destroyAllWindows()

    def save_attend(self, name):
        txt_log_name = str(self.folder_path).split("/")[-1] + "-" + str(self.current_date) + ".txt"
        full_path = self.attendance_folder_path + "/" + txt_log_name
        found = False
        try:
            with open(full_path) as file:
                current_data = file.read()
                if name in current_data:
                    found = True
        except Exception as identifier:
            pass
        
        with open(full_path,"a+") as file:
            if not found:
                file.write(name + "\n")

    def face_locations(self, img, number_of_times_to_upsample=1, model="hog"):
        if model == "cnn":
            return [self._trim_css_to_bounds(self._rect_to_css(face.rect), img.shape) for face in self._raw_face_locations(img, number_of_times_to_upsample, "cnn")]
        else:
            return [self._trim_css_to_bounds(self._rect_to_css(face), img.shape) for face in self._raw_face_locations(img, number_of_times_to_upsample, model)]

    def face_landmarks(self, face_image, face_locations=None):
        landmarks = self._raw_face_landmarks(face_image, face_locations)
        landmarks_as_tuples = [[(p.x, p.y) for p in landmark.parts()] for landmark in landmarks]
        return landmarks_as_tuples

    def _trim_css_to_bounds(self, css, image_shape):
        return max(css[0], 0), min(css[1], image_shape[1]), min(css[2], image_shape[0]), max(css[3], 0)

    def face_distance(self, face_encodings, face_to_compare):
        if len(face_encodings) == 0:
            return np.empty((0))

        return np.linalg.norm(face_encodings - face_to_compare, axis=1)

    def compare_faces(self, known_face_encodings, face_encoding_to_check, tolerance=0.6):
        return list(self.face_distance(known_face_encodings, face_encoding_to_check) <= tolerance)

    def face_encodings(self, face_image, known_face_locations=None, num_jitters=1):
        raw_landmarks = self._raw_face_landmarks(face_image, known_face_locations, model="large")
        return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for raw_landmark_set in raw_landmarks]

    def _raw_face_landmarks(self, face_image, face_locations=None, model="large"):
        if face_locations is None:
            face_locations = self._raw_face_locations(face_image)
        else:
            face_locations = [self._css_to_rect(face_location) for face_location in face_locations]

        pose_predictor = pose_predictor_68_point

        if model == "small":
            pose_predictor = pose_predictor_5_point

        return [pose_predictor(face_image, face_location) for face_location in face_locations]

    def _raw_face_locations(self, img, number_of_times_to_upsample=1, model="hog"):
        if model == "cnn":
            return cnn_face_detector(img, number_of_times_to_upsample)
        else:
            return face_detector(img, number_of_times_to_upsample)

    def _css_to_rect(self, css):
        return dlib.rectangle(css[3], css[0], css[1], css[2])

    def _rect_to_css(self, rect):
        return rect.top(), rect.right(), rect.bottom(), rect.left()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-i", "--images", dest="imagefolder",default="/Users/marceloaquino/Desktop/CursoFACENS/projeto-dnn/faces",
                        help="Folder with images", metavar="IMG")

    parser.add_argument("-o", "--output", dest="output",default="/Users/marceloaquino/Desktop/CursoFACENS/projeto-dnn/output",
                        help="Folder with output attendance text files", metavar="OUT")
    
    parser.add_argument("-v", "--video",dest="video", default=0,
                        help="Input value of the video reference")

    args = parser.parse_args()
    pprint(args)

    detector = FaceDetector(args.video, args.imagefolder, args.output)

    detector.find_students_images()
    detector.process()
