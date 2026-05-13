# import os
# # This MUST be at the very top to stop the Protobuf version war
# os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# import cv2
# import numpy as np
# import joblib
# from tensorflow.keras.models import load_model
# from google.protobuf import descriptor as _pb_descriptor
# from google.protobuf import message_factory as _pb_message_factory
# from google.protobuf import symbol_database as _pb_symbol_database

# # Compatibility shims for MediaPipe 0.10.14 with protobuf 7.x.
# if not hasattr(_pb_descriptor.FieldDescriptor, 'label'):
#     _pb_descriptor.FieldDescriptor.label = property(lambda self: self._label)

# if not hasattr(_pb_message_factory.MessageFactory, 'GetPrototype'):
#     def _get_prototype(self, msg_descriptor):
#         return _pb_message_factory.GetMessageClass(msg_descriptor)

#     _pb_message_factory.MessageFactory.GetPrototype = _get_prototype

# if not hasattr(_pb_symbol_database.SymbolDatabase, 'GetPrototype'):
#     def _symbol_get_prototype(self, msg_descriptor):
#         return _pb_message_factory.GetMessageClass(msg_descriptor)

#     _pb_symbol_database.SymbolDatabase.GetPrototype = _symbol_get_prototype

# # Import MediaPipe solutions directly to bypass the version-check crash
# try:
#     import mediapipe.python.solutions.face_mesh as mp_face_mesh
# except ImportError:
#     from mediapipe.solutions import face_mesh as mp_face_mesh

# # 1. Load Assets
# try:
#     model = load_model('autism_model.h5')
#     scaler = joblib.load('scaler.pkl')
#     label_encoder = joblib.load('label_encoder.pkl')
#     print("✅ System Ready: Model and Scalers loaded.")
# except Exception as e:
#     print(f"❌ Error loading assets: {e}")
#     exit()

# # 2. Initialize Face Mesh
# face_mesh = mp_face_mesh.FaceMesh(
#     max_num_faces=1,
#     refine_landmarks=True,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# # Landmark Indices
# LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
# RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
# LEFT_PUPIL = [468]
# RIGHT_PUPIL = [473]

# # 3. Start Camera
# cap = cv2.VideoCapture(0)
# print("🚀 Camera Starting... Press ESC to quit.")

# while cap.isOpened():
#     success, image = cap.read()
#     if not success: continue

#     image = cv2.flip(image, 1)
#     rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(rgb_image)

#     if results.multi_face_landmarks:
#         for face_landmarks in results.multi_face_landmarks:
#             h, w, _ = image.shape

#             def get_coords(indices):
#                 x = np.mean([face_landmarks.landmark[i].x * w for i in indices])
#                 y = np.mean([face_landmarks.landmark[i].y * h for i in indices])
#                 z = np.mean([face_landmarks.landmark[i].z for i in indices]) * 100
#                 return x, y, z

#             # Feature Extraction
#             re_x, re_y, re_z = get_coords(RIGHT_EYE)
#             le_x, le_y, le_z = get_coords(LEFT_EYE)
#             rp_x, rp_y, _ = get_coords(RIGHT_PUPIL)
#             lp_x, lp_y, _ = get_coords(LEFT_PUPIL)

#             # Assemble the 14 features
#             features = [re_x, re_y, re_z, le_x, le_y, le_z, rp_x, rp_y, lp_x, lp_y, rp_x, rp_y, lp_x, lp_y]

#             # 4. Predict
#             input_data = np.array(features).reshape(1, -1)
#             scaled_data = scaler.transform(input_data)
#             prediction = model.predict(scaled_data, verbose=0)
            
#             class_idx = np.argmax(prediction)
#             label = label_encoder.inverse_transform([class_idx])[0]
#             confidence = np.max(prediction) * 100

#             # 5. Visual Output
#             color = (0, 255, 0) if label == 'Neurotypical' else (0, 0, 255)
#             cv2.putText(image, f"{label} ({confidence:.1f}%)", (30, 50), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
#             cv2.circle(image, (int(rp_x), int(rp_y)), 2, (255, 255, 255), -1)
#             cv2.circle(image, (int(lp_x), int(lp_y)), 2, (255, 255, 255), -1)

#     cv2.imshow('Eye Tracking System', image)
#     if cv2.waitKey(5) & 0xFF == 27: break

# cap.release()
# cv2.destroyAllWindows()

import os
# Force python implementation for protobuf to avoid version conflicts
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import cv2
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from google.protobuf import descriptor as _pb_descriptor
from google.protobuf import message_factory as _pb_message_factory
from google.protobuf import symbol_database as _pb_symbol_database

# --- Compatibility Shims ---
if not hasattr(_pb_descriptor.FieldDescriptor, 'label'):
    _pb_descriptor.FieldDescriptor.label = property(lambda self: self._label)
if not hasattr(_pb_message_factory.MessageFactory, 'GetPrototype'):
    def _get_prototype(self, msg_descriptor):
        return _pb_message_factory.GetMessageClass(msg_descriptor)
    _pb_message_factory.MessageFactory.GetPrototype = _get_prototype
if not hasattr(_pb_symbol_database.SymbolDatabase, 'GetPrototype'):
    def _symbol_get_prototype(self, msg_descriptor):
        return _pb_message_factory.GetMessageClass(msg_descriptor)
    _pb_symbol_database.SymbolDatabase.GetPrototype = _symbol_get_prototype

try:
    import mediapipe.python.solutions.face_mesh as mp_face_mesh
except ImportError:
    from mediapipe.solutions import face_mesh as mp_face_mesh

# --- 1. Load Assets ---
try:
    model = load_model('autism_model.h5')
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    print("✅ System Ready: Model and Scalers loaded.")
except Exception as e:
    print(f"❌ Error loading assets: {e}")
    exit()

# --- 2. Configuration ---
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

COLUMN_NAMES = [
    'Eye Position Right X [mm]', 'Eye Position Right Y [mm]', 'Eye Position Right Z [mm]',
    'Eye Position Left X [mm]', 'Eye Position Left Y [mm]', 'Eye Position Left Z [mm]',
    'Pupil Position Right X [px]', 'Pupil Position Right Y [px]',
    'Pupil Position Left X [px]', 'Pupil Position Left Y [px]',
    'Point of Regard Right X [px]', 'Point of Regard Right Y [px]',
    'Point of Regard Left X [px]', 'Point of Regard Left Y [px]'
]

LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_PUPIL = [468]
RIGHT_PUPIL = [473]

# --- 3. Execution ---
cap = cv2.VideoCapture(0)
print("🚀 Camera Starting... Prediction results will print below.")

while cap.isOpened():
    success, image = cap.read()
    if not success: continue

    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = image.shape

            def get_coords(indices):
                x = np.mean([face_landmarks.landmark[i].x * w for i in indices])
                y = np.mean([face_landmarks.landmark[i].y * h for i in indices])
                z = np.mean([face_landmarks.landmark[i].z for i in indices]) * 100 
                return x, y, z

            re_x, re_y, re_z = get_coords(RIGHT_EYE)
            le_x, le_y, le_z = get_coords(LEFT_EYE)
            rp_x, rp_y, _ = get_coords(RIGHT_PUPIL)
            lp_x, lp_y, _ = get_coords(LEFT_PUPIL)

            features = [re_x, re_y, re_z, le_x, le_y, le_z, rp_x, rp_y, lp_x, lp_y, rp_x, rp_y, lp_x, lp_y]
            input_df = pd.DataFrame([features], columns=COLUMN_NAMES)

            # --- Prediction ---
            scaled_data = scaler.transform(input_df)
            prediction = model.predict(scaled_data, verbose=0)
            class_idx = np.argmax(prediction)
            label = label_encoder.inverse_transform([class_idx])[0]
            confidence = np.max(prediction) * 100

            # --- PRINT RESULT TO CONSOLE ---
            print(f"Result: {label} | Confidence: {confidence:.2f}% | Pupil L: ({int(lp_x)}, {int(lp_y)}) R: ({int(rp_x)}, {int(rp_y)})")

            # --- Visual UI ---
            color = (0, 255, 0) if label == 'Neurotypical' else (0, 0, 255)
            cv2.putText(image, f"{label} ({confidence:.1f}%)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.circle(image, (int(rp_x), int(rp_y)), 3, (255, 255, 255), -1)
            cv2.circle(image, (int(lp_x), int(lp_y)), 3, (255, 255, 255), -1)

    cv2.imshow('Real-Time Prediction', image)
    if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()