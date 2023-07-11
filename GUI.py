import cv2
from keras.models import load_model
import tensorflow_hub as hub
import numpy as np

if __name__ == '__main__':
    model = load_model('mobilenet_v3.h5', custom_objects={'KerasLayer': hub.KerasLayer})
    cap = cv2.VideoCapture(0)  # 0 for default camera, change if using a different camera
    labels = ['arborio', 'basmati', 'ipsala', 'jasmine', 'karacadag']

    while True:
        ret, frame = cap.read()  # Read a frame from the video
        
        # Preprocess the frame
        resized_frame = cv2.resize(frame, (224, 224))
        normalized_frame = resized_frame / 255.0  # Normalize the pixel values
        
        # Expand dimensions to match the input shape of the model
        input_frame = np.expand_dims(normalized_frame, axis=0)
        
        # Perform classification
        predictions = model.predict(input_frame)
        class_index = np.argmax(predictions)
        
        class_label = labels[class_index]
        
        # Display the class label and its accuracy on the frame in one line
        cv2.putText(frame, f'{class_label}: {predictions[0][class_index]:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Video Classification', frame)
        
        keyboard_input = cv2.waitKey(1)
        if keyboard_input == 27:  # 27 is the ASCII for the esc key on your keyboard
            break

    cap.release()
    cv2.destroyAllWindows()