import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from config import Config


# ====================== ADVANCED PREPROCESSING ======================
def preprocess_for_mnist(frame):
    """Highly improved preprocessing for camera"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Define Region of Interest (Center area)
    h, w = gray.shape
    roi_size = 300
    x1 = w//2 - roi_size//2
    y1 = h//2 - roi_size//2
    roi = gray[y1:y1+roi_size, x1:x1+roi_size]
    
    # Blur to reduce noise
    blurred = cv2.GaussianBlur(roi, (7, 7), 0)
    
    # Strong thresholding
    _, thresh = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY_INV)
    
    # Morphological operations to make digit thicker and cleaner
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    processed = np.zeros((28, 28), dtype=np.uint8)
    
    if contours:
        # Take the largest contour
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        
        if w > 10 and h > 10:  # Minimum size filter
            digit = thresh[y:y+h, x:x+w]
            
            # Resize while maintaining aspect ratio
            aspect = w / float(h)
            if aspect > 1:
                new_w = 20
                new_h = max(1, int(20 / aspect))
            else:
                new_h = 20
                new_w = max(1, int(20 * aspect))
            
            digit_resized = cv2.resize(digit, (new_w, new_h))
            
            # Center the digit in 28x28 image
            x_offset = (28 - new_w) // 2
            y_offset = (28 - new_h) // 2
            processed[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = digit_resized
    
    # If no good contour found, fallback
    if processed.max() == 0:
        processed = cv2.resize(thresh, (28, 28))
    
    # Normalize
    normalized = processed.astype('float32') / 255.0
    input_tensor = np.expand_dims(normalized, axis=(0, -1))
    
    return input_tensor, processed, roi


# ====================== LOAD MODEL ======================
if not os.path.exists(Config.MODEL_PATH):
    print("❌ Model not found! Train first:")
    print("   python train.py")
    exit()

model = load_model(Config.MODEL_PATH)
print("✅ Model Loaded Successfully!\n")


# ====================== MAIN APPLICATION ======================
def main():
    cap = cv2.VideoCapture(Config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    
    print("🎥 Improved Camera Detection Started")
    print("Write digit clearly inside the GREEN box")
    print("Press 'q' → Quit | 's' → Save Image\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        input_tensor, processed, roi = preprocess_for_mnist(frame)
        
        # Predict
        prediction = model.predict(input_tensor, verbose=0)
        digit = np.argmax(prediction[0])
        confidence = np.max(prediction[0]) * 100
        
        # Draw ROI box
        h, w = frame.shape[:2]
        roi_size = 300
        x1 = w//2 - roi_size//2
        y1 = h//2 - roi_size//2
        cv2.rectangle(frame, (x1, y1), (x1+roi_size, y1+roi_size), (0, 255, 0), 3)
        
        # Display Prediction
        color = (0, 255, 0) if confidence > 70 else (0, 140, 255)
        cv2.putText(frame, f"Digit: {digit}", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.3, color, 6)
        cv2.putText(frame, f"Confidence: {confidence:.1f}%", (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
        
        # Show windows
        cv2.imshow("Live Recognition", frame)
        cv2.imshow("Processed MNIST Input", cv2.resize(processed, (200, 200)))
        cv2.imshow("ROI", roi)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"digit_{digit}_{int(confidence)}perc.jpg"
            cv2.imwrite(filename, frame)
            print(f"💾 Saved: {filename}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()