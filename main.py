import cv2
from ultralytics import YOLO

# Load the pre-trained YOLO model (ensure the correct model path)
model = YOLO("yolov8n.pt")  # Use the most suitable model for your setup

# Initialize video capture (replace 1 with your webcam index if needed)
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Perform detection using YOLO
    results = model(frame)

    # Print the type of results object to inspect the data structure
    print(type(results))
    
    # Check if results contain any predictions
    if results:
        # Extracting bounding boxes, confidences, and class IDs
        for result in results[0].boxes:  # Access the first result, which is usually a list of boxes
            x1, y1, x2, y2 = result.xyxy[0]  # xyxy format (top-left, bottom-right coordinates)
            conf = result.conf[0]  # Confidence score for detection
            cls = result.cls[0]  # Class index (0 for person, 1 for bicycle, etc.)
            
            if conf > 0.5 and int(cls) == 0:  # Class 0 corresponds to 'person'
                # Draw the bounding box around the person
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Green box
                label = f"Person {conf:.2f}"  # Add confidence level as label
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show the frame with the bounding boxes
    cv2.imshow("Human Detection", frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
