import cv2
from ultralytics import YOLO
import time
import math
import os
from PIL import Image
from docx import Document
from docx.shared import Inches

# Load the pre-trained YOLO model (ensure the correct model path)
model = YOLO("yolov8n.pt")  # Use the most suitable model for your setup

# Initialize video capture (replace 1 with your webcam index if needed)
cap = cv2.VideoCapture(0)

# To store the last saved bounding box coordinates and time
last_saved_boxes = []
last_save_time = 0
image_counter = 0
save_delay = 2  # Minimum time delay in seconds between saving images
proximity_threshold = 50  # The maximum allowed distance (in pixels) to consider bounding boxes as the same

# Folder to save images and Word report
output_folder = "detections"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to generate a unique filename based on the current timestamp
def generate_unique_filename():
    timestamp = time.strftime("%Y%m%d-%H%M%S")  # Get current timestamp for unique naming
    return os.path.join(output_folder, f"human_detections_report_{timestamp}.docx")

# Define Word document filename
doc_filename = generate_unique_filename()

# Initialize a list to store human IDs and image file paths
human_images = []

# Function to calculate Euclidean distance between two bounding boxes (centers)
def calculate_distance(box1, box2):
    # Calculate the center of each bounding box
    center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
    center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
    # Calculate Euclidean distance
    distance = math.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
    return distance

# Function to create the Word document report
def create_doc_report(doc, total_humans):
    doc.add_heading('Human Detection Report', 0)
    doc.add_paragraph(f'Total Detected Humans: {total_humans}')
    doc.add_paragraph('--------------------------------------')

# Function to add image to the Word document
def add_image_to_doc(doc, image_path, image_id):
    doc.add_paragraph(f"Image ID: {image_id}")
    doc.add_picture(image_path, width=Inches(2))
    doc.add_paragraph('--------------------------------------')

# Loop to capture frames and perform detection
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Perform detection using YOLO
    results = model(frame)

    # Check if results contain any predictions
    if results:
        # Extracting bounding boxes, confidences, and class IDs
        for result in results[0].boxes:  # Access the first result, which is usually a list of boxes
            x1, y1, x2, y2 = result.xyxy[0]  # xyxy format (top-left, bottom-right coordinates)
            conf = result.conf[0]  # Confidence score for detection
            cls = result.cls[0]  # Class index (0 for person, 1 for bicycle, etc.)

            if conf > 0.5 and int(cls) == 0:  # Class 0 corresponds to 'person'
                # Check if the person is a new detection
                current_time = time.time()

                # Check if enough time has passed since the last save
                if current_time - last_save_time > save_delay:
                    # Check if any of the previous bounding boxes are close enough to the new one
                    new_detection = True
                    for box in last_saved_boxes:
                        if calculate_distance((x1, y1, x2, y2), box) < proximity_threshold:
                            new_detection = False
                            break
                    
                    # If it's a new detection, save the image and add to Word document
                    if new_detection:
                        # Save the image with a unique filename
                        timestamp = time.strftime("%Y%m%d-%H%M%S")  # Get current timestamp for unique naming
                        image_filename = f"human_{timestamp}_{image_counter}.jpg"
                        image_path = os.path.join(output_folder, image_filename)
                        cv2.imwrite(image_path, frame)
                        image_counter += 1  # Increment counter for the next image
                        last_saved_boxes.append((x1, y1, x2, y2))  # Store the bounding box of the saved detection
                        last_save_time = current_time  # Update the last save time

                        # Add the saved image to the list of human detections
                        human_images.append((image_path, image_filename.split('.')[0]))  # Save image path and ID

                # Draw the bounding box around the person
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Green box
                label = f"Person {conf:.2f}"  # Add confidence level as label
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show the frame with the bounding boxes
    cv2.imshow("Human Detection", frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Create a new Word document
doc = Document()

# Generate the Word report
create_doc_report(doc, len(human_images))

# Add images and their IDs to the Word document
for idx, (image_path, image_id) in enumerate(human_images, start=1):
    add_image_to_doc(doc, image_path, f"ID: {idx}")

# Save the Word document with a unique name
doc.save(doc_filename)

print(f"Word report saved as {doc_filename}")

# Release resources
cap.release()
cv2.destroyAllWindows()
