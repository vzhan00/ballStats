from ultralytics import YOLO
import cv2

model = YOLO('/opt/homebrew/runs/detect/train29/weights/best.pt')#.load('yolov8n.pt')
video_path = "./robotshot3.gif"
# video_path = "./shot2.mp4"
cap = cv2.VideoCapture(video_path)

# model.train(data='config.yaml', epochs=1)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()