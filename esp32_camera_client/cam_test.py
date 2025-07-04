import imagezmq
import cv2

# Create a sender object to receive the video stream
image_hub = imagezmq.ImageHub()

# Correct ESP32-CAM stream URL
stream_url = "http://192.168.0.102/"

cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("❌  Cannot open camera. Check the stream URL.")
else:
    print("✅  Camera connected successfully")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌  Failed to grab frame")
            break

        # Send frame to the image hub for processing
        image_hub.send_image('ESP32_Camera', frame)

        # Display the frame in a window
        cv2.imshow("ESP32-CAM Stream", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()
