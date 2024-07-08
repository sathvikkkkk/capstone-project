import cv2
import numpy as np

# Attach camera indexed as 0
camera = cv2.VideoCapture(0)

# Setting frame width and frame height as 640 x 480
camera.set(3, 640)
camera.set(4, 480)

# Loading the mountain image
mountain = cv2.imread('mountain.jpg')

# Check if the image was loaded successfully
if mountain is None:
    print("Error: Could not load the image 'mount everest.jpg'. Check the file path.")
    exit()

# Resizing the mountain image as 640 x 480
mountain = cv2.resize(mountain, (640, 480))

while True:
    # Read a frame from the attached camera
    status, frame = camera.read()

    # If we got the frame successfully
    if status:
        # Flip it
        frame = cv2.flip(frame, 1)

        # Convert the image to RGB for easy processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Define the range for white color
        lower_bound = np.array([100, 100, 100])
        upper_bound = np.array([255, 255, 255])

        # Create a mask to detect the white background
        mask = cv2.inRange(frame_rgb, lower_bound, upper_bound)

        # Invert the mask to get the foreground
        mask_inv = cv2.bitwise_not(mask)

        # Extract the foreground using the inverted mask
        foreground = cv2.bitwise_and(frame, frame, mask=mask_inv)

        # Create the background with the mountain image using the mask
        background = cv2.bitwise_and(mountain, mountain, mask=mask)

        # Combine the foreground and background to get the final image
        final_image = cv2.add(foreground, background)

        # Show the final image
        cv2.imshow('frame', final_image)

        # Wait for 1ms before displaying another frame, exit if spacebar is pressed
        code = cv2.waitKey(1)
        if code == 32:
            break

# Release the camera and close all opened wind
camera.release()
cv2.destroyAllWindows()
