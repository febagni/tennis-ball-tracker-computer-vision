import cv2

# Initialize the coordinate variables
clicked_coords = None

def show_frame_and_get_click(video_path, frame_number):
    global clicked_coords
    clicked_coords = None  # Reset coordinates for each function call

    # Capture the video
    cap = cv2.VideoCapture(video_path)

    # Set the frame to be captured
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the specific frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to capture frame.")
        return

    # Define the mouse callback function to get coordinates
    def get_coordinates(event, x, y, flags, param):
        global clicked_coords
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_coords = (x, y)
            print(f"Clicked coordinates: {clicked_coords}")
            cv2.circle(frame, clicked_coords, 5, (0, 255, 0), -1)  # Mark the clicked point with a green circle
            cv2.imshow("Frame", frame)

    # Display the frame
    cv2.imshow("Frame", frame)
    cv2.setMouseCallback("Frame", get_coordinates)

    # Wait until the user presses a key or clicks
    while True:
        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit the loop
            break
        if clicked_coords:
            break  # Exit loop if a click has been registered

    # Release the capture and close the window
    cap.release()
    cv2.destroyAllWindows()
    
    return clicked_coords


def show_specific_frame_and_coord(video_path, frame_number, x, y):
  """Opens a video and displays a specific frame.

  Args:
    video_path: Path to the video file.
    frame_number: The index of the frame to display (starting from 0).
  """

  # Open the video file
  cap = cv2.VideoCapture(video_path)

  # Check if the video file was opened successfully
  if not cap.isOpened():
    print("Error opening video file")
    return

  # Set the desired frame position
  cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

  # Read the frame
  ret, frame = cap.read()

  print(frame.shape[:2])

  if not ret:
    print("Error reading frame")
    return

  cv2.circle(frame, (x, y), radius=5, color=(255,255,255,0), thickness=50)

  # Display the frame
  cv2.imshow("Frame", frame)
  cv2.waitKey(10000)

  # Release the video capture object
  cap.release()
  cv2.destroyAllWindows()


def show_specific_frame(video_path, frame_number):
  """Opens a video and displays a specific frame.

  Args:
    video_path: Path to the video file.
    frame_number: The index of the frame to display (starting from 0).
  """

  # Open the video file
  cap = cv2.VideoCapture(video_path)

  # Check if the video file was opened successfully
  if not cap.isOpened():
    print("Error opening video file")
    return

  # Set the desired frame position
  cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

  # Read the frame
  ret, frame = cap.read()

  if not ret:
    print("Error reading frame")
    return


  frame = cv2.resize(frame, (928, 522))

  print(frame.shape)

  # Display the frame
  cv2.imshow("Frame", frame)
  cv2.waitKey(10000)

  # Release the video capture object
  cap.release()
  cv2.destroyAllWindows()


def show_specific_frame_and_full_path(video_path, frame_number, coords_x, coords_y, peaks):
    """Opens a video and displays a specific frame and plots circles in X and Y axis.
    """

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Set the desired frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the frame
    ret, frame = cap.read()

    print(frame.shape[:2])

    if not ret:
        print("Error reading frame")
        return

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('with_peaks.mp4', fourcc, 30.0, (frame.shape[1], frame.shape[0]))

    while True:
        # Plot circles in each frame
        for x, y in zip(coords_x, coords_y):
            cv2.circle(frame, (x, y), radius=5, color=(0, 165, 255, 0), thickness=8)

        for x, y in zip(coords_x[peaks], coords_y[peaks]):
            cv2.circle(frame, (x, y), radius=5, color=(0, 0, 0, 0), thickness=5)
        # Write the frame to the video file
        out.write(frame)

        # Display the frame
        cv2.imshow("Frame", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Read the next frame
        ret, frame = cap.read()

        # Break the loop if the video has ended
        if not ret:
            break

    # Release the video capture object
    cap.release()

    # Release the video writer object
    out.release()

    cv2.destroyAllWindows()