import cv2
import matplotlib.pyplot as plt
import numpy as np

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
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  # Display the frame
  plt.imshow(frame)
  plt.axis('off')
  plt.show()


def show_specific_frame_and_full_path(video_path, frame_number, coords_x, coords_y, peaks_both, filename):
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
    out = cv2.VideoWriter(f'Sources/{filename}', fourcc, 30.0, (frame.shape[1], frame.shape[0]))

    while True:
        # Plot circles in each frame
        for x, y in zip(coords_x, coords_y):
            cv2.circle(frame, (x, y), radius=5, color=(0, 165, 255, 0), thickness=8)

        for x, y in zip(coords_x[peaks_both], coords_y[peaks_both]):
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


def plot_top_down_view(input_video_path, M, frame_position, rect_coords):
  cap = cv2.VideoCapture(input_video_path)

  # Check if the video file was opened successfully
  if not cap.isOpened():
    print("Error opening video file")
    return

  # Set the desired frame position
  cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)

  # Read the frame
  ret, frame = cap.read()

  # Check if the frame was read successfully
  if not ret:
    print("Error reading frame")
    return

  # Convert frame to RGB (if needed)
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  # Perform perspective transformation
  top_down_view = cv2.warpPerspective(frame, M, (1920, 1080))

  # Draw a circle on the rectified image at the specified coordinates
  cv2.circle(top_down_view, rect_coords, radius=10, color=((0, 255, 255)), thickness=12)

  # Draw a black dot indicating the axis origin
  cv2.circle(top_down_view, (0, 0), radius=5, color=(0, 0, 0), thickness=25)
  cv2.putText(top_down_view, "Axis Origin", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.3, (0, 0, 0), 2)

  # Display the not rectified frame and the top-down view side by side
  fig, axes = plt.subplots(1, 2, figsize=(10, 5))
  axes[0].imshow(frame)
  axes[0].axis('off')
  axes[0].set_title('Not Rectified Frame')
  axes[1].imshow(top_down_view)
  axes[1].axis('off')
  axes[1].set_title('Rectified Frame')
  plt.tight_layout()
  plt.show()


def video_with_Y(video_path,frame_number, coords_y, peaks_Y, filename):
    """
    Opens a video and displays a specific frame with circles plotted on X and Y coordinates,
    and shows a scatter plot next to the video.

    Args:
        video_path: Path to the video file.
        frame_number: The index of the frame to display (starting from 0).
        coords_x: List of x-coordinates.
        coords_y: List of y-coordinates.
        peaks_Y: frames in which a peak in Y coordinate happens.
        filename: Output filename for saving the video.
    """

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file was opened successfully
    if not cap.isOpened():
      print("Error opening video file")
      return

    # Set the desired frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the initial frame
    ret, frame = cap.read()
    if not ret:
      print("Error reading frame")
      return

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'Sources/{filename}', fourcc, 30.0, (frame.shape[1] * 2, frame.shape[0]))

    # Setup the Matplotlib figure
    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots(figsize=(4, 3))
    scatter_plot, = ax.plot([], [], 'o', color='orange')  # Initial scatter plot settings
    ax.set_xlim(0, 500)
    ax.set_ylim(min(coords_y.max()-coords_y) - 10, max(coords_y.max()-coords_y) + 10)
    ax.set_title('Y Coordinate in time')
    ax.grid(True)

    while cap.isOpened() and frame_number < 440:
      # Update scatter plot with current frame coordinates
      scatter_plot.set_data(list(range(frame_number+1)), coords_y.max() - coords_y[:frame_number + 1])

      # Plot black dot if the frame is in peaks
      if frame_number in peaks_Y:
        ax.plot(frame_number, coords_y.max() - coords_y[frame_number], 'ko')

      fig.canvas.draw()
      fig.canvas.flush_events()

      # Convert Matplotlib figure to a NumPy array image
      chart_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
      chart_img = chart_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
      chart_img = cv2.cvtColor(chart_img, cv2.COLOR_RGB2BGR)

      # Resize to match the frame height and stack side by side
      chart_img_resized = cv2.resize(chart_img, (frame.shape[1], frame.shape[0]))
      combined_image = np.hstack((frame, chart_img_resized))

      # Write the combined image to the output video
      out.write(combined_image)

      # Display the combined image
      cv2.imshow("Video with Scatter Plot", combined_image)

      # Break the loop if 'q' is pressed
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

      # Read the next frame
      ret, frame = cap.read()
      frame_number += 1

      # Break the loop if the video has ended
      if not ret:
        break

    # Release the video capture and writer objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    plt.ioff()  # Turn off interactive mode
    plt.close(fig)



def video_with_coordinates(video_path, frame_number, coords_x, coords_y, peaks, filename):
  """
  Opens a video and displays a specific frame with circles plotted on X and Y coordinates,
  and shows a scatter plot next to the video.

  Args:
    video_path: Path to the video file.
    frame_number: The index of the frame to display (starting from 0).
    coords_x: List of x-coordinates.
    coords_y: List of y-coordinates.
    peaks: Indices of peak coordinates.
    filename: Output filename for saving the video.
  """

  # Open the video file
  cap = cv2.VideoCapture(video_path)

  # Check if the video file was opened successfully
  if not cap.isOpened():
    print("Error opening video file")
    return

  # Set the desired frame position
  cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

  # Read the initial frame
  ret, frame = cap.read()
  if not ret:
    print("Error reading frame")
    return

  # Create VideoWriter object
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out = cv2.VideoWriter(f'Sources/{filename}', fourcc, 30.0, (frame.shape[1] * 2, frame.shape[0]))

  # Setup the Matplotlib figure
  plt.ion()  # Enable interactive mode
  fig, ax = plt.subplots(figsize=(4, 3))
  scatter_plot, = ax.plot([], [], 'o', color='orange')  # Initial scatter plot settings
  ax.set_xlim(min(coords_x) - 10, max(coords_x) + 10)
  ax.set_ylim(min(coords_y.max()-coords_y) - 10, max(coords_y.max()-coords_y) + 10)
  ax.set_title('Scatter Plot of Coordinates')
  ax.grid(True)

  while cap.isOpened():
    # Update scatter plot with current frame coordinates
    scatter_plot.set_data(coords_x[:frame_number + 1], coords_y.max() - coords_y[:frame_number + 1])

    # Plot black dot if the frame is in peaks
    if frame_number in peaks:
      ax.plot(coords_x[frame_number], coords_y.max() - coords_y[frame_number], 'ko')

    fig.canvas.draw()
    fig.canvas.flush_events()

    # Convert Matplotlib figure to a NumPy array image
    chart_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    chart_img = chart_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    chart_img = cv2.cvtColor(chart_img, cv2.COLOR_RGB2BGR)

    # Resize to match the frame height and stack side by side
    chart_img_resized = cv2.resize(chart_img, (frame.shape[1], frame.shape[0]))
    combined_image = np.hstack((frame, chart_img_resized))

    # Write the combined image to the output video
    out.write(combined_image)

    # Display the combined image
    cv2.imshow("Video with Scatter Plot", combined_image)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

    # Read the next frame
    ret, frame = cap.read()
    frame_number += 1

    # Break the loop if the video has ended
    if not ret:
      break

  # Release the video capture and writer objects
  cap.release()
  out.release()
  cv2.destroyAllWindows()
  plt.ioff()  # Turn off interactive mode
  plt.close(fig)