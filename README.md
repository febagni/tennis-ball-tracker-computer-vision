<h1 align='center'>Tennis Tracking 🎾</h1>

<h1 align='center'> Image Analysis and Computer Vision project </h1>

<h3>Objectives</h3>
<ul>
  <li>Track the ball </li>
  <li>Detect court lines</li>
  <li>Detect when the ball bounce or is hit by a racket</li>
  <li>Use the court lines detected to rectify the image and obtain the exact position of these bounces</li>

</ul>

<p>To track the ball we used <a href='https://nol.cs.nctu.edu.tw:234/open-source/TrackNet'>TrackNet</a> - deep learning network for tracking high-speed objects.

  
Input            |  Output
:-------------------------:|:-------------------------:
![input_img1](https://github.com/ArtLabss/tennis-tracking/blob/00cfe10b18db1e6a68800921dfbda010f90a74bb/VideoOutput/ezgif.com-gif-maker(3).gif)  |  ![output_img1](https://github.com/ArtLabss/tennis-tracking/blob/0f684fdeef96a715984dc74b62b961f68ff95edc/VideoOutput/ezgif.com-gif-maker.gif)

<h3>How to run</h3>

<p>This project requires compatible <b>GPU</b> to install tensorflow, you can run it on your local machine in case you have one or use <a href='https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwissLL5-MvxAhXwlYsKHbkBDEUQFnoECAMQAw&url=https%3A%2F%2Fcolab.research.google.com%2Fnotebooks%2F&usg=AOvVaw0eDNVclINNdlOuD-YTYiiB'>Google Colaboratory</a> with <b>Runtime Type</b> changed to <b>GPU</b>.</p>
  
<ol>
  <li>
    Clone this repository
  </li>
  
  ```git
  git clone https://github.com/febagni/tennis-ball-tracker-computer-vision/
  ```
  
   <li>
     Download yolov3 weights (237 MB) from <a href="https://pjreddie.com/media/files/yolov3.weights">here</a> and add it to your <a href="/Yolov3">Yolov3 folder</a>.
  </li>
  
  <li>
    Install the requirements using pip 
  </li>
  
  ```python
  pip install -r requirements.txt
  ```
  
   <li>
    Run the following command in the command line
  </li>
  
  ```python
  !python3 predict_video.py --input_video_path=VideoInput/INPUT_play_5.mp4 --output_video_path=VideoOutput/output.mp4 --full_trajectory=1 --rectify=1
  ```
  
  <li>If you are using Google Colab upload all the files to Google Drive, including yolov3 weights from step <strong>2.</strong></li>
  
   <li>
    Create a Google Colaboratory Notebook in the same directory as <code>predict_video.py</code>, change Runtime Type to <strong>GPU</strong> and connect it to Google drive
  </li>
  
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  ```
  
  <li>
    Change the working directory to the one where the Colab Notebook and <code>predict_video.py</code> are. In my case,
  </li>
  
  ```python
  import os 
  os.chdir('drive/MyDrive/Colab Notebooks/tennis-tracking')
  ```
  
  <li>
    Install only 2 requirements, because Colab already has the rest
  </li>
  
  ```python
  !pip install filterpy sktime
  ```
  
  <li>
    Inside the notebook run <code>predict_video.py</code>
  </li>
  
  ```
  !python3 predict_video.py --input_video_path=VideoInput/INPUT_play_5.mp4 --output_video_path=VideoOutput/output.mp4 --full_trajectory=1 --rectify=1
  ```
  
  <p>After the compilation is completed, a new video will be created in <a href="/VideoOutput" target="_blank">VideoOutput folder </a> in which shows the trajectory being detected by the ball. We are not interested completely in this video, instead. We need are using the coordinates of the tracked ball during the frames and the detected court lines to obtain the occurence of bounces and their respective location. As explained in detail on the report.
  
</ol>
