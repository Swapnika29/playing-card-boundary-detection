# Detecting Playing Card Edges and Corners in Real-World Scenes

### Using Classical Computer Vision

---

## Developed By

*Mounika Teppola & Bala Swapnika Gopi (Group 10)*

---

## About the Project

This project focuses on detecting *Edges and Corners* features of playing cards in complex real-world environments using classical computer vision techniques by leveraging algorithms such as `Canny edge detection, Harris corner detection, and Shi-Tomasi corner detection`

The application supports both images and videos, allowing users to visualize edge maps and corner detections interactively. It highlights the difference between dense feature detection (Harris) and refined feature selection (Shi-Tomasi), making it useful for understanding feature extraction techniques. The project is implemented in both a `command-line version` and an `interactive web-based UI` using Streamlit.

---

## Technologies Used

* Python
* OpenCV
* NumPy
* Streamlit

---

## Project Structure

```id="p1m8jz"
├── main.py        # Command-line version
├── main_UI.py     # Streamlit UI application
├── data           # Dataset containing Images and Videos
├── README.md
```

---

## Features

* Edge detection using **Canny algorithm**
* Corner detection using:

  * Harris Corner Detection
  * Shi-Tomasi Corner Detection
* Supports both **image and video inputs**
* Interactive **threshold tuning**
* Real-time visualization 
* Download processed output
* Clean UI built with Streamlit

---

## File Descriptions

### `main.py`

* Command-line based program
* User provides **image or video filename manually**
* The system:

  * Reads the file
  * Processes it
  * Displays and saves output

Best for quick testing and backend execution

---

### `main_UI.py`

* Streamlit-based interactive application
* User can:

  * Upload image/video directly through UI
  * View results instantly
  * Download processed output

Best for demonstration and user interaction

---

## Installation

### 1. Clone the Repository

```bash id="a4m2j9"
git clone <your-repo-link>
cd <your-project-folder>
```

---

### 2. Create Virtual Environment (Optional but Recommended)

```bash id="k8z2dx"
python -m venv venv
```

#### Activate:

* Windows:

```bash id="z7x1lp"
venv\Scripts\activate
```

* Mac/Linux:

```bash id="y3k9rt"
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash id="n3j1fd"
pip install streamlit opencv-python numpy
```

---

## How to Run

### Run Command-Line Version

```bash id="p9d3xm"
python main.py
```

Enter file name (e.g., `image.jpg` or `video.mp4`) when prompted

---

### Run UI Application

```bash id="m8q2ws"
streamlit run main_UI.py
```

Open browser at: `http://localhost:8501`

---

## Input Requirements

* **Images:** `.jpg`, `.jpeg`, `.png`
* **Videos:** `.mp4`, `.avi`, `.mov`

---

## Output

* Original input
* Edge-detected image (Canny)
* Harris corner detections (dense)
* Shi-Tomasi corner detections (refined)
* Combined visualization
* Download option for processed results

---

## Future Improvements

* Playing card segmentation using contours
* Real-time webcam detection

---

# Ready to Use!

Clone → Install → Run → Upload → Detect → Download 
