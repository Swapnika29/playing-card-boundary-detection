import os
import cv2
import tempfile
import numpy as np
import streamlit as st

st.title("Detecting Playing Card Edges and Corners in Real-World Scenes Using Classical Computer Vision")

option = st.radio("Select input type:", ["Image", "Video"])

uploaded_file = st.file_uploader("Upload file", type=["jpg","jpeg","png","mp4","avi","mov"])

def t(img):
    return cv2.resize(img, (400, 300))

def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0,0), (len(text)*11+10, 28), (0,0,0), -1)
    cv2.putText(out, text, (5,20), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255,255,255), 1)
    return out

def process_frame(image):
    image = cv2.resize(image, (800, 600))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)
    edge_dilated = cv2.dilate(edges, np.ones((15,15), np.uint8))

    # Apply Harris corner detection, keep strong corners on edges and extract their coordinates   
    harris = cv2.cornerHarris(np.float32(gray), 4, 3, 0.04)
    harris = cv2.dilate(harris, None)
    harris_mask = (harris > 0.01 * harris.max()) & (edge_dilated > 0)
    coords = np.argwhere(harris_mask)

    # Detect Shi-Tomasi corners, filter those on edges and store their pixel coordinates  
    shi_pts = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)

    shi_corners = []
    if shi_pts is not None:
        for c in shi_pts:
            x, y = c.ravel().astype(int)
            if edge_dilated[y, x] > 0:
                shi_corners.append((x, y))

    p1 = label(t(image), "Original")
    p2 = label(t(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)), "Canny")

    harris_out = image.copy()
    for y, x in coords[::6]:
        cv2.circle(harris_out, (x,y), 2, (0,0,255), -1)
    p3 = label(t(harris_out), "Harris")

    shi_out = image.copy()
    for (x,y) in shi_corners:
        cv2.circle(shi_out, (x,y), 4, (0,255,255), -1)
    p4 = label(t(shi_out), "Shi-Tomasi")

    combined = image.copy()
    for y, x in coords[::6]:
        cv2.circle(combined, (x,y), 2, (0,0,255), -1)
    for (x,y) in shi_corners:
        cv2.circle(combined, (x,y), 4, (0,255,255), 1)
    p5 = label(t(combined), "Combined")

    empty = np.zeros_like(p1)

    grid = np.vstack([
        np.hstack([p1, p2, p3]),
        np.hstack([p4, p5, empty])
    ])

    return grid


if uploaded_file is not None:

    if st.button("Process"):

        # Save the uploaded file temporarily for processing
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        # Process an input image file, save the processed output and display the result
        if option == "Image":
            img = cv2.imread(tfile.name)
            output = process_frame(img)

            st.image(output, channels="BGR", caption="Processed Image")

            out_path = "output_image.jpg"
            cv2.imwrite(out_path, output)

            with open(out_path, "rb") as f:
                st.download_button("⬇ Download Image", f, file_name="output.jpg")

        # Load a video file, get its FPS and initialize a VideoWriter to save the processed output
        elif option == "Video":
            cap = cv2.VideoCapture(tfile.name)

            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            out_path = "output_video.mp4"
            out = cv2.VideoWriter(out_path, fourcc, fps, (1200, 600))

            st.info("Processing video... please wait")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                grid = process_frame(frame)
                out.write(grid)

            cap.release()
            out.release()

            st.success("Video processing complete!")

            # Open the processed video file and display it in the Streamlit app
            video_file = open(out_path, 'rb')
            st.video(video_file.read())

            # Provide a download button in Streamlit to save the processed video file
            with open(out_path, "rb") as f:
                st.download_button("Download Video", f, file_name="output.mp4")
