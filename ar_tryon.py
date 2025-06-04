import os
import streamlit as st
from PIL import Image, ImageOps
import cv2
import numpy as np
import io
import mediapipe as mp
from rembg import remove

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=2, min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

def detect_pose(image, is_garment=False):
    image = image.astype(np.uint8)
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    landmarks_data = {}

    if results.pose_landmarks:
        draw_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2) if is_garment else mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=3)
        conn_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1) if is_garment else mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, draw_spec, conn_spec)

        for idx, name in {
            11: "left_shoulder", 12: "right_shoulder", 13: "left_elbow",
            14: "right_elbow", 15: "left_wrist", 16: "right_wrist",
            23: "left_hip", 24: "right_hip"
        }.items():
            landmark = results.pose_landmarks.landmark[idx]
            landmarks_data[name] = {"x": landmark.x, "y": landmark.y, "z": landmark.z, "visibility": landmark.visibility}
    return image, landmarks_data

def remove_background(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    result = remove(img_byte_arr.getvalue(), post_process_mask=True)
    result_img = Image.open(io.BytesIO(result)).convert("RGBA")

    alpha = result_img.split()[-1]
    alpha = Image.fromarray(cv2.medianBlur(np.array(alpha).astype(np.uint8), 3))
    result_img.putalpha(alpha)
    return result_img

def calculate_garment_size(body_landmarks, image_shape, garment_img, size_factor=1.0):
    if all(key in body_landmarks for key in ["left_shoulder", "right_shoulder", "left_elbow"]):
        shoulder_width = abs(body_landmarks["left_shoulder"]["x"] - body_landmarks["right_shoulder"]["x"]) * image_shape[0]
        arm_length = abs(body_landmarks["left_shoulder"]["y"] - body_landmarks["left_elbow"]["y"]) * image_shape[1]

        garment_arr = np.array(garment_img).astype(np.uint8)
        if garment_arr.shape[2] == 4:
            garment_arr = cv2.cvtColor(garment_arr, cv2.COLOR_RGBA2BGR)
        _, garment_landmarks = detect_pose(garment_arr, is_garment=True)

        if garment_landmarks and "left_elbow" in garment_landmarks:
            garment_shoulder_width = abs(garment_landmarks["left_shoulder"]["x"] - garment_landmarks["right_shoulder"]["x"]) * garment_img.width
            garment_arm_length = abs(garment_landmarks["left_shoulder"]["y"] - garment_landmarks["left_elbow"]["y"]) * garment_img.height
            width_scale = (shoulder_width / garment_shoulder_width) * size_factor
            height_scale = (arm_length / garment_arm_length) * size_factor
            new_width = int(garment_img.width * width_scale)
            new_height = int(garment_img.height * height_scale)
        else:
            new_width = int(shoulder_width * 1.8 * size_factor)
            new_height = int(arm_length * 2.5 * size_factor)

        aspect_ratio = garment_img.width / garment_img.height
        max_width = int(image_shape[0] * 0.9)
        max_height = int(image_shape[1] * 0.8)
        if new_width / new_height > aspect_ratio:
            new_width = min(int(new_height * aspect_ratio), max_width)
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = min(int(new_width / aspect_ratio), max_height)
            new_width = int(new_height * aspect_ratio)

        return new_width, new_height
    return None, None

def blend_images(body_img, garment_img, position, opacity=0.9):
    body_img = body_img.convert("RGBA")
    result = body_img.copy()

    body_arr = np.array(body_img).astype(np.uint8)
    garment_arr = np.array(garment_img).astype(np.uint8)

    if garment_arr.shape[2] == 3:
        garment_arr = np.dstack([garment_arr, np.ones(garment_arr.shape[:2], dtype=np.uint8) * 255])

    garment_alpha = garment_arr[:, :, 3] / 255.0
    garment_rgb = garment_arr[:, :, :3]
    x, y = position
    h, w = garment_arr.shape[:2]

    if y + h <= body_arr.shape[0] and x + w <= body_arr.shape[1]:
        region = body_arr[y:y+h, x:x+w]
        if region.size == 0:
            print(f"Error: Region is empty at position ({x}, {y}) with size ({h}, {w})")
            return body_img

        garment_matched = garment_rgb.copy()  # Skip histogram matching to retain original color
        for c in range(3):
            region[:, :, c] = region[:, :, c] * (1 - garment_alpha * opacity) + garment_matched[:, :, c] * garment_alpha * opacity
        body_arr[y:y+h, x:x+w] = region
        return Image.fromarray(body_arr)
    return body_img

def align_garment(body_img, shirt_img, body_landmarks, size_factor=1.0, opacity=0.9, x_offset=0, y_offset=0):
    shirt_width, shirt_height = shirt_img.size
    body_width, body_height = body_img.size
    shirt_arr = np.array(shirt_img).astype(np.uint8)

    if shirt_arr.shape[2] == 4:
        shirt_arr = cv2.cvtColor(shirt_arr, cv2.COLOR_RGBA2BGR)
    _, shirt_landmarks = detect_pose(shirt_arr, is_garment=True)

    if shirt_landmarks and "left_elbow" in shirt_landmarks and "left_elbow" in body_landmarks:
        cx = (body_landmarks["left_shoulder"]["x"] + body_landmarks["right_shoulder"]["x"]) / 2
        y = body_landmarks["left_shoulder"]["y"]
        ey = body_landmarks["left_elbow"]["y"]
        arm_length_ratio = abs(y - ey) / abs(shirt_landmarks["left_shoulder"]["y"] - shirt_landmarks["left_elbow"]["y"]) * size_factor

        x_pos = int(cx * body_width - (shirt_landmarks["left_shoulder"]["x"] + shirt_landmarks["right_shoulder"]["x"]) / 2 * shirt_width * arm_length_ratio)
        y_pos = int(y * body_height - shirt_landmarks["left_shoulder"]["y"] * shirt_height * arm_length_ratio)
    else:
        cx = (body_landmarks["left_shoulder"]["x"] + body_landmarks["right_shoulder"]["x"]) / 2
        x_pos = int(cx * body_width - shirt_width / 2)
        y_pos = int(body_landmarks["left_shoulder"]["y"] * body_height - shirt_height * 0.15)

    x_pos += x_offset
    y_pos += y_offset
    return blend_images(body_img, shirt_img, (x_pos, y_pos), opacity)

def virtual_tryon_page():
    st.title("🧥 Virtual Try-On: Enhanced AR Fitting")
    st.markdown("*Upload your photo and a garment to see the virtual try-on result!*")

    col1, col2 = st.columns(2)
    with col1:
        body_file = st.file_uploader("Upload body photo", type=["jpg", "jpeg", "png"])
    with col2:
        shirt_file = st.file_uploader("Upload shirt/garment image", type=["jpg", "jpeg", "png"])

    if body_file and shirt_file:
        body_img = Image.open(body_file).convert("RGB")
        pose_img, body_landmarks = detect_pose(np.array(body_img).astype(np.uint8))
        st.image(pose_img, caption="Body Pose Detection", use_column_width=True)

        if "left_elbow" in body_landmarks:
            st.sidebar.header("Garment Controls")
            size_factor = st.sidebar.slider("Size Factor", 0.7, 1.5, 1.0, 0.05)
            opacity = st.sidebar.slider("Opacity", 0.5, 1.0, 0.9, 0.05)
            x_offset = st.sidebar.slider("Horizontal Shift", -100, 100, 0, 5)
            y_offset = st.sidebar.slider("Vertical Shift", -100, 100, 0, 5)

            with st.spinner("Removing background..."):
                shirt_img = remove_background(Image.open(shirt_file))
                st.image(shirt_img, caption="Garment After BG Removal", use_column_width=True)

            width, height = calculate_garment_size(body_landmarks, body_img.size, shirt_img, size_factor)
            if width and height:
                shirt_img = shirt_img.resize((width, height), Image.LANCZOS)
                final_img = align_garment(body_img.convert("RGBA"), shirt_img, body_landmarks, size_factor, opacity, x_offset, y_offset)
                st.image(final_img, caption="Final Try-On", use_column_width=True)

                buf = io.BytesIO()
                final_img.save(buf, format="PNG")
                st.download_button("Download Result", buf.getvalue(), "tryon_result.png", "image/png")
            else:
                st.error("Unable to calculate size — ensure elbows are clearly visible.")
        else:
            st.error("Pose detection failed — please upload a photo with visible elbows.")

if __name__ == "__main__":
    main()
