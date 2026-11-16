from __future__ import annotations

from collections import Counter
from pathlib import Path
import os
import tempfile

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import streamlit as st
from ultralytics import YOLO

from helmet_logic import (
    associate_helmets,
    helmet_label,
    helmet_status_mode,
)

BASE_DIR = Path(__file__).resolve().parent
DEMO_MODEL = BASE_DIR / "yolov8n.pt"
DATASET_DIRS = (
    BASE_DIR / "dataset" / "images" / "train",
    BASE_DIR / "dataset" / "images" / "val",
    BASE_DIR / "dataset" / "labels" / "train",
    BASE_DIR / "dataset" / "labels" / "val",
)

st.set_page_config(
    page_title="AI Helmet Detection",
    page_icon="🏍️",
    layout="wide",
)


def find_helmet_model() -> Path | None:
    model_candidates = [
        BASE_DIR / "best.pt",
        *(BASE_DIR / "runs" / "detect").glob(
            "helmet_detection*/weights/best.pt"
        ),
    ]
    existing_models = [path for path in model_candidates if path.is_file()]
    return max(existing_models, key=lambda path: path.stat().st_mtime, default=None)


@st.cache_resource(max_entries=3, show_spinner="Loading YOLO model...")
def load_model(model_path: str | None, model_bytes: bytes | None) -> YOLO:
    if model_bytes is None:
        if model_path is None:
            raise FileNotFoundError("No model weights were selected.")
        return YOLO(model_path)

    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as model_file:
            model_file.write(model_bytes)
            temporary_path = model_file.name
        return YOLO(temporary_path)
    finally:
        if temporary_path is not None:
            os.unlink(temporary_path)


def annotate_frame(
    model: YOLO,
    frame: np.ndarray,
    confidence: float,
    image_size: int,
) -> tuple[np.ndarray, list[str]]:
    result = model.predict(
        frame,
        conf=confidence,
        imgsz=image_size,
        verbose=False,
    )[0]
    output = frame.copy()
    labels: list[str] = []

    if result.boxes is None:
        return output, labels

    detections: list[tuple[int, int, int, int, str, float]] = []
    motorcycle_boxes: list[tuple[float, float, float, float]] = []
    helmet_boxes: list[tuple[float, float, float, float]] = []
    for box in result.boxes:
        x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
        class_id = int(box.cls[0].item())
        score = float(box.conf[0].item())
        class_name = str(result.names[class_id])
        label = helmet_label(class_name)
        detections.append((x1, y1, x2, y2, label, score))
        coordinates = (float(x1), float(y1), float(x2), float(y2))
        if label == "Motorcycle":
            motorcycle_boxes.append(coordinates)
        elif label == "Helmet":
            helmet_boxes.append(coordinates)

    status_mode = helmet_status_mode(result.names.values())
    paired_statuses = (
        associate_helmets(motorcycle_boxes, helmet_boxes)
        if status_mode == "paired"
        else []
    )
    motorcycle_index = 0

    for x1, y1, x2, y2, label, score in detections:
        status = label
        if status_mode == "paired" and label == "Motorcycle":
            status = (
                "With helmet"
                if paired_statuses[motorcycle_index]
                else "No helmet"
            )
            motorcycle_index += 1
            labels.append(status)
            display_label = f"Motorcycle - {status}"
        elif status_mode == "explicit":
            if label == "Helmet":
                status = "With helmet"
            if status in {"With helmet", "No helmet"}:
                labels.append(status)
            display_label = status
        else:
            labels.append(label)
            display_label = label

        if status == "No helmet":
            color = (35, 75, 235)
        elif status == "With helmet" or label == "Helmet":
            color = (45, 175, 75)
        else:
            color = (220, 145, 35)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        text = display_label
        if status_mode != "paired" or label != "Motorcycle":
            text = f"{text}: {score:.2f}"
        text_size, _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            2,
        )
        text_y = max(y1 - 8, text_size[1] + 8)
        cv2.rectangle(
            output,
            (x1, text_y - text_size[1] - 8),
            (x1 + text_size[0] + 8, text_y + 4),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            output,
            text,
            (x1 + 4, text_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return output, labels


def show_summary(labels: list[str], status_mode: str | None) -> None:
    counts = Counter(labels)
    if status_mode is not None:
        with_helmet = counts["With helmet"]
        without_helmet = counts["No helmet"]
        columns = st.columns(3)
        if status_mode == "explicit":
            columns[0].metric("Riders classified", with_helmet + without_helmet)
            columns[1].metric("With helmet", with_helmet)
            columns[2].metric("Without helmet", without_helmet)
        else:
            columns[0].metric("Motorcycles checked", with_helmet + without_helmet)
            columns[1].metric("Helmet detected", with_helmet)
            columns[2].metric("No helmet detected", without_helmet)
        if status_mode == "paired":
            st.caption(
                "Status is estimated by matching a detected helmet to each "
                "motorcycle; missed or hidden helmets can cause false warnings."
            )
    elif counts:
        columns = st.columns(2)
        columns[0].metric("Objects detected", len(labels))
        columns[1].metric("Object classes", len(counts))
        st.table(
            [
                {"Class": name, "Detections": count}
                for name, count in sorted(counts.items())
            ]
        )


def render_training_setup() -> None:
    st.subheader("Train a helmet-detection model")
    st.write(
        "The included `yolov8n.pt` model detects common objects but has not "
        "been trained to identify helmets. A labeled YOLO dataset is required "
        "for helmet-specific results. The provided data labels helmet and "
        "motorcycle objects; no-helmet status is estimated from a missing "
        "helmet detection, not learned as a separate class."
    )

    missing = [path for path in DATASET_DIRS if not path.is_dir()]
    if missing:
        st.warning(
            "The required dataset folders are not present. Add your dataset "
            "before starting training."
        )
        st.code(
            "dataset/\n"
            "  images/\n"
            "    train/  # training images\n"
            "    val/    # validation images\n"
            "  labels/\n"
            "    train/  # YOLO .txt labels matching the image names\n"
            "    val/\n"
            "\n"
            "python train.py\n"
            "streamlit run app.py"
        )
    else:
        st.success("All expected dataset folders are available.")

    st.caption(f"Dataset configuration: {BASE_DIR / 'dataset.yaml'}")
    st.caption("Dataset classes: helmet, motorcycle.")


st.title("AI helmet detection")
st.caption("Image and video inference with YOLO.")

helmet_model_path = find_helmet_model()
model_options: list[str] = []
if helmet_model_path is not None:
    model_options.append("Local helmet-status model")
if DEMO_MODEL.is_file():
    model_options.append("General-object demo (not helmet classification)")
model_options.append("Upload helmet model (.pt)")

with st.sidebar:
    st.header("Detection settings")
    page = st.radio("Workspace", ("Detect", "Training setup"))

if page == "Training setup":
    render_training_setup()
    st.stop()

if len(model_options) == 1 and not DEMO_MODEL.is_file():
    st.error(
        "No local model was found. Upload trained `.pt` weights in the sidebar "
        "or add `yolov8n.pt` to this project for the general-object demo."
    )

with st.sidebar:
    selected_model = st.selectbox("Model", model_options)
    uploaded_model = None
    if selected_model == "Upload helmet model (.pt)":
        uploaded_model = st.file_uploader(
            "Choose trained YOLO weights",
            type=["pt"],
            help=(
                "Use a model trained with helmet and motorcycle classes, "
                "or explicit with/without-helmet classes."
            ),
        )
    confidence = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.45,
        step=0.05,
    )
    image_size = st.select_slider(
        "Inference image size",
        options=(320, 480, 640, 800, 960, 1280),
        value=1280,
    )
    detection_type = st.radio("Input type", ("Image", "Video"))

if selected_model == "Local helmet-status model":
    model_path = str(helmet_model_path)
    model_bytes = None
    helmet_model = True
elif selected_model == "General-object demo (not helmet classification)":
    model_path = str(DEMO_MODEL)
    model_bytes = None
    helmet_model = False
else:
    model_path = None
    model_bytes = uploaded_model.getvalue() if uploaded_model else None
    helmet_model = True
    if model_bytes is None:
        st.info(
            "Upload trained helmet-model weights, or select the general-object "
            "demo in the sidebar to try the app without a helmet dataset."
        )

if model_path is not None or model_bytes is not None:
    try:
        model = load_model(model_path, model_bytes)
    except Exception as error:
        st.error(f"Could not load the selected YOLO model: {error}")
        st.stop()

    status_mode = helmet_status_mode(model.names.values())
    if not helmet_model:
        st.info(
            "Demo mode detects general objects only. It does not determine "
            "whether a rider is wearing a helmet."
        )
    elif status_mode == "paired":
        st.warning(
            "Helmet status is estimated from helmet and motorcycle detections. "
            "If a helmet is missed or hidden, the motorcycle may be flagged "
            "as having no helmet."
        )
    elif status_mode == "explicit":
        st.info(
            "This model predicts With Helmet and Without Helmet directly. "
            "Small riders, occlusions, and background objects can still be "
            "misclassified; review the boxes before acting on a result."
        )
    elif status_mode is None:
        st.warning(
            "These weights do not contain helmet-status classes. Upload a "
            "model trained for helmet/motorcycle or with/without-helmet "
            "classes to get helmet-status results."
        )

    if detection_type == "Image":
        uploaded_image = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png", "webp"],
        )
        if uploaded_image is not None:
            try:
                image = Image.open(uploaded_image).convert("RGB")
                frame = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
                output, labels = annotate_frame(
                    model,
                    frame,
                    confidence,
                    image_size,
                )
                output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

                original_column, result_column = st.columns(2)
                with original_column:
                    st.image(
                        image,
                        caption="Original image",
                        alt="Uploaded image",
                        width="stretch",
                    )
                with result_column:
                    st.image(
                        output_rgb,
                        caption="Detection result",
                        alt="Image annotated with YOLO detections",
                        width="stretch",
                    )

                show_summary(labels, status_mode)
                success, encoded = cv2.imencode(".png", output)
                if not success:
                    raise RuntimeError("Could not encode the annotated image.")
                st.download_button(
                    "Download annotated image",
                    data=encoded.tobytes(),
                    file_name="helmet_detection_result.png",
                    mime="image/png",
                )
            except (UnidentifiedImageError, OSError, ValueError, RuntimeError) as error:
                st.error(f"Image processing failed: {error}")
    else:
        uploaded_video = st.file_uploader(
            "Choose a video",
            type=["mp4", "avi", "mov", "mkv"],
        )
        if uploaded_video is not None and st.button(
            "Process video",
            type="primary",
        ):
            try:
                with tempfile.TemporaryDirectory() as temporary_directory:
                    video_suffix = Path(uploaded_video.name).suffix.lower()
                    input_path = Path(temporary_directory) / f"upload{video_suffix}"
                    output_path = Path(temporary_directory) / "annotated.mp4"
                    input_path.write_bytes(uploaded_video.getvalue())

                    capture = cv2.VideoCapture(str(input_path))
                    if not capture.isOpened():
                        raise RuntimeError("OpenCV could not open this video.")

                    fps = capture.get(cv2.CAP_PROP_FPS)
                    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    if width <= 0 or height <= 0:
                        capture.release()
                        raise RuntimeError("The video has invalid dimensions.")

                    fps = fps if fps > 0 else 25.0
                    writer = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        capture.release()
                        raise RuntimeError("OpenCV could not create the output video.")

                    counts: Counter[str] = Counter()
                    processed_frames = 0
                    progress = st.progress(0, text="Preparing video...")
                    try:
                        while True:
                            success, frame = capture.read()
                            if not success:
                                break
                            output, labels = annotate_frame(
                                model,
                                frame,
                                confidence,
                                image_size,
                            )
                            writer.write(output)
                            counts.update(labels)
                            processed_frames += 1
                            if total_frames > 0:
                                progress.progress(
                                    min(processed_frames / total_frames, 1.0),
                                    text=f"Processed {processed_frames} of {total_frames} frames",
                                )
                    finally:
                        capture.release()
                        writer.release()

                    if processed_frames == 0:
                        raise RuntimeError("No readable frames were found.")
                    video_bytes = output_path.read_bytes()

                progress.progress(1.0, text="Video processing complete")
                st.video(
                    video_bytes,
                    alt="Video annotated with YOLO detections",
                )
                st.download_button(
                    "Download processed video",
                    data=video_bytes,
                    file_name="helmet_detection_result.mp4",
                    mime="video/mp4",
                )
                st.metric("Frames processed", processed_frames)
                if status_mode is not None:
                    checked = counts["With helmet"] + counts["No helmet"]
                    summary_columns = st.columns(3)
                    summary_columns[0].metric(
                        "Motorcycle-frame checks",
                        checked,
                    )
                    summary_columns[1].metric(
                        "Helmet detected",
                        counts["With helmet"],
                    )
                    summary_columns[2].metric(
                        "No helmet detected",
                        counts["No helmet"],
                    )
                else:
                    st.caption(f"Total detections across frames: {sum(counts.values())}")
            except (OSError, RuntimeError, ValueError, cv2.error) as error:
                st.error(f"Video processing failed: {error}")
