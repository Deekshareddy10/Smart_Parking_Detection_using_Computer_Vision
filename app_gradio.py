"""
Smart Parking Detection - Gradio Interface
Alternative interface using Gradio (lighter weight for HF Spaces)

Author: Deeksha Reddy Patlolla
University of Colorado Denver
"""

import gradio as gr
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from shapely.geometry import Polygon, Point, box
from typing import List, Tuple, Dict

# Global model (loaded once)
model = None

# Constants
PKLOT_REF_WIDTH = 1280
PKLOT_REF_HEIGHT = 720


def load_model():
    """Load YOLO model."""
    global model
    if model is None:
        model = YOLO('yolov8n.pt')
    return model


def create_sample_rois():
    """Create sample parking slot ROIs."""
    sample_slots = []
    
    # Row 1
    base_y = 200
    slot_width = 80
    slot_height = 120
    
    for i in range(8):
        x_start = 150 + i * (slot_width + 10)
        slot = [
            (x_start, base_y),
            (x_start + slot_width, base_y),
            (x_start + slot_width, base_y + slot_height),
            (x_start, base_y + slot_height)
        ]
        sample_slots.append(slot)
    
    # Row 2
    base_y = 380
    for i in range(8):
        x_start = 150 + i * (slot_width + 10)
        slot = [
            (x_start, base_y),
            (x_start + slot_width, base_y),
            (x_start + slot_width, base_y + slot_height),
            (x_start, base_y + slot_height)
        ]
        sample_slots.append(slot)
    
    # Row 3
    base_y = 550
    for i in range(6):
        x_start = 200 + i * (slot_width + 20)
        offset = 15
        slot = [
            (x_start + offset, base_y),
            (x_start + slot_width + offset, base_y),
            (x_start + slot_width, base_y + slot_height - 20),
            (x_start, base_y + slot_height - 20)
        ]
        sample_slots.append(slot)
    
    return sample_slots


def scale_polygons(polygons, width, height):
    """Scale ROI polygons to image dimensions."""
    scale_x = width / PKLOT_REF_WIDTH
    scale_y = height / PKLOT_REF_HEIGHT
    
    scaled = []
    for polygon in polygons:
        scaled_poly = [(int(x * scale_x), int(y * scale_y)) for x, y in polygon]
        scaled.append(scaled_poly)
    
    return scaled


def identify_dataset(shape, filename=""):
    """Identify dataset type."""
    height, width = shape[:2]
    filename_lower = filename.lower()
    
    if 'pklot' in filename_lower or 'ufpr' in filename_lower or 'pucpr' in filename_lower:
        return 'PKLot'
    if 'cnr' in filename_lower:
        return 'CNRPark'
    
    aspect_ratio = width / height
    if 1.7 < aspect_ratio < 1.8:
        return 'PKLot'
    
    return 'Unknown'


def detect_vehicles(image):
    """Run vehicle detection."""
    vehicle_classes = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
    
    model = load_model()
    results = model(image, verbose=False)
    
    detections = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id in vehicle_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                class_name = vehicle_classes[cls_id]
                detections.append((x1, y1, x2, y2, conf, class_name))
    
    return detections


def check_occupancy(polygon, detections, threshold=0.3):
    """Check if slot is occupied."""
    slot = Polygon(polygon)
    if not slot.is_valid:
        slot = slot.buffer(0)
    
    for det in detections:
        x1, y1, x2, y2 = det[:4]
        center = Point((x1 + x2) / 2, (y1 + y2) / 2)
        
        if slot.contains(center):
            return True
        
        vehicle = box(x1, y1, x2, y2)
        if slot.is_valid and vehicle.is_valid:
            intersection = slot.intersection(vehicle)
            if intersection.area > 0:
                iou = intersection.area / slot.area
                if iou > threshold:
                    return True
    
    return False


def draw_results(image, polygons, states, detections, show_boxes=True):
    """Draw visualization on image."""
    output = image.copy()
    
    # Draw slots
    for polygon, state in zip(polygons, states):
        pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
        
        if state == 'occupied':
            color = (0, 0, 255)
            fill = (0, 0, 200)
        else:
            color = (0, 255, 0)
            fill = (0, 200, 0)
        
        overlay = output.copy()
        cv2.fillPoly(overlay, [pts], fill)
        cv2.addWeighted(overlay, 0.3, output, 0.7, 0, output)
        cv2.polylines(output, [pts], True, color, 2)
    
    # Draw detections
    if show_boxes:
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(output, f"{cls} {conf:.2f}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return output


def process_image(input_image, show_bounding_boxes):
    """Main processing function for Gradio."""
    if input_image is None:
        return None, "Please upload an image"
    
    # Convert to BGR for OpenCV
    image = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
    height, width = image.shape[:2]
    
    # Identify dataset
    dataset = identify_dataset(image.shape)
    
    # Detect vehicles
    detections = detect_vehicles(image)
    
    if dataset == 'PKLot':
        # Use ROI-based classification
        base_polygons = create_sample_rois()
        polygons = scale_polygons(base_polygons, width, height)
        
        # Classify slots
        states = []
        for polygon in polygons:
            occupied = check_occupancy(polygon, detections)
            states.append('occupied' if occupied else 'empty')
        
        # Draw results
        output = draw_results(image, polygons, states, detections, show_bounding_boxes)
        
        # Statistics
        total = len(states)
        occupied = states.count('occupied')
        empty = states.count('empty')
        rate = (occupied / total * 100) if total > 0 else 0
        
        stats = f"""
### Detection Results

**Dataset Detected:** {dataset}

**Occupancy Summary:**
- Total Slots: {total}
- Occupied: {occupied} (Red)
- Empty: {empty} (Green)
- Occupancy Rate: {rate:.1f}%

**Vehicles Detected:** {len(detections)}
"""
    else:
        # Detection-only mode
        output = image.copy()
        
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(output, f"{cls} {conf:.2f}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add count text
        cv2.putText(output, f"Vehicles: {len(detections)}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        stats = f"""
### Detection Results

**Dataset Detected:** {dataset}
**Mode:** Detection Only (No ROI data)

**Vehicles Detected:** {len(detections)}

*Note: Upload PKLot images for slot-level occupancy classification.*
"""
    
    # Convert back to RGB
    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    
    return output_rgb, stats


# Create Gradio interface
with gr.Blocks(title="Smart Parking Detection") as demo:
    gr.Markdown("""
    # 🚗 Smart Parking Detection Using Computer Vision
    
    **Author:** Deeksha Reddy Patlolla | **Institution:** University of Colorado Denver
    
    Upload a parking lot image to detect vehicles and analyze occupancy.
    
    **Supported:**
    - **PKLot Dataset**: ROI-based slot classification
    - **CNRPark-EXT**: Detection-only mode
    - **Custom Images**: Detection-only mode
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                type="pil",
                label="Upload Parking Lot Image"
            )
            show_boxes = gr.Checkbox(
                value=True,
                label="Show Vehicle Bounding Boxes"
            )
            process_btn = gr.Button("🔍 Analyze Parking Lot", variant="primary")
        
        with gr.Column(scale=1):
            output_image = gr.Image(
                type="numpy",
                label="Detection Results"
            )
            output_stats = gr.Markdown(label="Statistics")
    
    # Examples
    gr.Markdown("---")
    gr.Markdown("### How to Use")
    gr.Markdown("""
    1. **Upload** a parking lot image (JPG, PNG)
    2. **Click** 'Analyze Parking Lot'
    3. **View** the results with color-coded slots:
       - 🟢 **Green** = Empty slot
       - 🔴 **Red** = Occupied slot
       - 🔵 **Blue boxes** = Detected vehicles
    """)
    
    # Connect processing
    process_btn.click(
        fn=process_image,
        inputs=[input_image, show_boxes],
        outputs=[output_image, output_stats]
    )
    
    # Also process on image upload
    input_image.change(
        fn=process_image,
        inputs=[input_image, show_boxes],
        outputs=[output_image, output_stats]
    )


if __name__ == "__main__":
    demo.launch()
