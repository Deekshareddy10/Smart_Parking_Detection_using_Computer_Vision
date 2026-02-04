"""
Smart Parking Detection Using Computer Vision
Author: Deeksha Reddy Patlolla
University of Colorado Denver - MS Computer Science

This application detects parking occupancy using YOLOv8 and ROI-based slot classification.
Supports PKLot and CNRPark-EXT datasets.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union
import xml.etree.ElementTree as ET
import os
import tempfile
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Smart Parking Detection",
    page_icon="🚗",
    layout="wide"
)

# Constants for PKLot reference resolution
PKLOT_REF_WIDTH = 1280
PKLOT_REF_HEIGHT = 720

# PKLot known resolutions for detection
PKLOT_RESOLUTIONS = [
    (1280, 720),
    (1920, 1080),
    (640, 480),
    (1024, 768),
]

# CNRPark typical resolutions
CNRPARK_RESOLUTIONS = [
    (1000, 750),
    (2592, 1944),
    (1920, 1080),
]


@st.cache_resource
def load_yolo_model():
    """Load YOLOv8 model with caching for performance."""
    try:
        model = YOLO('yolov8n.pt')  # Using nano model for speed on HF Spaces
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None


def detect_dataset_type(image, filename=""):
    """
    Detect whether image is from PKLot, CNRPark-EXT, or Unknown dataset.
    Uses resolution, filename patterns, and aspect ratio.
    """
    height, width = image.shape[:2]
    aspect_ratio = width / height
    filename_lower = filename.lower()
    
    # Check filename patterns
    if any(x in filename_lower for x in ['pklot', 'ufpr', 'pucpr']):
        return "PKLot"
    if any(x in filename_lower for x in ['cnrpark', 'cnr_park', 'cnr-park']):
        return "CNRPark-EXT"
    
    # Check resolution patterns
    for res in PKLOT_RESOLUTIONS:
        if abs(width - res[0]) < 50 and abs(height - res[1]) < 50:
            return "PKLot"
    
    for res in CNRPARK_RESOLUTIONS:
        if abs(width - res[0]) < 50 and abs(height - res[1]) < 50:
            return "CNRPark-EXT"
    
    # Check aspect ratio (PKLot is typically 16:9)
    if 1.7 < aspect_ratio < 1.85:
        return "PKLot"
    elif 1.3 < aspect_ratio < 1.4:
        return "CNRPark-EXT"
    
    return "Unknown"


def load_pklot_rois(xml_path, target_width, target_height):
    """
    Load and scale PKLot ROI polygons from XML file.
    Scales from reference resolution to target image size.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        slots = []
        scale_x = target_width / PKLOT_REF_WIDTH
        scale_y = target_height / PKLOT_REF_HEIGHT
        
        for space in root.findall('.//space'):
            slot_id = space.get('id', 'unknown')
            occupied = space.get('occupied', '0') == '1'
            
            # Get contour points
            contour = space.find('contour')
            if contour is not None:
                points = []
                for point in contour.findall('point'):
                    x = float(point.get('x', 0)) * scale_x
                    y = float(point.get('y', 0)) * scale_y
                    points.append((x, y))
                
                if len(points) >= 3:
                    slots.append({
                        'id': slot_id,
                        'polygon': Polygon(points),
                        'points': np.array(points, dtype=np.int32),
                        'ground_truth': occupied
                    })
        
        return slots
    except Exception as e:
        st.warning(f"Error loading ROI file: {e}")
        return []


def generate_default_pklot_rois(width, height):
    """
    Generate default PKLot-style ROIs for demonstration.
    Creates a grid of parking slots.
    """
    slots = []
    
    # Define parking area (central region of image)
    start_x = int(width * 0.1)
    end_x = int(width * 0.9)
    start_y = int(height * 0.2)
    end_y = int(height * 0.85)
    
    # Slot dimensions
    slot_width = int((end_x - start_x) / 8)
    slot_height = int((end_y - start_y) / 3)
    
    slot_id = 0
    for row in range(3):
        for col in range(8):
            x1 = start_x + col * slot_width
            y1 = start_y + row * slot_height
            x2 = x1 + slot_width - 5
            y2 = y1 + slot_height - 5
            
            points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            slots.append({
                'id': str(slot_id),
                'polygon': Polygon(points),
                'points': np.array(points, dtype=np.int32),
                'ground_truth': None
            })
            slot_id += 1
    
    return slots


def detect_vehicles(model, image, confidence_threshold=0.25):
    """
    Detect vehicles in image using YOLOv8.
    Returns list of bounding boxes and detection results.
    """
    # Run inference
    results = model(image, conf=confidence_threshold, verbose=False)
    
    detections = []
    
    # Vehicle class IDs in COCO dataset
    vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
    
    for result in results:
        boxes = result.boxes
        for i, box_item in enumerate(boxes):
            cls_id = int(box_item.cls[0])
            if cls_id in vehicle_classes:
                x1, y1, x2, y2 = box_item.xyxy[0].cpu().numpy()
                confidence = float(box_item.conf[0])
                class_name = result.names[cls_id]
                
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': confidence,
                    'class': class_name,
                    'center': ((x1 + x2) / 2, (y1 + y2) / 2)
                })
    
    return detections


def classify_slot_occupancy(slots, detections, iou_threshold=0.15):
    """
    Classify each parking slot as occupied or empty.
    Uses center-point heuristic with optional IoU check.
    """
    slot_states = []
    
    for slot in slots:
        slot_polygon = slot['polygon']
        is_occupied = False
        matching_vehicle = None
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            center_x, center_y = detection['center']
            
            # Primary check: center point inside polygon
            center_point = Point(center_x, center_y)
            if slot_polygon.contains(center_point):
                is_occupied = True
                matching_vehicle = detection
                break
            
            # Secondary check: IoU overlap
            vehicle_box = box(x1, y1, x2, y2)
            if slot_polygon.is_valid and vehicle_box.is_valid:
                intersection = slot_polygon.intersection(vehicle_box)
                if not intersection.is_empty:
                    iou = intersection.area / slot_polygon.area
                    if iou > iou_threshold:
                        is_occupied = True
                        matching_vehicle = detection
                        break
        
        slot_states.append({
            'slot': slot,
            'occupied': is_occupied,
            'vehicle': matching_vehicle
        })
    
    return slot_states


def draw_results(image, slot_states, detections, dataset_type):
    """
    Draw detection results on image.
    - Green polygons for empty slots
    - Red polygons for occupied slots
    - Blue bounding boxes for detected vehicles
    """
    result_image = image.copy()
    
    # Draw slot polygons (for PKLot)
    if dataset_type == "PKLot" and slot_states:
        for state in slot_states:
            points = state['slot']['points']
            color = (255, 0, 0) if state['occupied'] else (0, 255, 0)  # Red if occupied, Green if empty (RGB)
            
            # Draw filled polygon with transparency
            overlay = result_image.copy()
            cv2.fillPoly(overlay, [points], color)
            cv2.addWeighted(overlay, 0.3, result_image, 0.7, 0, result_image)
            
            # Draw polygon outline
            cv2.polylines(result_image, [points], True, color, 2)
            
            # Draw slot ID
            centroid = state['slot']['polygon'].centroid
            cv2.putText(result_image, state['slot']['id'], 
                       (int(centroid.x) - 10, int(centroid.y) + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Draw vehicle bounding boxes
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        label = f"{detection['class']} {detection['confidence']:.2f}"
        
        # Cyan bounding box for visibility
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
        
        # Label background
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(result_image, (x1, y1 - text_height - 5), (x1 + text_width, y1), (0, 255, 255), -1)
        cv2.putText(result_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return result_image


def main():
    """Main Streamlit application."""
    
    # Header
    st.title("🚗 Smart Parking Detection — PKLot ROI + YOLO")
    st.markdown("""
    Upload an image for parking detection. The system automatically detects the dataset type 
    and applies appropriate processing:
    - **PKLot**: ROI-based slot occupancy classification
    - **CNRPark-EXT**: Vehicle detection and counting
    - **Unknown**: Detection-only mode
    """)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    confidence_threshold = st.sidebar.slider("Detection Confidence", 0.1, 0.9, 0.25, 0.05)
    iou_threshold = st.sidebar.slider("IoU Threshold for Occupancy", 0.05, 0.5, 0.15, 0.05)
    show_debug = st.sidebar.checkbox("Show Debug Info", False)
    
    # Load model
    with st.spinner("Loading YOLOv8 model..."):
        model = load_yolo_model()
    
    if model is None:
        st.error("Failed to load YOLO model. Please refresh the page.")
        return
    
    st.sidebar.success("✅ Model loaded successfully")
    
    # File upload
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader(
        "Drag and drop file here",
        type=['jpg', 'jpeg', 'png'],
        help="Limit 200MB per file • JPG, JPEG, PNG"
    )
    
    # Optional ROI file upload
    st.sidebar.header("📁 ROI Configuration")
    roi_file = st.sidebar.file_uploader(
        "Upload PKLot ROI XML (optional)",
        type=['xml'],
        help="Upload PKLot XML file with slot annotations"
    )
    
    use_default_rois = st.sidebar.checkbox("Use default ROI grid (for demo)", True)
    
    if uploaded_file is not None:
        # Load image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        height, width = image.shape[:2]
        
        # Detect dataset type
        dataset_type = detect_dataset_type(image, uploaded_file.name)
        
        # Display dataset detection
        if dataset_type == "PKLot":
            st.success(f"**Detected dataset: PKLot** 📊")
        elif dataset_type == "CNRPark-EXT":
            st.info(f"**Detected dataset: CNRPark-EXT** 📊")
        else:
            st.warning(f"**Detected dataset: Unknown** — Using detection-only mode")
        
        if show_debug:
            st.write(f"Image dimensions: {width} x {height}")
            st.write(f"Aspect ratio: {width/height:.2f}")
        
        # Process based on dataset type
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image_rgb, use_container_width=True)
        
        # Run vehicle detection
        with st.spinner("Detecting vehicles..."):
            detections = detect_vehicles(model, image_rgb, confidence_threshold)
        
        # Process slots if PKLot
        slots = []
        slot_states = []
        
        if dataset_type == "PKLot":
            # Load ROIs
            if roi_file is not None:
                # Save uploaded XML temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.xml') as tmp:
                    tmp.write(roi_file.read())
                    tmp_path = tmp.name
                slots = load_pklot_rois(tmp_path, width, height)
                os.unlink(tmp_path)
            elif use_default_rois:
                slots = generate_default_pklot_rois(width, height)
            
            if slots:
                slot_states = classify_slot_occupancy(slots, detections, iou_threshold)
        
        # Draw results
        result_image = draw_results(image_rgb, slot_states, detections, dataset_type)
        
        with col2:
            st.subheader("Detection Results")
            st.image(result_image, use_container_width=True)
        
        # Statistics
        st.subheader("📊 Occupancy Summary")
        
        if dataset_type == "PKLot" and slot_states:
            total_slots = len(slot_states)
            occupied_slots = sum(1 for s in slot_states if s['occupied'])
            empty_slots = total_slots - occupied_slots
            
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Total Slots", total_slots)
            col_b.metric("Occupied", occupied_slots, delta=None)
            col_c.metric("Empty", empty_slots, delta=None)
            col_d.metric("Occupancy Rate", f"{(occupied_slots/total_slots)*100:.1f}%")
            
            # Progress bar
            st.progress(occupied_slots / total_slots if total_slots > 0 else 0)
            
            if show_debug:
                st.subheader("Slot States")
                slot_data = []
                for state in slot_states:
                    slot_data.append({
                        'Slot ID': state['slot']['id'],
                        'Status': 'Occupied' if state['occupied'] else 'Empty',
                        'Vehicle': state['vehicle']['class'] if state['vehicle'] else '-'
                    })
                st.dataframe(slot_data)
        else:
            # Detection-only statistics
            col_a, col_b = st.columns(2)
            col_a.metric("Vehicles Detected", len(detections))
            
            # Vehicle type breakdown
            vehicle_counts = {}
            for d in detections:
                vehicle_counts[d['class']] = vehicle_counts.get(d['class'], 0) + 1
            
            if vehicle_counts:
                col_b.write("**Vehicle Types:**")
                for vtype, count in vehicle_counts.items():
                    col_b.write(f"- {vtype}: {count}")
        
        # Detection details
        if show_debug and detections:
            st.subheader("Detection Details")
            det_data = []
            for i, d in enumerate(detections):
                det_data.append({
                    'ID': i,
                    'Class': d['class'],
                    'Confidence': f"{d['confidence']:.2f}",
                    'BBox': str(d['bbox']),
                    'Center': f"({d['center'][0]:.0f}, {d['center'][1]:.0f})"
                })
            st.dataframe(det_data)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Smart Parking Detection System** | Built with YOLOv8 + Streamlit  
    *University of Colorado Denver - MS Computer Science Project*  
    Author: Deeksha Reddy Patlolla
    """)


if __name__ == "__main__":
    main()
