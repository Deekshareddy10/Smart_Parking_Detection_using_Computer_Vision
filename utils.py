"""
Utility functions for Smart Parking Detection System
"""

import cv2
import numpy as np
from shapely.geometry import Polygon, Point, box
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Optional


# Constants
PKLOT_REF_WIDTH = 1280
PKLOT_REF_HEIGHT = 720

PKLOT_RESOLUTIONS = [
    (1280, 720),
    (1920, 1080),
    (640, 480),
    (1024, 768),
]

CNRPARK_RESOLUTIONS = [
    (1000, 750),
    (2592, 1944),
    (1920, 1080),
]

# COCO vehicle class IDs
VEHICLE_CLASSES = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}


def detect_dataset_type(image: np.ndarray, filename: str = "") -> str:
    """
    Detect whether image is from PKLot, CNRPark-EXT, or Unknown dataset.
    
    Args:
        image: Input image as numpy array
        filename: Original filename for pattern matching
        
    Returns:
        Dataset type string: "PKLot", "CNRPark-EXT", or "Unknown"
    """
    height, width = image.shape[:2]
    aspect_ratio = width / height
    filename_lower = filename.lower()
    
    # Check filename patterns
    pklot_patterns = ['pklot', 'ufpr', 'pucpr', 'ufpr04', 'ufpr05']
    cnrpark_patterns = ['cnrpark', 'cnr_park', 'cnr-park', 'camera']
    
    if any(x in filename_lower for x in pklot_patterns):
        return "PKLot"
    if any(x in filename_lower for x in cnrpark_patterns):
        return "CNRPark-EXT"
    
    # Check resolution patterns
    for res in PKLOT_RESOLUTIONS:
        if abs(width - res[0]) < 50 and abs(height - res[1]) < 50:
            return "PKLot"
    
    for res in CNRPARK_RESOLUTIONS:
        if abs(width - res[0]) < 50 and abs(height - res[1]) < 50:
            return "CNRPark-EXT"
    
    # Check aspect ratio
    if 1.7 < aspect_ratio < 1.85:  # 16:9 typical for PKLot
        return "PKLot"
    elif 1.3 < aspect_ratio < 1.4:  # 4:3 typical for CNRPark
        return "CNRPark-EXT"
    
    return "Unknown"


def load_pklot_rois_from_xml(xml_path: str, target_width: int, target_height: int) -> List[Dict]:
    """
    Load and scale PKLot ROI polygons from XML file.
    
    Args:
        xml_path: Path to PKLot XML annotation file
        target_width: Target image width for scaling
        target_height: Target image height for scaling
        
    Returns:
        List of slot dictionaries with polygon info
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
        print(f"Error loading ROI file: {e}")
        return []


def generate_grid_rois(
    width: int, 
    height: int, 
    rows: int = 3, 
    cols: int = 6,
    margin: float = 0.1,
    gap: int = 5
) -> List[Dict]:
    """
    Generate a grid of parking slot ROIs for demonstration.
    
    Args:
        width: Image width
        height: Image height
        rows: Number of rows in grid
        cols: Number of columns in grid
        margin: Margin from image edges (as fraction)
        gap: Gap between slots in pixels
        
    Returns:
        List of slot dictionaries
    """
    slots = []
    
    start_x = int(width * margin)
    end_x = int(width * (1 - margin))
    start_y = int(height * (margin + 0.1))
    end_y = int(height * (1 - margin))
    
    slot_width = int((end_x - start_x - (cols - 1) * gap) / cols)
    slot_height = int((end_y - start_y - (rows - 1) * gap) / rows)
    
    slot_id = 0
    for row in range(rows):
        for col in range(cols):
            x1 = start_x + col * (slot_width + gap)
            y1 = start_y + row * (slot_height + gap)
            x2 = x1 + slot_width
            y2 = y1 + slot_height
            
            points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            slots.append({
                'id': str(slot_id),
                'polygon': Polygon(points),
                'points': np.array(points, dtype=np.int32),
                'ground_truth': None
            })
            slot_id += 1
    
    return slots


def classify_slot_occupancy(
    slots: List[Dict], 
    detections: List[Dict], 
    iou_threshold: float = 0.15,
    use_center_point: bool = True
) -> List[Dict]:
    """
    Classify each parking slot as occupied or empty.
    
    Args:
        slots: List of slot dictionaries
        detections: List of vehicle detections
        iou_threshold: Minimum IoU for occupancy classification
        use_center_point: Use center-point heuristic as primary check
        
    Returns:
        List of slot state dictionaries
    """
    slot_states = []
    
    for slot in slots:
        slot_polygon = slot['polygon']
        is_occupied = False
        matching_vehicle = None
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            center_x, center_y = detection['center']
            
            # Primary: center point check
            if use_center_point:
                center_point = Point(center_x, center_y)
                if slot_polygon.contains(center_point):
                    is_occupied = True
                    matching_vehicle = detection
                    break
            
            # Secondary: IoU check
            vehicle_box = box(x1, y1, x2, y2)
            if slot_polygon.is_valid and vehicle_box.is_valid:
                try:
                    intersection = slot_polygon.intersection(vehicle_box)
                    if not intersection.is_empty:
                        iou = intersection.area / slot_polygon.area
                        if iou > iou_threshold:
                            is_occupied = True
                            matching_vehicle = detection
                            break
                except:
                    pass
        
        slot_states.append({
            'slot': slot,
            'occupied': is_occupied,
            'vehicle': matching_vehicle
        })
    
    return slot_states


def draw_detections(
    image: np.ndarray, 
    detections: List[Dict],
    color: Tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw vehicle detection bounding boxes on image.
    
    Args:
        image: Input image (RGB)
        detections: List of detection dictionaries
        color: Box color in RGB
        thickness: Line thickness
        
    Returns:
        Image with drawn detections
    """
    result = image.copy()
    
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        label = f"{detection['class']} {detection['confidence']:.2f}"
        
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
        
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            result, 
            (x1, y1 - text_height - 5), 
            (x1 + text_width, y1), 
            color, 
            -1
        )
        cv2.putText(
            result, label, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
        )
    
    return result


def draw_slot_occupancy(
    image: np.ndarray,
    slot_states: List[Dict],
    occupied_color: Tuple[int, int, int] = (255, 0, 0),
    empty_color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.3
) -> np.ndarray:
    """
    Draw slot occupancy overlays on image.
    
    Args:
        image: Input image (RGB)
        slot_states: List of slot state dictionaries
        occupied_color: Color for occupied slots (RGB)
        empty_color: Color for empty slots (RGB)
        alpha: Transparency for fill
        
    Returns:
        Image with slot overlays
    """
    result = image.copy()
    
    for state in slot_states:
        points = state['slot']['points']
        color = occupied_color if state['occupied'] else empty_color
        
        # Draw filled polygon with transparency
        overlay = result.copy()
        cv2.fillPoly(overlay, [points], color)
        cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)
        
        # Draw outline
        cv2.polylines(result, [points], True, color, 2)
        
        # Draw slot ID
        try:
            centroid = state['slot']['polygon'].centroid
            cv2.putText(
                result, state['slot']['id'],
                (int(centroid.x) - 10, int(centroid.y) + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1
            )
        except:
            pass
    
    return result


def compute_metrics(slot_states: List[Dict]) -> Dict:
    """
    Compute occupancy statistics.
    
    Args:
        slot_states: List of slot state dictionaries
        
    Returns:
        Dictionary with occupancy metrics
    """
    total = len(slot_states)
    occupied = sum(1 for s in slot_states if s['occupied'])
    empty = total - occupied
    
    return {
        'total_slots': total,
        'occupied': occupied,
        'empty': empty,
        'occupancy_rate': (occupied / total * 100) if total > 0 else 0
    }
