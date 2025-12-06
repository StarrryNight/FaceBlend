#!/usr/bin/env python3
"""
Face Blender - Face Morphing Tool
Creates seamless face composites from two images.
"""

import cv2
import numpy as np
import argparse
import sys
import os
from pathlib import Path

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================
DEFAULT_IMAGE_1 = "person_a.jpg"
DEFAULT_IMAGE_2 = "person_b.jpg"
OUTPUT_FILENAME = "blended_result.jpg"
OUTPUT_SIZE = (600, 600)
BLEND_RATIO = 0.5


def fix_image_orientation(image_path):
    """Load image and fix EXIF orientation."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
        
        pil_img = Image.open(image_path)
        exif = pil_img._getexif()
        
        if exif:
            for tag_id, value in exif.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == 'Orientation':
                    if value == 3:
                        img = cv2.rotate(img, cv2.ROTATE_180)
                    elif value == 6:
                        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                    elif value == 8:
                        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    break
    except Exception:
        pass
    
    return img


def subtle_white_balance(img, strength=0.3):
    """Apply subtle white balance correction - preserves skin tones."""
    result = img.copy().astype(np.float32)
    original = result.copy()
    
    # Calculate mean of each channel
    avg_b = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_r = np.mean(result[:, :, 2])
    
    # Calculate gray average
    avg_gray = (avg_b + avg_g + avg_r) / 3
    
    # Scale each channel
    if avg_b > 0:
        result[:, :, 0] *= avg_gray / avg_b
    if avg_g > 0:
        result[:, :, 1] *= avg_gray / avg_g
    if avg_r > 0:
        result[:, :, 2] *= avg_gray / avg_r
    
    result = np.clip(result, 0, 255)
    
    # Blend with original to preserve skin tones
    result = original * (1 - strength) + result * strength
    
    return result.astype(np.uint8)


def gentle_contrast(img):
    """Apply gentle contrast enhancement - less aggressive CLAHE."""
    # Convert to LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Very gentle CLAHE
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    # Blend with original L channel (50% strength)
    l = cv2.addWeighted(l, 0.5, l_enhanced, 0.5, 0)
    
    # Merge and convert back
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def normalize_brightness(img, target_brightness=135):
    """Gently normalize brightness without washing out colors."""
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # Get current brightness
    current_brightness = np.mean(hsv[:, :, 2])
    
    # Only adjust if significantly different (more than 20%)
    if abs(current_brightness - target_brightness) > target_brightness * 0.2:
        # Gentle scaling - move 50% toward target
        scale = 1.0 + (target_brightness / current_brightness - 1.0) * 0.5
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * scale, 0, 255)
    
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def get_face_bounds(landmarks):
    """Get face bounding box and key measurements from landmarks."""
    # Face oval indices
    oval_idx = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 
                397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 
                172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    
    oval_pts = np.array([landmarks[i] for i in oval_idx])
    
    x_min, y_min = oval_pts.min(axis=0)
    x_max, y_max = oval_pts.max(axis=0)
    
    width = x_max - x_min
    height = y_max - y_min
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    
    return center_x, center_y, width, height


def standardize_face(img, landmarks, target_size=500):
    """
    Standardize face position and size:
    - Center face in frame
    - Scale to consistent size based on face dimensions
    - Output square image with face centered
    """
    h, w = img.shape[:2]
    
    # Get face measurements
    face_cx, face_cy, face_w, face_h = get_face_bounds(landmarks)
    face_size = max(face_w, face_h)
    
    # Calculate scale to make face fill ~70% of target size
    desired_face_size = target_size * 0.7
    scale = desired_face_size / face_size if face_size > 0 else 1.0
    
    # Create transformation: scale + translate to center
    # First scale around face center, then translate to image center
    M = np.zeros((2, 3), dtype=np.float32)
    M[0, 0] = scale  # scale x
    M[1, 1] = scale  # scale y
    
    # New face center after scaling
    new_face_cx = face_cx * scale
    new_face_cy = face_cy * scale
    
    # Translate to center face in target
    target_center = target_size / 2
    M[0, 2] = target_center - new_face_cx
    M[1, 2] = target_center - new_face_cy
    
    # Apply transformation
    standardized = cv2.warpAffine(img, M, (target_size, target_size),
                                   borderMode=cv2.BORDER_REPLICATE)
    
    # Transform landmarks
    ones = np.ones((len(landmarks), 1))
    lm_h = np.hstack([landmarks, ones])
    new_landmarks = M.dot(lm_h.T).T
    
    return standardized, new_landmarks


def rectify_face_angle(img, landmarks):
    """Straighten face based on eye positions."""
    left_eye, right_eye = get_eye_centers(landmarks)
    
    # Calculate rotation angle
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    
    # Only rectify if angle is significant
    if abs(angle) < 0.5:
        return img, landmarks
    
    # Get image center
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    
    # Create rotation matrix (no scaling)
    M = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), float(angle), 1.0)
    
    # For square images, keep same size
    new_w, new_h = w, h
    
    # Rotate image
    rotated = cv2.warpAffine(img, M, (new_w, new_h), 
                              borderMode=cv2.BORDER_REPLICATE)
    
    # Transform landmarks
    ones = np.ones((len(landmarks), 1))
    lm_h = np.hstack([landmarks, ones])
    new_landmarks = M.dot(lm_h.T).T
    
    return rotated, new_landmarks


def preprocess_image(img, landmarks, standard_size=600):
    """
    Full preprocessing pipeline:
    1. Standardize face size and center position
    2. Rectify face angle (straighten)
    3. Subtle color corrections (preserve skin tones)
    """
    # First: standardize size and center the face
    img, landmarks = standardize_face(img, landmarks, target_size=standard_size)
    
    # Second: straighten based on eye positions
    img, landmarks = rectify_face_angle(img, landmarks)
    
    # Third: gentle color corrections - preserve natural skin tones
    img = subtle_white_balance(img, strength=0.2)
    img = gentle_contrast(img)
    img = normalize_brightness(img, target_brightness=140)
    
    return img, landmarks


def get_landmarks(image):
    """Get 468 facial landmarks using MediaPipe."""
    mp_face_mesh = mp.solutions.face_mesh
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        
        if not results.multi_face_landmarks:
            return None
        
        h, w = image.shape[:2]
        landmarks = []
        for lm in results.multi_face_landmarks[0].landmark:
            landmarks.append((lm.x * w, lm.y * h))
        
        return np.array(landmarks, dtype=np.float32)


def get_eye_centers(landmarks):
    """Get eye center positions."""
    left_eye_idx = [33, 133, 160, 159, 158, 144, 145, 153]
    right_eye_idx = [362, 263, 387, 386, 385, 373, 374, 380]
    
    left = np.mean([landmarks[i] for i in left_eye_idx], axis=0)
    right = np.mean([landmarks[i] for i in right_eye_idx], axis=0)
    return left, right


def align_face(image, landmarks, output_size):
    """
    Final alignment pass - fine-tune position after preprocessing.
    Since preprocessing already standardized size/center, this just ensures
    consistent eye position across both faces.
    """
    h, w = image.shape[:2]
    left_eye, right_eye = get_eye_centers(landmarks)
    
    # Eye center (midpoint)
    eye_center_x = (left_eye[0] + right_eye[0]) / 2
    eye_center_y = (left_eye[1] + right_eye[1]) / 2
    
    # Desired eye position (centered horizontally, 38% from top)
    target_x = output_size[0] / 2
    target_y = output_size[1] * 0.38
    
    # Simple translation to align eye centers
    tx = target_x - eye_center_x
    ty = target_y - eye_center_y
    
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    
    # Warp
    aligned = cv2.warpAffine(image, M, output_size,
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    
    # Transform landmarks
    new_landmarks = landmarks.copy()
    new_landmarks[:, 0] += tx
    new_landmarks[:, 1] += ty
    
    return aligned, new_landmarks


def match_skin_tone(source, target):
    """Match source skin tones to target using LAB color space."""
    src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    tgt_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    for i in range(3):
        src_mean, src_std = src_lab[:,:,i].mean(), src_lab[:,:,i].std()
        tgt_mean, tgt_std = tgt_lab[:,:,i].mean(), tgt_lab[:,:,i].std()
        
        if src_std > 0:
            src_lab[:,:,i] = (src_lab[:,:,i] - src_mean) * (tgt_std / src_std) + tgt_mean
    
    src_lab = np.clip(src_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(src_lab, cv2.COLOR_LAB2BGR)


# Face oval indices for masking
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
    378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109
]

# Key points for morphing
MORPH_POINTS = [
    # Face outline
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
    378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109,
    # Eyebrows
    70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
    300, 293, 334, 296, 336, 285, 295, 282, 283, 276,
    # Eyes
    33, 133, 160, 159, 158, 144, 145, 153, 154, 155, 157, 173, 246,
    362, 263, 387, 386, 385, 373, 374, 380, 381, 382, 384, 398,
    # Nose
    6, 197, 195, 5, 4, 1, 19, 94, 2, 164,
    # Mouth
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
]


def get_triangulation(points, w, h):
    """Get Delaunay triangulation indices."""
    rect = (0, 0, w, h)
    subdiv = cv2.Subdiv2D(rect)
    
    # Insert points (clamped to bounds)
    for p in points:
        x = max(0, min(w-1, p[0]))
        y = max(0, min(h-1, p[1]))
        try:
            subdiv.insert((float(x), float(y)))
        except:
            pass
    
    triangles = subdiv.getTriangleList()
    indices = []
    
    for t in triangles:
        tri_pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        idx = []
        for tp in tri_pts:
            for i, p in enumerate(points):
                if abs(tp[0] - p[0]) < 2 and abs(tp[1] - p[1]) < 2:
                    idx.append(i)
                    break
        if len(idx) == 3 and len(set(idx)) == 3:
            indices.append(idx)
    
    return indices


def warp_triangle(src, dst, t_src, t_dst):
    """Warp a triangle from src to dst."""
    r_src = cv2.boundingRect(np.float32([t_src]))
    r_dst = cv2.boundingRect(np.float32([t_dst]))
    
    # Offset triangles
    t_src_off = [(t_src[i][0] - r_src[0], t_src[i][1] - r_src[1]) for i in range(3)]
    t_dst_off = [(t_dst[i][0] - r_dst[0], t_dst[i][1] - r_dst[1]) for i in range(3)]
    
    # Get source patch
    patch = src[r_src[1]:r_src[1]+r_src[3], r_src[0]:r_src[0]+r_src[2]]
    if patch.size == 0 or r_dst[2] <= 0 or r_dst[3] <= 0:
        return
    
    # Warp
    M = cv2.getAffineTransform(np.float32(t_src_off), np.float32(t_dst_off))
    warped = cv2.warpAffine(patch, M, (r_dst[2], r_dst[3]), 
                            borderMode=cv2.BORDER_REFLECT_101)
    
    # Mask
    mask = np.zeros((r_dst[3], r_dst[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t_dst_off), (1, 1, 1), 16)
    
    # Blend into destination
    y1, y2 = r_dst[1], r_dst[1] + r_dst[3]
    x1, x2 = r_dst[0], r_dst[0] + r_dst[2]
    
    if y2 <= dst.shape[0] and x2 <= dst.shape[1] and y1 >= 0 and x1 >= 0:
        region = dst[y1:y2, x1:x2].astype(np.float32)
        warped = warped.astype(np.float32)
        result = region * (1 - mask) + warped * mask
        dst[y1:y2, x1:x2] = result.astype(np.uint8)


def create_face_mask(landmarks, size, expand=20):
    """Create a face+hair mask from face oval landmarks."""
    w, h = size
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Get face oval points
    oval_pts = np.array([landmarks[i] for i in FACE_OVAL if i < len(landmarks)], dtype=np.float32)
    
    # Expand top points upward for hair
    top_indices = [0, 1, 2, 3, 4, 31, 32, 33, 34, 35]  # Top portion of oval
    for i in top_indices:
        if i < len(oval_pts):
            oval_pts[i][1] -= h * 0.12  # Move up for hair
    
    # Expand sides for hair width
    center_x = w / 2
    for i in range(len(oval_pts)):
        if oval_pts[i][0] < center_x:
            oval_pts[i][0] -= w * 0.03
        else:
            oval_pts[i][0] += w * 0.03
    
    oval_pts = np.clip(oval_pts, 0, [w-1, h-1]).astype(np.int32)
    
    # Fill the face region
    cv2.fillConvexPoly(mask, oval_pts, 255)
    
    # Expansion for hair
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand, expand))
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Anti-aliasing
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    
    return mask


def create_face_only_mask(landmarks, size):
    """Create mask covering ONLY the face (no hair) - for blending."""
    w, h = size
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Face oval - tight to face, below hairline
    face_oval_idx = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
    ]
    
    pts = np.array([landmarks[i] for i in face_oval_idx if i < len(landmarks)], dtype=np.int32)
    cv2.fillConvexPoly(mask, pts, 255)
    
    # Small expansion
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Smooth edges for blending
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    
    return mask


def create_full_head_mask(landmarks, size):
    """Create mask covering face AND hair area - for final output."""
    w, h = size
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Face oval points
    face_oval_idx = [
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
    ]
    
    pts = np.array([landmarks[i] for i in face_oval_idx if i < len(landmarks)], dtype=np.float32)
    
    # Expand upward for hair
    for i, idx in enumerate(face_oval_idx):
        if idx in [10, 338, 297, 109, 67, 103, 54, 21, 162, 127]:
            pts[i][1] -= h * 0.18
    
    # Expand sides for hair
    center_x = w / 2
    for i in range(len(pts)):
        if pts[i][0] < center_x:
            pts[i][0] -= w * 0.08
        else:
            pts[i][0] += w * 0.08
    
    pts = np.clip(pts, 0, [w-1, h-1]).astype(np.int32)
    cv2.fillConvexPoly(mask, pts, 255)
    
    # Expand for full hair coverage
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Smooth edges
    mask = cv2.GaussianBlur(mask, (11, 11), 0)
    
    return mask


def morph_faces(img1, img2, lm1, lm2, alpha, size):
    """
    Morph faces with:
    - Hair from image 1
    - Background from image 2
    - Blended face features
    """
    w, h = size
    
    # Create face-only mask (excludes hair)
    face_mask1 = create_face_only_mask(lm1, size)
    face_mask2 = create_face_only_mask(lm2, size)
    
    # Create hair mask (full head minus face)
    full_head_mask1 = create_full_head_mask(lm1, size)
    hair_mask = cv2.subtract(full_head_mask1, face_mask1)
    hair_mask = cv2.GaussianBlur(hair_mask, (21, 21), 0)
    
    # Get subset of landmarks
    pts1 = np.array([lm1[i] for i in MORPH_POINTS if i < len(lm1)], dtype=np.float32)
    pts2 = np.array([lm2[i] for i in MORPH_POINTS if i < len(lm2)], dtype=np.float32)
    
    # Add corner/edge points
    corners = np.array([
        [0, 0], [w/2, 0], [w-1, 0],
        [0, h/2], [w-1, h/2],
        [0, h-1], [w/2, h-1], [w-1, h-1]
    ], dtype=np.float32)
    
    pts1 = np.vstack([pts1, corners])
    pts2 = np.vstack([pts2, corners])
    
    # Compute average shape
    pts_avg = (1 - alpha) * pts1 + alpha * pts2
    
    # Triangulate
    triangles = get_triangulation(pts_avg.tolist(), w, h)
    
    # Warp both images to average shape
    warped1 = np.zeros((h, w, 3), dtype=np.uint8)
    warped2 = np.zeros((h, w, 3), dtype=np.uint8)
    
    for tri in triangles:
        i, j, k = tri
        
        t1 = [pts1[i], pts1[j], pts1[k]]
        t2 = [pts2[i], pts2[j], pts2[k]]
        t_avg = [pts_avg[i], pts_avg[j], pts_avg[k]]
        
        warp_triangle(img1, warped1, t1, t_avg)
        warp_triangle(img2, warped2, t2, t_avg)
    
    # Start with background from image 2
    result = warped2.copy()
    
    # Add hair from image 1
    hair_mask_3ch = (hair_mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
    result = (warped1.astype(np.float32) * hair_mask_3ch + 
              result.astype(np.float32) * (1 - hair_mask_3ch))
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # Blend faces
    blend_mask = cv2.addWeighted(face_mask1, 0.5, face_mask2, 0.5, 0)
    blend_mask = cv2.GaussianBlur(blend_mask, (31, 31), 0)
    blend_mask_3ch = (blend_mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
    
    morphed_face = cv2.addWeighted(warped1, 1 - alpha, warped2, alpha, 0)
    
    # Composite blended face
    result = (morphed_face.astype(np.float32) * blend_mask_3ch + 
              result.astype(np.float32) * (1 - blend_mask_3ch))
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # Create full output mask (face + hair from image 1)
    full_mask = create_full_head_mask(lm1, size)
    _, full_mask = cv2.threshold(full_mask, 100, 255, cv2.THRESH_BINARY)
    full_mask = cv2.GaussianBlur(full_mask, (7, 7), 0)
    full_mask_3ch = (full_mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
    
    # White background - no image background
    white_bg = np.full((h, w, 3), 255, dtype=np.uint8)
    
    final = (result.astype(np.float32) * full_mask_3ch + 
             white_bg.astype(np.float32) * (1 - full_mask_3ch))
    
    return np.clip(final, 0, 255).astype(np.uint8)


def process_images(image1_path, image2_path, output_path, output_size, blend_ratio):
    """Main processing."""
    print(f"\n{'='*60}")
    print("FACE BLENDER")
    print(f"{'='*60}\n")
    
    if not MEDIAPIPE_AVAILABLE:
        print("ERROR: pip install mediapipe")
        return False
    
    if not os.path.exists(image1_path) or not os.path.exists(image2_path):
        print("ERROR: Image file not found")
        return False
    
    print(f"Loading images...")
    img1 = fix_image_orientation(image1_path)
    img2 = fix_image_orientation(image2_path)
    
    if img1 is None or img2 is None:
        print("ERROR: Could not load images")
        return False
    
    print(f"  Image 1: {img1.shape[1]}x{img1.shape[0]}")
    print(f"  Image 2: {img2.shape[1]}x{img2.shape[0]}")
    
    print(f"\nDetecting faces...")
    lm1 = get_landmarks(img1)
    lm2 = get_landmarks(img2)
    
    if lm1 is None or lm2 is None:
        print("ERROR: Could not detect face in one or both images")
        return False
    
    print(f"  Found 468 landmarks in each face")
    
    print(f"\nPreprocessing images...")
    print(f"  - Centering faces")
    print(f"  - Standardizing face sizes")
    print(f"  - Rectifying angles")
    print(f"  - Subtle color correction")
    img1, lm1 = preprocess_image(img1, lm1, standard_size=output_size[0])
    img2, lm2 = preprocess_image(img2, lm2, standard_size=output_size[0])
    
    print(f"\nAligning faces...")
    aligned1, lm1_aligned = align_face(img1, lm1, output_size)
    aligned2, lm2_aligned = align_face(img2, lm2, output_size)
    
    print(f"\nMatching skin tones...")
    # Match both toward each other based on blend ratio
    matched1 = match_skin_tone(aligned1, aligned2)
    matched2 = match_skin_tone(aligned2, aligned1)
    
    # Blend matched versions with originals
    aligned1 = cv2.addWeighted(aligned1, 1-blend_ratio, matched1, blend_ratio, 0)
    aligned2 = cv2.addWeighted(aligned2, blend_ratio, matched2, 1-blend_ratio, 0)
    
    print(f"\nMorphing faces (blend: {blend_ratio:.0%} / {1-blend_ratio:.0%})...")
    result = morph_faces(aligned1, aligned2, lm1_aligned, lm2_aligned, 
                         blend_ratio, output_size)
    
    # Smooth out artifacts
    result = cv2.bilateralFilter(result, 5, 40, 40)
    
    print(f"\nSaving to: {output_path}")
    cv2.imwrite(output_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    print(f"\n{'='*60}")
    print("SUCCESS!")
    print(f"{'='*60}\n")
    return True


def main():
    parser = argparse.ArgumentParser(description='Morph two faces together')
    parser.add_argument('image1', nargs='?', default=DEFAULT_IMAGE_1)
    parser.add_argument('image2', nargs='?', default=DEFAULT_IMAGE_2)
    parser.add_argument('-o', '--output', default=OUTPUT_FILENAME)
    parser.add_argument('-s', '--size', type=int, default=OUTPUT_SIZE[0])
    parser.add_argument('-b', '--blend', type=float, default=BLEND_RATIO)
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent.resolve()
    
    img1 = Path(args.image1)
    if not img1.is_absolute():
        img1 = script_dir / img1
    
    img2 = Path(args.image2)
    if not img2.is_absolute():
        img2 = script_dir / img2
    
    out = Path(args.output)
    if not out.is_absolute():
        out = script_dir / out
    
    success = process_images(
        str(img1), str(img2), str(out),
        (args.size, args.size),
        max(0.0, min(1.0, args.blend))
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
