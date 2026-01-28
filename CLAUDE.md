# CLAUDE.md - AI Assistant Guide for HKSAFaceBlend

## Project Overview

HKSAFaceBlend is a Python-based face morphing application that creates seamless face composites by blending facial features from two images. It uses MediaPipe for facial landmark detection and OpenCV for image processing.

## Codebase Structure

```
HKSAFaceBlend/
├── face_blender.py      # Core face morphing engine (main module)
├── batch_blend.py       # Batch processing orchestrator
├── requirements.txt     # Python dependencies
├── Faces.txt            # Configuration file with face pairs to blend
├── Faces/               # Input directory for face images (PNG format)
├── Results/             # Output directory for blended results
└── .gitignore           # Git ignore patterns
```

## Key Technologies

| Technology | Purpose | Min Version |
|------------|---------|-------------|
| Python | Core language | 3.x |
| MediaPipe | Facial landmark detection (468 points) | ≥0.10.0 |
| OpenCV | Image processing, morphing, transformations | ≥4.5.0 |
| NumPy | Numerical computing, array operations | ≥1.19.0 |
| Pillow | Image I/O, EXIF handling | ≥8.0.0 |

## Entry Points

### Single Pair Processing
```bash
python3 face_blender.py <image1> <image2> [options]

Options:
  -o, --output FILE    Output file (default: blended_result.jpg)
  -s, --size SIZE      Output size in pixels (default: 600)
  -b, --blend RATIO    Blend ratio 0.0-1.0 (default: 0.5)
```

### Batch Processing
```bash
python3 batch_blend.py
```
Reads pairs from `Faces.txt`, processes images from `Faces/`, outputs to `Results/`.

## Configuration Files

### Faces.txt Format
```
Name1, Name2
Name3, Name4
# Comments start with #
```
- Each line contains two names separated by `, `
- Names correspond to PNG files in `Faces/` directory (e.g., `Name1.png`)
- Lines starting with `#` are ignored

### requirements.txt
Standard pip requirements file. Install with:
```bash
pip install -r requirements.txt
```

## Architecture & Processing Pipeline

### Face Processing Steps (in order)
1. **EXIF Orientation Correction** - Handles rotated camera images
2. **Face Standardization** - Centers face, scales to ~70% of output size
3. **Face Angle Rectification** - Straightens based on eye positions (>0.5° threshold)
4. **Color Normalization** - Subtle white balance, gentle CLAHE contrast, brightness normalization
5. **Face Alignment** - Fine-tunes eye position to 38% from top
6. **Skin Tone Matching** - LAB color space matching between both faces
7. **Delaunay Triangulation** - Mesh generation on averaged landmarks
8. **Triangle Warping** - Affine transforms per triangle
9. **Multi-Tier Masking** - Separates face, hair, and background regions
10. **Final Compositing** - Hair from image 1, blended face features, white background

### Key Functions in face_blender.py

| Function | Purpose | Line |
|----------|---------|------|
| `fix_image_orientation()` | Load image with EXIF rotation fix | 30 |
| `preprocess_image()` | Full preprocessing pipeline | 224 |
| `get_landmarks()` | MediaPipe 468-point face detection | 245 |
| `align_face()` | Final alignment pass | 279 |
| `match_skin_tone()` | LAB color space skin matching | 315 |
| `morph_faces()` | Core morphing algorithm | 525 |
| `process_images()` | Main processing orchestrator | 613 |

### MediaPipe Landmarks
- 468 facial landmarks per face
- Key indices defined in `MORPH_POINTS` (face_blender.py:339)
- Face oval indices in `FACE_OVAL` (face_blender.py:332)

## Code Conventions

### Style
- Uses standard Python conventions (PEP 8)
- Docstrings on major functions
- Inline comments for complex algorithms
- Constants in UPPER_SNAKE_CASE at module level

### Configuration Constants (face_blender.py:23-27)
```python
DEFAULT_IMAGE_1 = "person_a.jpg"
DEFAULT_IMAGE_2 = "person_b.jpg"
OUTPUT_FILENAME = "blended_result.jpg"
OUTPUT_SIZE = (600, 600)
BLEND_RATIO = 0.5
```

### Error Handling
- Functions return `None` or `False` on failure
- Batch processor continues on individual failures
- Graceful degradation with try/except blocks

### Image Processing Conventions
- Images use BGR color order (OpenCV standard)
- LAB color space for perceptually accurate operations
- HSV for brightness adjustments
- Masks use 0-255 grayscale values

## Common Development Tasks

### Adding New Face Pairs
1. Add PNG images to `Faces/` directory
2. Add pair entry to `Faces.txt`: `Name1, Name2`
3. Run `python3 batch_blend.py`

### Modifying Blend Behavior
- Adjust `BLEND_RATIO` constant for default blend
- Use `-b` flag for one-off adjustments
- Blend ratio 0.0 = mostly image 1, 1.0 = mostly image 2

### Adjusting Output Quality
- Output size: modify `OUTPUT_SIZE` or use `-s` flag
- JPEG quality: change value in `cv2.imwrite()` call (line 677)
- PNG output: batch processor uses PNG format automatically

### Testing Face Detection
```python
from face_blender import get_landmarks, fix_image_orientation
img = fix_image_orientation("path/to/image.png")
landmarks = get_landmarks(img)
if landmarks is None:
    print("No face detected")
```

## Dependencies Installation

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Output Specifications

- **Default size**: 600x600 pixels
- **JPEG quality**: 95%
- **Background**: White (RGB 255, 255, 255)
- **Naming convention (batch)**: `{Name1}+{Name2}.png`

## Troubleshooting

### "Could not detect face"
- Ensure face is clearly visible and well-lit
- Face should be roughly frontal (not extreme profile)
- Image should have sufficient resolution

### Memory issues
- Large images are scaled down during preprocessing
- Batch processing handles images sequentially

### Color artifacts
- Skin tone matching uses LAB color space
- Bilateral filtering smooths minor artifacts
- Adjust white balance strength in `subtle_white_balance()` if needed

## Important Notes for AI Assistants

1. **Image format**: Input images in `Faces/` must be PNG format
2. **Face detection**: MediaPipe requires frontal or near-frontal faces
3. **Blend ratio**: 0.5 means equal contribution from both faces
4. **Hair handling**: Hair is always taken from image 1 (first in pair)
5. **Background**: Final output always has white background
6. **Error tolerance**: Batch processing continues even if individual pairs fail
