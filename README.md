# HKSAFaceBlend

Face morphing tool that creates seamless face composites by blending facial features from two images using MediaPipe landmark detection.

## Installation

```bash
git clone https://github.com/USERNAME/HKSAFaceBlend.git
cd HKSAFaceBlend
pip install -r requirements.txt
```

Or install directly via pip:

```bash
pip install git+https://github.com/USERNAME/HKSAFaceBlend.git
```

## Usage

### Single Pair

```bash
python face_blender.py image1.png image2.png -o output.jpg
```

### Batch Processing

1. Add face images to the `Faces/` directory (PNG format)
2. Edit `Faces.txt` with pairs to blend:
   ```
   Name1, Name2
   Name3, Name4
   ```
3. Run batch processing:
   ```bash
   python batch_blend.py
   ```

Results are saved to the `Results/` directory.

## CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `-o`, `--output` | Output filename | `blended_result.jpg` |
| `-s`, `--size` | Output size in pixels | `600` |
| `-b`, `--blend` | Blend ratio (0.0-1.0) | `0.5` |

### Blend Ratio

- `0.0` = Features mostly from image 1
- `0.5` = Equal blend from both images
- `1.0` = Features mostly from image 2

## Requirements

- Python 3.8+
- Frontal or near-frontal face photos
- Clear, well-lit images

## Limitations

- Requires faces to be clearly visible and roughly frontal
- Extreme profile angles may not be detected
- Hair is always taken from the first image in a pair

## License

MIT License - see [LICENSE](LICENSE) for details.
