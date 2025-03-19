# StegoDCT

A command-line tool for embedding and extracting textual messages in images using DCT (Discrete Cosine Transform) watermarking technique.

StegoDCT imperceptibly inscribes textual data within the frequency domain of JPEG and PNG images through manipulation of discrete cosine transform coefficients.

## Features

- Hides text messages within images without perceptible quality loss
- Works with both JPEG and PNG images
- Strong resistance against visual detection
- Survives moderate image processing operations
- Simple command-line interface

## Installation

### Prerequisites

StegoDCT requires Python 3.6+ and the following dependencies:
- NumPy
- OpenCV (cv2)
- Pillow (PIL)

### Installation Steps

1. Clone this repository or download the script

2. Install the required dependencies:

```bash
pip install numpy opencv-python pillow
```

### Architecture-specific Issues

If you encounter architecture-related errors with NumPy or OpenCV (especially on Apple Silicon Macs), try reinstalling the packages with architecture-specific builds:

```bash
# For Apple Silicon (M1/M2/M3) Macs:
pip install --upgrade --force-reinstall numpy opencv-python pillow
```

### Common Installation Issues

1. **NumPy source directory error**:
   ```
   ImportError: Error importing numpy: you should not try to import numpy from its source directory
   ```
   **Solution**: Run the script from a directory that is not inside the NumPy package directory.

2. **Architecture mismatch** (especially on Apple Silicon Macs):
   ```
   ImportError: dlopen(...): tried: '...' (mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64'))
   ```
   **Solution**: Reinstall the packages with the correct architecture using the command above.

3. **OpenCV import error**:
   ```
   ImportError: No module named cv2
   ```
   **Solution**: Install OpenCV using `pip install opencv-python`

## Usage

The script provides two main commands: `encrypt` for embedding messages and `decrypt` for extracting them.

### Embedding a Message

To embed a secret message into an image:

```bash
python StegoDCT.py encrypt -i input_image.jpg -m "Your secret message" -o output_image -f png
```

Parameters:
- `-i, --input`: Input image path (supports JPG, PNG and other common formats)
- `-m, --message`: Text message to embed in the image
- `-o, --output`: Output image path (extension will be added automatically)
- `-f, --format`: Output format: `png` (better quality/recommended) or `jpeg` (smaller size)
- `--max-size`: Maximum file size in bytes (optional)

### Extracting a Message

To extract a hidden message from an image:

```bash
python StegoDCT.py decrypt -i output_image.png
```

Parameters:
- `-i, --input`: Path to the image containing the hidden message

### Best Practices

- Use PNG as the output format for maximum message preservation
- The message length is limited by the image dimensions - larger images can store longer messages
- The tool will automatically calculate and verify if your message will fit in the provided image
- If the image has been modified, compressed, or processed after the message was embedded, the extraction may fail or produce incomplete/corrupted results

## How It Works

StegoDCT works by embedding message bits in the mid-frequency coefficients of the Discrete Cosine Transform (DCT) of the image. This technique:

1. Divides the image into 8x8 pixel blocks
2. Applies DCT to each block
3. Modifies specific DCT coefficients to encode message bits
4. Applies inverse DCT to restore the image

The changes are subtle enough to be imperceptible to human vision but can be detected by the algorithm.

## Limitations

- The maximum message length depends on the image size (larger images can store more text)
- Heavy compression or significant image modification may corrupt the hidden message
- The technique is most effective with PNG output (lossless compression)

## License

This project is licensed under the MIT License - see below:

```
MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
``` 
