#!/usr/bin/env python3
"""
StegoDCT: A CLI tool for embedding and extracting textual messages in images using DCT watermarking.

This module provides functionality to imperceptibly inscribe textual data within the frequency 
domain of JPEG and PNG images through manipulation of discrete cosine transform coefficients.

Usage examples:
  # Embed a message into an image and save as PNG:
  python3 StegoDCT.py encrypt -i input.jpg -m "Secret message" -o output -f png
  
  # Extract a message from an image:
  python3 StegoDCT.py decrypt -i output.png
  
The technique works by modifying the DCT coefficients in the mid-frequency range, 
ensuring that changes remain imperceptible to the human eye while being robust 
enough to survive common image processing operations.
"""

import argparse
import numpy as np
import cv2
from PIL import Image
import io
import os
import sys
from typing import Tuple, Union, Optional


class StegoDCT:
    """Implementation of steganography using Discrete Cosine Transform."""
    
    # Constants for DCT processing
    BLOCK_SIZE = 8  # DCT block size
    QUANTIZATION_FACTOR = 25  # Controls embedding strength
    THRESHOLD = 15  # Threshold for detection
    
    def __init__(self, max_file_size: Optional[int] = None):
        """
        Initialize the StegoDCT object.
        
        Args:
            max_file_size: Maximum file size in bytes (optional)
        """
        self.max_file_size = max_file_size
    
    def _string_to_bits(self, message: str) -> list:
        """Convert a string to a list of bits."""
        # Convert message to bytes and then to bits
        byte_array = message.encode('utf-8')
        bits = []
        for byte in byte_array:
            for i in range(7, -1, -1):  # Most significant bit first
                bits.append((byte >> i) & 1)
        
        # Add terminator sequence (16 ones)
        bits.extend([1] * 16)
        return bits
    
    def _bits_to_string(self, bits: list) -> str:
        """Convert a list of bits to a string."""
        # Group bits into bytes
        bytes_data = bytearray()
        for i in range(0, len(bits), 8):
            if i + 8 > len(bits):  # Incomplete byte at the end
                break
            
            byte = 0
            for j in range(8):
                byte = (byte << 1) | bits[i + j]
            bytes_data.append(byte)
        
        # Decode bytes to string
        try:
            return bytes_data.decode('utf-8')
        except UnicodeDecodeError:
            # Handle potential decode errors by returning up to the last valid UTF-8 sequence
            for i in range(len(bytes_data), 0, -1):
                try:
                    return bytes_data[:i].decode('utf-8')
                except UnicodeDecodeError:
                    continue
            return ""
    
    def _prepare_image(self, image_path: str) -> np.ndarray:
        """Load and prepare the image for DCT processing."""
        # Check if file exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        # Check file size if max_file_size is specified
        if self.max_file_size is not None:
            file_size = os.path.getsize(image_path)
            if file_size > self.max_file_size:
                raise ValueError(f"Input file size ({file_size} bytes) exceeds maximum allowed size ({self.max_file_size} bytes)")
        
        # Read the image
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            # Try using PIL if OpenCV fails (better support for various formats)
            try:
                pil_img = Image.open(image_path)
                img = np.array(pil_img.convert('RGB'))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            except Exception as e:
                raise ValueError(f"Failed to open image: {e}")
        
        # Check if image is valid
        if img.size == 0 or img.shape[0] < self.BLOCK_SIZE or img.shape[1] < self.BLOCK_SIZE:
            raise ValueError(f"Image is too small. Minimum dimensions required: {self.BLOCK_SIZE}x{self.BLOCK_SIZE}")
            
        return img
    
    def _save_image(self, img: np.ndarray, output_path: str, format_type: str) -> None:
        """Save the image in the specified format."""
        format_type = format_type.lower()
        
        # Make sure output_path has the correct extension
        base_path, _ = os.path.splitext(output_path)
        if format_type == 'png':
            output_path = f"{base_path}.png"
            cv2.imwrite(output_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        elif format_type == 'jpeg' or format_type == 'jpg':
            output_path = f"{base_path}.jpg"
            cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            raise ValueError(f"Unsupported output format: {format_type}")
            
        # Verify the file was created
        if not os.path.exists(output_path):
            raise IOError(f"Failed to save image to {output_path}")
    
    def _embed_bit_in_dct_block(self, block: np.ndarray, bit: int) -> np.ndarray:
        """Embed a single bit into a DCT coefficient block."""
        # Convert block to float32 for DCT
        block_float = np.float32(block)
        
        # Apply DCT to the block
        dct_block = cv2.dct(block_float)
        
        # Modify the mid-frequency coefficient (4,4) based on the bit
        # This position balances between robustness and perceptibility
        if bit == 1:
            # Ensure the coefficient is positive and above threshold
            if dct_block[4, 4] < self.THRESHOLD:
                dct_block[4, 4] = self.THRESHOLD + self.QUANTIZATION_FACTOR
        else:
            # Ensure the coefficient is negative or below threshold
            if dct_block[4, 4] > -self.THRESHOLD:
                dct_block[4, 4] = -self.THRESHOLD - self.QUANTIZATION_FACTOR
        
        # Apply inverse DCT
        idct_block = cv2.idct(dct_block)
        
        # Clip values to valid range
        return np.clip(idct_block, 0, 255).astype(np.uint8)
    
    def _extract_bit_from_dct_block(self, block: np.ndarray) -> int:
        """Extract a single bit from a DCT coefficient block."""
        # Convert block to float32 for DCT
        block_float = np.float32(block)
        
        # Apply DCT to the block
        dct_block = cv2.dct(block_float)
        
        # Get the bit based on the mid-frequency coefficient (4,4)
        return 1 if dct_block[4, 4] > 0 else 0
    
    def encrypt(self, image_path: str, message: str, output_path: str, output_format: str) -> None:
        """
        Embed a message into an image using DCT.
        
        Args:
            image_path: Path to the input image
            message: Text message to embed
            output_path: Path to save the output image
            output_format: Output format ('png' or 'jpeg')
        """
        img = self._prepare_image(image_path)
        height, width, channels = img.shape
        
        # Convert the message to a bit sequence
        bits = self._string_to_bits(message)
        
        # Calculate the maximum number of bits we can embed
        # Account for complete blocks only
        blocks_height = height // self.BLOCK_SIZE
        blocks_width = width // self.BLOCK_SIZE
        max_bits = blocks_height * blocks_width * channels
        
        if len(bits) > max_bits:
            raise ValueError(f"Message too long for this image. Maximum bits: {max_bits}, Required: {len(bits)}")
        
        bit_index = 0
        modified_img = np.copy(img)
        
        # Process each channel separately
        for channel in range(3):  # RGB channels
            if bit_index >= len(bits):
                break
                
            channel_data = modified_img[:, :, channel]
            
            # Process 8x8 blocks
            for y in range(0, blocks_height * self.BLOCK_SIZE, self.BLOCK_SIZE):
                for x in range(0, blocks_width * self.BLOCK_SIZE, self.BLOCK_SIZE):
                    if bit_index >= len(bits):
                        break
                        
                    # Get current block
                    block = channel_data[y:y+self.BLOCK_SIZE, x:x+self.BLOCK_SIZE]
                    
                    # Embed the bit
                    modified_block = self._embed_bit_in_dct_block(block, bits[bit_index])
                    
                    # Update the image with the modified block
                    modified_img[y:y+self.BLOCK_SIZE, x:x+self.BLOCK_SIZE, channel] = modified_block
                    
                    bit_index += 1
                
                if bit_index >= len(bits):
                    break
        
        # Save the modified image
        self._save_image(modified_img, output_path, output_format)
        print(f"Message successfully embedded in {output_path}")
    
    def decrypt(self, image_path: str) -> str:
        """
        Extract a message from an image using DCT.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            The extracted message as a string
        """
        img = self._prepare_image(image_path)
        height, width, channels = img.shape
        
        # Get only complete blocks
        blocks_height = height // self.BLOCK_SIZE
        blocks_width = width // self.BLOCK_SIZE
        
        extracted_bits = []
        consecutive_ones = 0
        
        # Process each channel separately
        for channel in range(3):  # RGB channels
            channel_data = img[:, :, channel]
            
            # Process 8x8 blocks
            for y in range(0, blocks_height * self.BLOCK_SIZE, self.BLOCK_SIZE):
                for x in range(0, blocks_width * self.BLOCK_SIZE, self.BLOCK_SIZE):
                    # Check for termination sequence (16 consecutive ones)
                    if consecutive_ones >= 16:
                        break
                        
                    # Get current block
                    block = channel_data[y:y+self.BLOCK_SIZE, x:x+self.BLOCK_SIZE]
                    
                    # Extract the bit
                    bit = self._extract_bit_from_dct_block(block)
                    extracted_bits.append(bit)
                    
                    # Check for termination sequence
                    if bit == 1:
                        consecutive_ones += 1
                    else:
                        consecutive_ones = 0
                
                if consecutive_ones >= 16:
                    break
            
            if consecutive_ones >= 16:
                break
        
        # Remove the termination sequence
        if consecutive_ones >= 16:
            extracted_bits = extracted_bits[:-consecutive_ones]
        else:
            print("Warning: No termination sequence found. Message might be incomplete or corrupted.")
        
        # Convert bits to string
        return self._bits_to_string(extracted_bits)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="""
StegoDCT - A tool for embedding and extracting messages in images using DCT watermarking

This tool allows you to:
1. Hide text messages within images using the Discrete Cosine Transform (DCT) technique
2. Extract hidden messages from images that were previously encoded

The technique works by subtly modifying frequency coefficients in the image in a way that is 
imperceptible to human vision but can be detected by the algorithm.

ADVANTAGES:
- Provides strong resistance against visual detection
- Works with both JPEG and PNG images
- Maintains high visual quality of the carrier image
- Survives moderate image processing operations
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Embed a message in an image and save as PNG:
  python3 StegoDCT.py encrypt -i input.jpg -m "Secret message" -o output -f png
  
  # Extract a message from a stego image:
  python3 StegoDCT.py decrypt -i output.png
  
For best results, use PNG as the output format for maximum message preservation.
"""
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Encrypt command
    encrypt_parser = subparsers.add_parser("encrypt", 
        help="Embed a message into an image",
        description="""
Embed a text message into an image using the DCT steganography technique.

The process divides the image into 8x8 pixel blocks and modifies specific 
frequency coefficients to encode the message bits. The visual appearance 
of the image is preserved while the message is securely embedded.

The message length is limited by the image dimensions - larger images can 
store longer messages. The tool will automatically calculate and verify if
your message will fit in the provided image.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter)
        
    encrypt_parser.add_argument("-i", "--input", required=True, 
                            help="Input image path (supports JPG, PNG and other common formats)")
    encrypt_parser.add_argument("-m", "--message", required=True, 
                            help="Text message to embed in the image")
    encrypt_parser.add_argument("-o", "--output", required=True, 
                            help="Output image path (extension will be added automatically based on format)")
    encrypt_parser.add_argument("-f", "--format", choices=["png", "jpeg"], default="png", 
                            help="Output format: png (better quality/recommended) or jpeg (smaller size)")
    encrypt_parser.add_argument("--max-size", type=int, 
                            help="Maximum file size in bytes (optional, limits the size of input file)")
    
    # Decrypt command
    decrypt_parser = subparsers.add_parser("decrypt", 
        help="Extract a message from an image",
        description="""
Extract a hidden message from an image that was previously encoded using this tool.

The process analyzes the DCT coefficients of 8x8 pixel blocks to recover the 
embedded message bits. The extraction process terminates when it encounters 
the special termination sequence (16 consecutive '1' bits).

Note that if the image has been modified, compressed, or processed after the 
message was embedded, the extraction may fail or produce incomplete/corrupted results.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter)
        
    decrypt_parser.add_argument("-i", "--input", required=True, 
                             help="Path to the image containing the hidden message")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == "encrypt":
            steganographer = StegoDCT(max_file_size=args.max_size)
            steganographer.encrypt(args.input, args.message, args.output, args.format)
        
        elif args.command == "decrypt":
            steganographer = StegoDCT()
            message = steganographer.decrypt(args.input)
            print(f"Extracted message: {message}")
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
    