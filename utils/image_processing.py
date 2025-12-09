"""
Image Processing Utilities
Advanced image manipulation and computer vision functions
"""
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import io


class ImageProcessor:
    """Handle various image processing operations"""
    
    def __init__(self, image_path=None):
        self.image = None
        if image_path:
            self.load_image(image_path)
    
    def load_image(self, path):
        """Load image from file path"""
        self.image = Image.open(path)
        return self.image
    
    def resize(self, width, height, maintain_aspect=True):
        """Resize image to specified dimensions"""
        if maintain_aspect:
            self.image.thumbnail((width, height), Image.Resampling.LANCZOS)
        else:
            self.image = self.image.resize((width, height), Image.Resampling.LANCZOS)
        return self.image
    
    def rotate(self, angle):
        """Rotate image by specified angle"""
        self.image = self.image.rotate(angle, expand=True)
        return self.image
    
    def flip(self, direction='horizontal'):
        """Flip image horizontally or vertically"""
        if direction == 'horizontal':
            self.image = self.image.transpose(Image.FLIP_LEFT_RIGHT)
        elif direction == 'vertical':
            self.image = self.image.transpose(Image.FLIP_TOP_BOTTOM)
        return self.image
    
    def apply_filter(self, filter_type='blur'):
        """Apply various filters to image"""
        filters = {
            'blur': ImageFilter.BLUR,
            'sharpen': ImageFilter.SHARPEN,
            'smooth': ImageFilter.SMOOTH,
            'edge_enhance': ImageFilter.EDGE_ENHANCE,
            'contour': ImageFilter.CONTOUR,
            'emboss': ImageFilter.EMBOSS
        }
        
        if filter_type in filters:
            self.image = self.image.filter(filters[filter_type])
        return self.image
    
    def adjust_brightness(self, factor=1.5):
        """Adjust image brightness"""
        enhancer = ImageEnhance.Brightness(self.image)
        self.image = enhancer.enhance(factor)
        return self.image
    
    def adjust_contrast(self, factor=1.5):
        """Adjust image contrast"""
        enhancer = ImageEnhance.Contrast(self.image)
        self.image = enhancer.enhance(factor)
        return self.image
    
    def adjust_saturation(self, factor=1.5):
        """Adjust color saturation"""
        enhancer = ImageEnhance.Color(self.image)
        self.image = enhancer.enhance(factor)
        return self.image
    
    def convert_to_grayscale(self):
        """Convert image to grayscale"""
        self.image = self.image.convert('L')
        return self.image
    
    def crop(self, left, top, right, bottom):
        """Crop image to specified rectangle"""
        self.image = self.image.crop((left, top, right, bottom))
        return self.image
    
    def add_border(self, border_width=10, color='black'):
        """Add border around image"""
        from PIL import ImageOps
        self.image = ImageOps.expand(self.image, border=border_width, fill=color)
        return self.image
    
    def to_array(self):
        """Convert image to numpy array"""
        return np.array(self.image)
    
    def from_array(self, array):
        """Create image from numpy array"""
        self.image = Image.fromarray(array.astype('uint8'))
        return self.image
    
    def save(self, path, quality=95):
        """Save image to file"""
        self.image.save(path, quality=quality, optimize=True)
    
    def get_histogram(self):
        """Get image histogram data"""
        return self.image.histogram()
    
    def detect_edges(self):
        """Simple edge detection"""
        self.image = self.image.filter(ImageFilter.FIND_EDGES)
        return self.image
    
    def apply_gaussian_blur(self, radius=2):
        """Apply Gaussian blur"""
        self.image = self.image.filter(ImageFilter.GaussianBlur(radius))
        return self.image
    
    def create_thumbnail(self, size=(128, 128)):
        """Create thumbnail version"""
        thumb = self.image.copy()
        thumb.thumbnail(size, Image.Resampling.LANCZOS)
        return thumb


def batch_process_images(input_dir, output_dir, operations):
    """
    Process multiple images with specified operations
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save processed images
        operations: List of operations to apply
    """
    import os
    from pathlib import Path
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    
    for file_path in input_path.iterdir():
        if file_path.suffix.lower() in image_extensions:
            processor = ImageProcessor(str(file_path))
            
            for operation, params in operations:
                if hasattr(processor, operation):
                    method = getattr(processor, operation)
                    method(**params)
            
            output_file = output_path / file_path.name
            processor.save(str(output_file))
            print(f"Processed: {file_path.name}")


def create_collage(image_paths, grid_size=(2, 2), image_size=(200, 200)):
    """
    Create image collage from multiple images
    
    Args:
        image_paths: List of image file paths
        grid_size: Tuple of (rows, cols)
        image_size: Size to resize each image
    """
    rows, cols = grid_size
    width, height = image_size
    
    collage = Image.new('RGB', (cols * width, rows * height), 'white')
    
    for idx, path in enumerate(image_paths[:rows * cols]):
        if idx >= rows * cols:
            break
        
        img = Image.open(path)
        img.thumbnail(image_size, Image.Resampling.LANCZOS)
        
        row = idx // cols
        col = idx % cols
        
        collage.paste(img, (col * width, row * height))
    
    return collage


def extract_dominant_colors(image_path, num_colors=5):
    """Extract dominant colors from image"""
    from collections import Counter
    
    img = Image.open(image_path)
    img = img.resize((150, 150))
    img = img.convert('RGB')
    
    pixels = list(img.getdata())
    
    # Count color frequency
    color_counts = Counter(pixels)
    dominant = color_counts.most_common(num_colors)
    
    return [color for color, count in dominant]
