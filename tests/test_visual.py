import unittest
import numpy as np
import cv2
from perception.visual import calculate_blur, calculate_brightness, calculate_region_brightness

class TestVisualAlgorithms(unittest.TestCase):
    def setUp(self):
        # Create a simple synthetic 100x100 BGR image (solid gray)
        self.flat_image = np.full((100, 100, 3), 128, dtype=np.uint8)
        
        # Create a noisy image
        self.noisy_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Create an image with a bright center
        self.bright_center = np.full((100, 100, 3), 50, dtype=np.uint8)
        self.bright_center[40:60, 40:60] = (255, 255, 255)

    def test_calculate_brightness(self):
        # Flat image should be exactly 128
        b = calculate_brightness(self.flat_image)
        self.assertAlmostEqual(b, 128.0, delta=1.0)
        
    def test_calculate_region_brightness(self):
        # Region in the bright center (nx=0.4, ny=0.4, nw=0.2, nh=0.2)
        bbox = (0.4, 0.4, 0.2, 0.2)
        b = calculate_region_brightness(self.bright_center, bbox)
        self.assertAlmostEqual(b, 255.0, delta=1.0)
        
        # Region outside the bright center
        bbox_outside = (0.1, 0.1, 0.2, 0.2)
        b_outside = calculate_region_brightness(self.bright_center, bbox_outside)
        self.assertAlmostEqual(b_outside, 50.0, delta=1.0)

    def test_calculate_blur(self):
        # Flat image has 0 variance in laplacian (completely blurred/flat)
        blur_flat = calculate_blur(self.flat_image)
        self.assertAlmostEqual(blur_flat, 0.0, delta=0.1)
        
        # Noisy image should have high variance (sharp edges)
        blur_noisy = calculate_blur(self.noisy_image)
        self.assertGreater(blur_noisy, 100.0)

if __name__ == '__main__':
    unittest.main()
