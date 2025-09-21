import cv2
import numpy as np
from PIL import Image
import os
import pandas as pd
from pathlib import Path
import shutil

class ImageQualityClassifier:
    def __init__(self):
        """
        Initialize the Image Quality Classifier (without OCR)
        """
        # Classification thresholds - optimized for OCR readability
        self.blur_threshold = 100.0  # Laplacian variance threshold
        self.contrast_threshold = 30.0  # Standard deviation threshold
        self.brightness_min = 40
        self.brightness_max = 220
        self.noise_threshold = 15.0
        
    def calculate_blur_score(self, image):
        """
        Calculate blur score using Laplacian variance
        Higher values indicate sharper images
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var
    
    def calculate_contrast_score(self, image):
        """
        Calculate contrast score using standard deviation
        Higher values indicate better contrast
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray.std()
    
    def calculate_brightness_score(self, image):
        """
        Calculate average brightness
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray.mean()
    
    def calculate_noise_score(self, image):
        """
        Estimate noise using high frequency components
        Lower values indicate less noise
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur and subtract from original
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = cv2.subtract(gray, blurred)
        return noise.std()
    
    def detect_text_regions(self, image):
        """
        Detect potential text regions using edge detection and morphology
        Returns the percentage of image that likely contains text
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours that could be text
        text_area = 0
        total_area = image.shape[0] * image.shape[1]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:  # Minimum area for text
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                if 0.1 < aspect_ratio < 10:  # Reasonable aspect ratio for text
                    text_area += area
        
        text_percentage = (text_area / total_area) * 100
        return text_percentage
    
    def classify_image(self, image_path):
        """
        Classify a single image as clear or unclear for OCR
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                return {"classification": "error", "reason": "Could not read image"}
            
            # Calculate image quality metrics
            blur_score = self.calculate_blur_score(image)
            contrast_score = self.calculate_contrast_score(image)
            brightness_score = self.calculate_brightness_score(image)
            noise_score = self.calculate_noise_score(image)
            text_percentage = self.detect_text_regions(image)
            
            # Classification logic
            reasons = []
            score = 0
            
            # Blur check (most important for OCR)
            if blur_score >= self.blur_threshold:
                score += 3
            else:
                reasons.append(f"Too blurry (score: {blur_score:.1f}, threshold: {self.blur_threshold})")
            
            # Contrast check (important for text readability)
            if contrast_score >= self.contrast_threshold:
                score += 3
            else:
                reasons.append(f"Low contrast (score: {contrast_score:.1f}, threshold: {self.contrast_threshold})")
            
            # Brightness check (not too dark or too bright)
            if self.brightness_min <= brightness_score <= self.brightness_max:
                score += 2
            else:
                reasons.append(f"Poor brightness (score: {brightness_score:.1f}, range: {self.brightness_min}-{self.brightness_max})")
            
            # Noise check
            if noise_score <= self.noise_threshold:
                score += 1
            else:
                reasons.append(f"Too noisy (score: {noise_score:.1f}, threshold: {self.noise_threshold})")
            
            # Text presence check
            if text_percentage >= 5:  # At least 5% of image should contain potential text
                score += 1
            else:
                reasons.append(f"Low text content (percentage: {text_percentage:.1f}%)")
            
            # Final classification (need at least 7/10 points for clear)
            classification = "clear" if score >= 7 else "unclear"
            
            return {
                "classification": classification,
                "score": score,
                "max_score": 10,
                "blur_score": blur_score,
                "contrast_score": contrast_score,
                "brightness_score": brightness_score,
                "noise_score": noise_score,
                "text_percentage": text_percentage,
                "reasons": reasons if classification == "unclear" else []
            }
            
        except Exception as e:
            return {"classification": "error", "reason": str(e)}
    
    def classify_batch(self, input_folder, output_folder=None, create_report=True):
        """
        Classify all images in a folder
        """
        input_path = Path(input_folder)
        if output_folder:
            output_path = Path(output_folder)
            output_path.mkdir(exist_ok=True)
            clear_folder = output_path / "clear"
            unclear_folder = output_path / "unclear"
            error_folder = output_path / "error"
            
            clear_folder.mkdir(exist_ok=True)
            unclear_folder.mkdir(exist_ok=True)
            error_folder.mkdir(exist_ok=True)
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        print(f"Found {len(image_files)} images to classify...")
        
        results = []
        clear_count = 0
        unclear_count = 0
        error_count = 0
        
        for i, image_file in enumerate(image_files, 1):
            print(f"Processing {i}/{len(image_files)}: {image_file.name}")
            
            result = self.classify_image(str(image_file))
            result['filename'] = image_file.name
            results.append(result)
            
            classification = result['classification']
            
            # Move files to appropriate folders if output folder specified
            if output_folder:
                try:
                    if classification == "clear":
                        shutil.copy2(image_file, clear_folder / image_file.name)
                        clear_count += 1
                    elif classification == "unclear":
                        shutil.copy2(image_file, unclear_folder / image_file.name)
                        unclear_count += 1
                    else:
                        shutil.copy2(image_file, error_folder / image_file.name)
                        error_count += 1
                except Exception as e:
                    print(f"Error copying {image_file.name}: {e}")
                    error_count += 1
            else:
                if classification == "clear":
                    clear_count += 1
                elif classification == "unclear":
                    unclear_count += 1
                else:
                    error_count += 1
        
        # Create report
        if create_report:
            self.create_report(results, input_folder, output_folder)
        
        print(f"\nClassification Summary:")
        print(f"Clear images: {clear_count}")
        print(f"Unclear images: {unclear_count}")
        print(f"Error images: {error_count}")
        print(f"Total processed: {len(image_files)}")
        
        return results
    
    def create_report(self, results, input_folder, output_folder=None):
        """
        Create a detailed classification report
        """
        df = pd.DataFrame(results)
        
        # Save detailed CSV report
        if output_folder:
            report_path = Path(output_folder) / "classification_report.csv"
        else:
            report_path = Path(input_folder) / "classification_report.csv"
        
        df.to_csv(report_path, index=False)
        print(f"Detailed report saved to: {report_path}")
        
        # Create summary statistics
        summary = df['classification'].value_counts()
        print(f"\nClassification Summary:")
        for classification, count in summary.items():
            print(f"{classification}: {count}")
        
        # Print some statistics for clear images
        clear_images = df[df['classification'] == 'clear']
        if not clear_images.empty:
            print(f"\nClear Images Statistics:")
            print(f"Average blur score: {clear_images['blur_score'].mean():.1f}")
            print(f"Average contrast score: {clear_images['contrast_score'].mean():.1f}")
            print(f"Average brightness score: {clear_images['brightness_score'].mean():.1f}")
            print(f"Average text percentage: {clear_images['text_percentage'].mean():.1f}%")
        
        # Print some statistics for unclear images
        unclear_images = df[df['classification'] == 'unclear']
        if not unclear_images.empty:
            print(f"\nUnclear Images Statistics:")
            print(f"Average blur score: {unclear_images['blur_score'].mean():.1f}")
            print(f"Average contrast score: {unclear_images['contrast_score'].mean():.1f}")
            print(f"Average brightness score: {unclear_images['brightness_score'].mean():.1f}")
            print(f"Average text percentage: {unclear_images['text_percentage'].mean():.1f}%")
            
            # Show common reasons for unclear classification
            all_reasons = []
            for reasons in unclear_images['reasons']:
                if isinstance(reasons, list):
                    all_reasons.extend(reasons)
            
            if all_reasons:
                print(f"\nMost common issues with unclear images:")
                reason_counts = pd.Series(all_reasons).value_counts()
                for reason, count in reason_counts.head(5).items():
                    print(f"  {reason}: {count} images")

if __name__ == "__main__":
    # Initialize classifier
    classifier = ImageQualityClassifier()
    
    # Set the input folder (current directory)
    input_folder = "."
    output_folder = "classified_images"
    
    # Run classification
    results = classifier.classify_batch(
        input_folder=input_folder,
        output_folder=output_folder,
        create_report=True
    )