# 📸 OCR Image Quality Classifier

A powerful Python tool that automatically classifies images as **clear** or **unclear** for Optical Character Recognition (OCR) processing. This tool uses advanced computer vision techniques to analyze image quality metrics and determine OCR readability.

## 🎯 Overview

This project was developed to solve the common problem of preprocessing images before OCR processing. By automatically filtering out poor-quality images, you can significantly improve OCR accuracy and processing efficiency.

### Key Features

- ✅ **Automated Classification**: Batch process hundreds of images quickly
- 🔍 **Multi-metric Analysis**: Combines blur detection, contrast analysis, brightness assessment, and noise estimation
- 📊 **Detailed Reporting**: Generates comprehensive CSV reports with quality metrics
- 📁 **Organized Output**: Automatically sorts images into clear/unclear folders
- 🚀 **Fast Processing**: Optimized algorithms for quick batch processing
- 📈 **Quality Metrics**: Provides detailed quality scores for each image

## 🛠️ Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Kalana-Lakshan/Mobitel_Hackathon_Classified_Images.git
   cd Mobitel_Hackathon_Classified_Images
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   
   # On Windows
   .venv\Scripts\activate
   
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. **Install required packages**
   ```bash
   pip install opencv-python numpy pillow pandas matplotlib
   ```

## 🚀 Usage

### Quick Start

1. **Place your images** in the project directory
2. **Run the classifier**:
   ```bash
   python quick_classifier.py
   ```

### Custom Configuration

You can customize the classification parameters by modifying the thresholds in the `ImageQualityClassifier` class:

```python
# Classification thresholds
self.blur_threshold = 100.0      # Laplacian variance threshold
self.contrast_threshold = 30.0   # Standard deviation threshold
self.brightness_min = 40         # Minimum brightness
self.brightness_max = 220        # Maximum brightness
self.noise_threshold = 15.0      # Noise level threshold
```

### Advanced Usage

```python
from quick_classifier import ImageQualityClassifier

# Initialize classifier
classifier = ImageQualityClassifier()

# Classify single image
result = classifier.classify_image("path/to/image.jpg")
print(f"Classification: {result['classification']}")
print(f"Quality score: {result['score']}/10")

# Batch process images
results = classifier.classify_batch(
    input_folder="input_images/",
    output_folder="classified_output/",
    create_report=True
)
```

## 📊 How It Works

### Quality Metrics Analyzed

1. **Blur Detection** 🔍
   - Uses Laplacian variance to measure image sharpness
   - Higher values indicate sharper, clearer images
   - Threshold: 100.0

2. **Contrast Analysis** ⚡
   - Calculates standard deviation of pixel intensities
   - Better contrast improves OCR accuracy
   - Threshold: 30.0

3. **Brightness Assessment** ☀️
   - Ensures optimal brightness range (40-220)
   - Avoids overly dark or bright images

4. **Noise Estimation** 📡
   - Detects high-frequency noise that interferes with OCR
   - Lower noise levels preferred

5. **Text Region Detection** 📝
   - Identifies potential text areas using morphological operations
   - Ensures sufficient text content for OCR processing

### Scoring System

Each image receives a score out of 10 points:
- **Blur Detection**: 3 points
- **Contrast Analysis**: 3 points  
- **Brightness Assessment**: 2 points
- **Noise Level**: 1 point
- **Text Content**: 1 point

**Classification Threshold**: 7/10 points for "clear" classification

## 📁 Output Structure

After processing, your images will be organized as follows:

```
classified_images/
├── clear/                    # Images suitable for OCR (score ≥ 7/10)
├── unclear/                  # Images needing improvement (score < 7/10)
├── error/                    # Images that couldn't be processed
└── classification_report.csv # Detailed analysis report
```

## 📈 Sample Results

### Classification Summary
- **Total Images Processed**: 94
- **Clear Images**: 64 (68.1%)
- **Unclear Images**: 30 (31.9%)
- **Processing Time**: ~2-3 seconds per image

### Quality Statistics

| Metric | Clear Images | Unclear Images |
|--------|-------------|----------------|
| **Average Blur Score** | 1,162.3 | 126.5 |
| **Average Contrast** | 44.4 | 20.4 |
| **Average Brightness** | 172.5 | 144.1 |
| **Text Coverage** | 95.3% | 95.5% |

## 🔧 Customization

### Adjusting Sensitivity

To make the classifier more **strict** (fewer clear images):
```python
self.blur_threshold = 150.0     # Increase blur threshold
self.contrast_threshold = 40.0  # Increase contrast threshold
```

To make the classifier more **lenient** (more clear images):
```python
self.blur_threshold = 80.0      # Decrease blur threshold
self.contrast_threshold = 25.0  # Decrease contrast threshold
```

### Adding Custom Metrics

You can extend the classifier by adding your own quality metrics:

```python
def custom_quality_metric(self, image):
    # Your custom analysis here
    return quality_score

def classify_image(self, image_path):
    # Add your custom metric to the scoring
    custom_score = self.custom_quality_metric(image)
    # Include in final classification logic
```

## 📝 Report Analysis

The generated `classification_report.csv` contains detailed information for each image:

| Column | Description |
|--------|-------------|
| `filename` | Original image filename |
| `classification` | clear/unclear/error |
| `score` | Overall quality score (0-10) |
| `blur_score` | Laplacian variance value |
| `contrast_score` | Standard deviation value |
| `brightness_score` | Average brightness |
| `noise_score` | Noise estimation |
| `text_percentage` | Percentage of text regions |
| `reasons` | Issues found (for unclear images) |

## 🎯 Use Cases

- **Document Digitization**: Pre-filter scanned documents
- **Receipt Processing**: Classify receipt photos for OCR
- **License Plate Recognition**: Filter clear vehicle images  
- **Text Extraction**: Prepare images for text recognition
- **Quality Control**: Automated image quality assessment
- **Batch Processing**: Handle large image datasets efficiently

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏆 Acknowledgments

- **OpenCV** for computer vision capabilities
- **NumPy** for numerical operations  
- **Pandas** for data analysis and reporting
- **PIL/Pillow** for image processing
- **Mobitel Hackathon** for inspiring this solution

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/Kalana-Lakshan/Mobitel_Hackathon_Classified_Images/issues) page
2. Create a new issue with detailed description
3. Include sample images and error messages if applicable

## 🚀 Future Enhancements

- [ ] **OCR Integration**: Add Tesseract OCR confidence scoring
- [ ] **Deep Learning**: Implement CNN-based quality assessment
- [ ] **GUI Interface**: Create user-friendly desktop application
- [ ] **API Endpoint**: REST API for web integration
- [ ] **Mobile App**: Flutter/React Native mobile version
- [ ] **Cloud Integration**: AWS/Azure cloud processing
- [ ] **Real-time Processing**: Live camera feed classification

---

**⭐ Star this repository if you found it helpful!**

Made with ❤️ for the Mobitel Hackathon