# This contains a series of commands to install the COCO API which is used for retrieving image captions.
# For more information, visit https://github.com/cocodataset/cocoapi.

# Remove existing folders
rm -fr caption_generator/datasets/pycocotools
rm -fr caption_generator/datasets/cocoapi

# Create directory to hold COCOAPI
mkdir caption_generator/datasets/cocoapi

# Clone COCOAPI
git clone https://github.com/cocodataset/cocoapi.git caption_generator/datasets/cocoapi

# Install pycocotools
cd caption_generator/datasets/cocoapi/PythonAPI
make install

# Move pycocotools to caption_generator/datasets
mv ./pycocotools ../../

# Clean up
rm -fr ../../cocoapi
