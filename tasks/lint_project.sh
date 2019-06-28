# This contains a convenient way to lint the project.
echo "Linting..."
pylint --score n image_caption_generator/ training/
echo "Done!"
echo ""

echo "Type-checking..."
mypy image_caption_generator/ training/
echo "Done!"
echo ""
