# This contains a convenient way to lint the project.
echo "Linting..."
pylint --score n caption_generator/
echo "Done!"
echo ""

echo "Type-checking..."
mypy caption_generator/
echo "Done!"
echo ""
