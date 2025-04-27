from PIL import Image

# Load the image
input_path = 'input/emoji.png'  # original image path
output_path = 'input/emoji1.png'

# Open the image
img = Image.open(input_path)

# Compute new size (4x upscale)
new_size = (img.width // 4, img.height // 4)

# Resize using high-quality resampling
upscaled_img = img.resize(new_size, Image.LANCZOS)

# Save the upscaled image
upscaled_img.save(output_path)