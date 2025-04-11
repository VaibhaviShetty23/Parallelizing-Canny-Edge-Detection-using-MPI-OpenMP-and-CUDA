from PIL import Image
import numpy as np

def png_to_pgm_p2(input_png, output_pgm):
    # Open the image
    img = Image.open(input_png)

    # Convert to grayscale (L mode)
    img = img.convert('L')

    # Convert the image to a numpy array
    img_array = np.array(img)

    # Get image dimensions
    height, width = img_array.shape

    # Open the output PGM file in write mode
    with open(output_pgm, 'w') as f:
        # Write the PGM header
        f.write(f'P2\n{width} {height}\n255\n')

        # Write pixel values
        for row in img_array:
            f.write(' '.join(map(str, row)) + '\n')

# Example usage
png_to_pgm_p2("emoji.png", "output_image.pgm")
