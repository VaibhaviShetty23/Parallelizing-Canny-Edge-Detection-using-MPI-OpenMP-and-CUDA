from PIL import Image

def pgm_to_png(input_pgm, output_png):
    # Open the PGM file
    img = Image.open(input_pgm)

    # Save it as PNG
    img.save(output_png)

# Example usage
pgm_to_png("out.pgm", "show.png")
