import csv
from PIL import Image

# Create a 28x28 image object
img = Image.new('L', (28, 28))

# Read the CSV file
with open(r'C:\Users\johns\Documents\NNandML\OpenSet\Project\CSCI-736\Experiments\Original_image.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        # Parse the row into x, y, and pixel value
        # print(row)
        x, y = row[0].split('x')
        pixel = int(row[1])
        x, y, pixel = int(x)-1, int(y)-1, int(pixel)
        # Set the pixel value in the image
        print((x, y), pixel)
        img.putpixel((y, x), pixel)

# Save the image
img.save('digit_original.png')
