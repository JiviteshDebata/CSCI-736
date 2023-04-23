import pygame

# Initialize Pygame
pygame.init()

# Define the image filename and size
filename = "present_image.png"
size = (800, 600)

# Load the image
img = pygame.image.load(filename)
img = pygame.transform.scale(img, size)

# Create a window with the same size as the image
screen = pygame.display.set_mode(size)
pygame.display.set_caption("Image Viewer")

# Set the background color to white
screen.fill((255, 255, 255))

# Display the image in the center of the screen
screen.blit(img, (size[0] // 2 - img.get_width() // 2, size[1] // 2 - img.get_height() // 2))

# Start the event loop
while True:
    # Process Pygame events
    for event in pygame.event.get():
        # If the user presses a key, log the key code
        if event.type == pygame.KEYDOWN:
            print("Key pressed:", event.key)
        
        # If the user clicks the close button, exit the program
        elif event.type == pygame.QUIT:
            pygame.quit()
            exit()
    
    # Update the display
    pygame.display.update()
