# Import the necessary libraries
import numpy as np
import imageio
from skimage import filters, color
import matplotlib.pyplot as plt

# Defined class SeamEnergyWithBackPointer for dynamic programming of optimal seam path
class Seam_energy_minEnergy_pointer():
    def __init__(self, energy, x_coordinate_in_previous_row=None):
        self.energy = energy
        self.x_coordinate_in_previous_row = x_coordinate_in_previous_row

def min_seam_energy(pixel_energies, direction = 'vertical'):

    # Rotate image 90 degree if seam direction is horizontal
    if direction == 'horizontal':
        pixel_energies = pixel_energies.T

    seam_energies = []
    # Initialize the top row of seam energies by copying over the top
    # row of the pixel energies. There are no back pointers in the
    # top row.
    seam_energies.append([
        Seam_energy_minEnergy_pointer(pixel_energy)
        for pixel_energy in pixel_energies[0]
    ])
    # Skip the first row in the following loop.

    for y in range(1, pixel_energies.shape[0]):
        pixel_energies_row = pixel_energies[y]
        seam_energies_row = []
        for x, pixel_energy in enumerate(pixel_energies_row):
            # Determine the range of x values to iterate over in the
            # previous row. The range depends on if the current pixel
            # is in the middle of the image, or on one of the edges.
            x_left = max(x - 1, 0)
            x_right = min(x + 1, len(pixel_energies_row) - 1)
            x_range = range(x_left, x_right + 1)
            min_parent_x = min(
                                x_range,
                                key=lambda x_i: seam_energies[y - 1][x_i].energy
            )
            min_seam_energy = Seam_energy_minEnergy_pointer(
                pixel_energy + seam_energies[y - 1][min_parent_x].energy,
                min_parent_x
            )
            seam_energies_row.append(min_seam_energy)
        seam_energies.append(seam_energies_row)

    # Reverse back image if seam direction is horizontal
    if direction == 'horizontal':
        seam_energies = np.array(seam_energies).T.tolist()

    return seam_energies

def retrieve_seam(seam_energies, direction = 'vertical'):

    # Find the x coordinate with minimal seam energy in the bottom row.
    if direction == 'vertical':
        # seam_energies is a list of arrays here, looking for min energy at bottom row
        min_seam_end_x = min(
            range(len(seam_energies[-1])),
            key=lambda x: seam_energies[-1][x].energy
        )

        # Follow the back pointers to form a list of coordinates that
        # form the lowest-energy seam.
        seam = []
        seam_point_x = min_seam_end_x
        # Reversely adding local minimum energy coordinates back to top row of image
        for y in range(len(seam_energies) - 1, -1, -1):
            seam.append((seam_point_x, y))
            seam_point_x = \
                seam_energies[y][seam_point_x].x_coordinate_in_previous_row
        seam.reverse()

    elif direction == 'horizontal':
        # seam_energies below is a numpy arrays, used different approach than above,
        # looking for min energy at right most column
        seam_energies = np.array(seam_energies)
        min_seam_end_y = min(
            range(seam_energies.shape[0]),
            key=lambda x: seam_energies[:,-1][x].energy
        )

        # Follow the back pointers to form a list of coordinates that
        # form the lowest-energy seam.
        seam = []
        seam_point_y = min_seam_end_y
        # Reversely adding local minimum energy coordinates back to left most column of image
        for x in range(seam_energies.shape[1] - 1, -1, -1):
            seam.append((x, seam_point_y))
            seam_point_y = \
                seam_energies[:, x][seam_point_y].x_coordinate_in_previous_row
        seam.reverse()

    return seam

def remove_seam(seam, img, direction = 'vertical'):

    if direction == 'vertical':
        # Create a copy of img with size decreased
        new_pixels = np.zeros((img.shape[0], img.shape[1]-1, 3), dtype = 'float32')
        for row in range(img.shape[0]):
            offset = 0
            for col in range(img.shape[1]-1):
                if (col, row) in seam:
                    offset = 1
                new_pixels[row, col, :] = img[row, col + offset, :]

    if direction == 'horizontal':
        # Create a copy of img with size decreased
        new_pixels = np.zeros((img.shape[0]-1, img.shape[1], 3), dtype = 'float32')
        for col in range(img.shape[1]):
            offset = 0
            for row in range(img.shape[0]-1):
                if (col, row) in seam:
                    offset = 1
                new_pixels[row, col, :] = img[row + offset, col, :]

    return new_pixels

def add_seam(seam, img, direction = 'vertical'):

    if direction == 'vertical':
        # Create a copy of img with size increased
        new_pixels = np.zeros((img.shape[0], img.shape[1]+1, 3), dtype = 'float32')
        for row in range(img.shape[0]):
            offset = 0
            for col in range(img.shape[1]):
                if (col, row) in seam:
                    new_pixels[row, col, :] = img[row, col, :]
                    # Duplicate pixel (col, row) to next column with averaged pixel value between neighbors
                    new_pixels[row, col + 1, :] = (img[row, col, :] + img[row, col+1, :])/2
                    offset = 1
                    continue
                new_pixels[row, col+offset, :] = img[row, col, :]

    elif direction == 'horizontal':
        # Create a copy of img with size increased
        new_pixels = np.zeros((img.shape[0] + 1, img.shape[1], 3), dtype='float32')
        for col in range(img.shape[1]):
            offset = 0
            for row in range(img.shape[0]):
                if (col, row) in seam:
                    new_pixels[row, col, :] = img[row, col, :]
                    # Duplicate pixel (col, row) to next row with averaged pixel value between neighbors
                    new_pixels[row+1, col, :] = (img[row, col, :] + img[row+1, col, :])/2
                    offset = 1
                    continue
                new_pixels[row+offset, col, :] = img[row, col, :]

    return new_pixels

def fill_seam(seam, img):

    # change img pixels in the seam to red
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            if (col, row) in seam:
                img[row, col, :] = [255,0,0]

    return img

def seam_mark(img, direction, pixels):

    # img is for finding first n ( in here is pixels) seams to be removed, img_copy is for adding seams
    img_copy = np.copy(img)
    lst_seam = []

    for i in range(pixels):
        print(f"Looking for {direction} seam No.{i + 1}")

        # Calculate img energy and assign to each pixel point
        pixel_energies = energy(img)

        # Calculate list of arrays of minimum energy at each pixel point
        seam_energies = min_seam_energy(pixel_energies, direction)

        # Find the seam path that goes from top to bottom or left to right
        seam = retrieve_seam(seam_energies, direction)

        # Collect all the seam path
        lst_seam.append(seam)

        # Remove the seam and change img
        img = remove_seam(seam, img, direction)

    for i in range(pixels):
        print(f"Adding {direction} seam No.{i + 1}")
        seam = lst_seam.pop(0)

        # Fill seam in the img_copy with red pixel value
        img_copy = fill_seam(seam, img_copy)
        lst_seam = update_seams(lst_seam, seam, direction)

    return img_copy

def update_seams(remaining_seam, popped_seam, direction):

    updated_lst_seam = []

    if direction == 'vertical':
        for seam in remaining_seam:
            for i in range(len(popped_seam)):
                if popped_seam[i][0] < seam[i][0]:
                    seam[i] = list(seam[i])
                    seam[i][0] += 1
                    seam[i] = tuple(seam[i])
            updated_lst_seam.append(seam)

    elif direction == 'horizontal':
        for seam in remaining_seam:
            for i in range(len(popped_seam)):
                if popped_seam[i][1] < seam[i][1]:
                    seam[i] = list(seam[i])
                    seam[i][1] += 1
                    seam[i] = tuple(seam[i])
            updated_lst_seam.append(seam)

    return updated_lst_seam

def energy(img):

    # Used Sobel operator for energy gradient at each pixel point in img
    img_gradient = filters.sobel(color.rgb2gray(img))

    return img_gradient

def seam_carving_reducing(img,direction,pixels):

    # Repeatedly remove seam in each loop
    for i in range(pixels):
        print(f"Removing {direction} seam No.{i+1}")

        # Calculate img energy and assign to each pixel point
        pixel_energies = energy(img)

        # Calculate list of arrays of minimum energy at each pixel point
        seam_energies = min_seam_energy(pixel_energies, direction)

        # Find the seam path that goes from top to bottom or left to right
        seam = retrieve_seam(seam_energies, direction)

        #Return new image with seam removed
        img = remove_seam(seam, img, direction)

    return img

def seam_carving_increasing(img,direction,pixels):

    # img is for finding first n ( in here is pixels) seams to be removed, img_copy is for adding seams
    img_copy = np.copy(img)
    lst_seam = []

    for i in range(pixels):
        print(f"Looking for {direction} seam No.{i + 1}")

        # Calculate img energy and assign to each pixel point
        pixel_energies = energy(img)

        # Calculate list of arrays of minimum energy at each pixel point
        seam_energies = min_seam_energy(pixel_energies, direction)

        # Find the seam path that goes from top to bottom or left to right
        seam = retrieve_seam(seam_energies, direction)

        # Collect all the seam path
        lst_seam.append(seam)

        # Remove the seam and change img
        img = remove_seam(seam, img, direction)

    for i in range(pixels):
        print(f"Adding {direction} seam No.{i + 1}")
        seam = lst_seam.pop(0)

        # Add seams in order as of removing them, and then update their coordinates if necessary
        img_copy = add_seam(seam, img_copy, direction)
        lst_seam = update_seams(lst_seam, seam, direction)

    return img_copy

if __name__ == '__main__':

    print("\nHello this is a program for content-aware image resizing\n" )
    print('*-' * 60)
    print()

    # Ask user for inputs and parameters
    pic = input("Please type name of your image (Ex: surfer.jpg):")
    operation = input("What kind of operation do you need (Ex: increase/decrease):")
    pixels = input("How much in pixel value do you want to adjust image (Note: integer):")
    direction = input("Which dimension do you want to adjust (Ex: height/width):")
    mark = input("Do you need seam marks on a copy of the image? (y/n):")
    print()

    # read the input image
    img = imageio.imread(pic).astype(np.float32)

    # Determine seam direction from operation direction
    h = 'horizontal'
    v = 'vertical'
    if direction == 'height':
        seam_direction = h
    elif direction == 'width':
        seam_direction = v

    # Determine which operation function to use
    if operation == 'increase':
        img_resized = seam_carving_increasing(img, seam_direction, int(pixels))
    elif operation == 'decrease':
        img_resized = seam_carving_reducing(img, seam_direction, int(pixels))

    # Display user inputs
    print("\nSummary:")
    print(f"Image: {pic}, operation: {operation}, direction: {direction}, pixels: {pixels}")
    print(f"Image original size is: {img.shape[0]}x{img.shape[1]}")
    print(f"Resized image is: {img_resized.shape[0]}x{img_resized.shape[1]}")

    # Display original image and resized image
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(1,2,1)
    original_img_plot = plt.imshow(img.astype(np.uint8))
    plt.title('Original Image')
    fig.add_subplot(1,2,2)
    resized_img_plot = plt.imshow(img_resized.astype(np.uint8))
    plt.title('Resized Image')
    plt.show()

    # Save resized image in default pattern name
    img_name = pic.split(".")[0]
    output_name_tuple = (operation, img_name, direction, pixels, '.jpg')
    output_name = "_".join(output_name_tuple)
    img_resized = img_resized.astype(np.uint8)
    imageio.imwrite(output_name, img_resized)
    print(f"Image saved as {output_name}")
    print()

    # If img with seam marks needed:
    if mark == 'y':
        img_mark = seam_mark(img, seam_direction, int(pixels))

        img_mark_name = 'seam_' + seam_direction + '_' + pixels + '.jpg'
        print(f"\nSeam marked on image saved as {img_mark_name}")
        img_mark = img_mark.astype(np.uint8)
        imageio.imwrite(img_mark_name, img_mark)

        plt.figure()
        plt.title('Processed Image with Seam-Carving filling lines')
        plt.imshow(img_mark)
        plt.show()

















