import numpy as np
from PIL import Image
import cv2
import pandas as pd

cond_images_path = "fill50k/conditioning_images/"
target_images_path = "fill50k/images/"
prompts_path = "fill50k/train.jsonl"


def add_3_channel_color_condition(cond_image, target_image):
    '''
    cond_image - PIL RGB image (h,w,3), or GrayScale image (h,w)
    target_image - colored PIL RGB image (h,w,3)
    '''
    cond_image_array = np.array(cond_image)
    target_image_array = np.array(target_image)
    
    if len(cond_image_array.shape) == 3:
        cond_image_array_gs = np.array(cond_image.convert('L'))
    elif len(cond_image_array.shape) == 2:
        cond_image_array_gs = np.array(cond_image)

    edge_coordinates = np.argwhere(cond_image_array_gs == 255)

    point_inner_coordinates = (edge_coordinates[0] + edge_coordinates[-1]) / 2
    point_inner_coordinates = point_inner_coordinates.astype(int)

    color_values_inner = target_image_array[point_inner_coordinates[0],point_inner_coordinates[1]]

    try:
        point_outer_coordinates = np.argwhere(target_image_array[:,:, 0] != color_values_inner[0])[0]
    except:
        try:
            point_outer_coordinates = np.argwhere(target_image_array[:,:, 1] != color_values_inner[1])[0]
        except:
            try:
                point_outer_coordinates = np.argwhere(target_image_array[:,:, 2] != color_values_inner[2])[0]
            except:
                point_outer_coordinates = point_inner_coordinates
    
    color_values_outer = target_image_array[point_outer_coordinates[0], point_outer_coordinates[1]]

    # (256, 512, 3)
    cond_color_array_shape = (int(cond_image_array.shape[0] / 2), cond_image_array.shape[1], color_values_outer.shape[0])

    cond_color_array_upper = np.full(cond_color_array_shape, color_values_inner)
    cond_color_array_lower = np.full(cond_color_array_shape, color_values_outer)

    for i in range(color_values_outer.shape[0]):
        channel_upper = cond_color_array_upper[:,:, i]
        channel_lower = cond_color_array_lower[:,:, i]
        if np.unique(channel_upper)[0] == color_values_inner[i] and \
            np.unique(channel_lower)[0] == color_values_outer[i]:
                continue
        else:
            print('Incorrect condition color array filling !')

    cond_color_array = np.vstack((cond_color_array_upper, cond_color_array_lower))

    # (512, 512, 3)
    image = Image.fromarray(cond_color_array)
    return image

    # cond_image_color_array = np.dstack((cond_image_array, cond_color_array))
    # (512, 512, 6)
    # return cond_image_color_array

def get_validation_conditioning_colors(validation_set_len=5):
    idxs = np.random.randint(50000, size=5)
    metadata = pd.read_json(prompts_path, lines=True)

    validation_metadata = metadata.loc[idxs]
    validation_metadata['text'].to_csv('validation_set/prompts.csv', index=False)

    for i, idx in enumerate(idxs):
        cond_image_path = "fill50k/" + validation_metadata.loc[idx]['conditioning_image']
        cond_image = Image.open(cond_image_path)
        target_image_path = "fill50k/" + validation_metadata.loc[idx]['image']
        target_image = Image.open(target_image_path)
        color_image = add_3_channel_color_condition(cond_image=cond_image, 
                                                    target_image=target_image)

        color_image.save(f"validation_set/conditioning_color_{i+1}.png")


# get_validation_conditioning_colors()

# validation_relevant_colors_idxs = [19588, 43604, 24147, 16929, 22555]

# i = 0
# for idx in validation_relevant_colors_idxs:
#     i += 1
#     cond_image = Image.open(cond_images_path + f'{idx}.png')
    # target_image = Image.open(target_images_path + f'{idx}.png')
#     im = add_3_channel_color_condition(cond_image=cond_image, 
#                                     target_image=target_image)
#     im.save(f"conditioning_color_{i}.png")

def process_validation_target_image(image_path, true_colors_path, n):
    image = Image.open(image_path).convert('RGB')
    colors = Image.open(true_colors_path)

    colors_array = np.array(colors)
    colors_array = colors_array.astype(float)

    inner_color_array = colors_array[0, 0, :]
    outer_color_array = colors_array[511, 511, :]

    image_array = np.array(image)
    image_array = image_array.astype(float)

    assert image_array.shape == (512, 512, 3)
    image_array_512 = image_array

    # image_array_512 = cv2.resize(image_array, (512, 512), 
    #            interpolation = cv2.INTER_AREA)

    for i in range(image_array_512.shape[0]):
        for j in range(image_array_512.shape[1]):
            dist_to_inner = np.sum(np.abs(inner_color_array - image_array_512[i, j, :]))
            dist_to_outer = np.sum(np.abs(outer_color_array - image_array_512[i, j, :]))

            if dist_to_inner < dist_to_outer:
                image_array_512[i, j, :] = inner_color_array
            else:
                image_array_512[i, j, :] = outer_color_array

            # if n in [2,3,5]:
            #     if i < 5 or j < 5:
            #         image_array_512[i, j, :] = outer_color_array

    image_array_512 = image_array_512.astype(np.uint8)
    # assert image_array_512.shape == (512, 512, 3)

    image_output = Image.fromarray(image_array_512)
    image_output.save(f"validation_set/target_image_{n}.png")


# for n in range(1, 6):
#     process_validation_target_image(image_path=f'validation_set/target_image_{n}_src.png',
#                                     true_colors_path=f'validation_set/conditioning_color_{n}.png',
#                                     n=n)
