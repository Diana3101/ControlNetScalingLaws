import numpy as np
import torch
from kornia.augmentation import RandomErasing, RandomThinPlateSpline
from PIL import Image
import torchvision.transforms.functional as tvf
from torchvision import transforms

# cut the part of the circle (RandomErasing)
# def get_erased_image(image: torch.Tensor):
#     """
#     - scale: range of proportion of erased area against input image.
#     - ratio: range of aspect ratio of erased area.
#     """
#     aug = RandomErasing(scale=(0.2, 0.2), ratio=(1.0, 1.0), p=1.0, 
#                         value=0.0, same_on_batch=False, keepdim=True)
#     aug_image = aug(image)

#     n_ = 0
#     while (aug_image == image).all() or \
#     (int(torch.sum(aug_image != image)) < 1000 or int(torch.sum(aug_image != image)) > 2000):
#         aug_image = aug(image)
        
#         n_ += 1
#         if n_ > 50:
#             # import pdb; pdb.set_trace()
#             raise Exception("RandomErasing didn't cut part of the circle! \
#             Try other parameter for the RandomErasing!")

#     # aug_image_pil = tvf.to_pil_image(aug_image)
#     # aug_image_pil.save(f'img_{idx}.png')
#     # print(f"{idx} image: changed pixels = {int(torch.sum(aug_image != image))}")
#     return aug_image

# cut the part of the circle (manually)
def get_erased_image(image: torch.Tensor, mode: str):
    # 3 channels in black&white image are the same, so take only the 1-st channel for simplicity
    image = image[0]
    edge_idxs = torch.argwhere(image > 0.5)

    if mode == 'slightly':
        n_idxs_to_modify = 500
    elif mode == 'hard':
        # erase 60% of the circle
        n_idxs_to_modify = int(len(edge_idxs) * 0.6)

    n_ = int(len(edge_idxs) / n_idxs_to_modify)
    if n_ == 0:
        edge_idxs_to_cut = edge_idxs[:n_idxs_to_modify]
    else:
        start_i = np.random.randint(n_, size=1)[0]
        edge_idxs_to_cut = edge_idxs[start_i*n_idxs_to_modify : (start_i+1)*n_idxs_to_modify]

    linear_idxs = edge_idxs_to_cut[:, 0] * image.shape[0] + edge_idxs_to_cut[:, 1]
    aug_image = torch.reshape(image, (-1,)).detach().clone()
    aug_image[linear_idxs] = 0
    aug_image = torch.reshape(aug_image, image.shape)
    aug_image = aug_image.expand(3, aug_image.shape[0], aug_image.shape[1])
    
    # idx = np.random.randint(10, size=1)[0]
    # aug_image_pil = tvf.to_pil_image(aug_image)
    # aug_image_pil.save(f'test_prob_corruption/img_erased_{idx}.png')

    return aug_image

def get_erased_image_by_fraction(image: torch.Tensor, part_to_erase: float):
    if part_to_erase == 0:
        return image
    # 3 channels in black&white image are the same, so take only the 1-st channel for simplicity
    image = image[0]
    edge_idxs = torch.argwhere(image > 0.5)

    n_idxs_to_modify = int(len(edge_idxs) * part_to_erase)

    n_ = int(len(edge_idxs) / n_idxs_to_modify)
    if n_ == 0:
        edge_idxs_to_cut = edge_idxs[:n_idxs_to_modify]
    else:
        start_i = np.random.randint(n_, size=1)[0]
        edge_idxs_to_cut = edge_idxs[start_i*n_idxs_to_modify : (start_i+1)*n_idxs_to_modify]

    linear_idxs = edge_idxs_to_cut[:, 0] * image.shape[0] + edge_idxs_to_cut[:, 1]
    aug_image = torch.reshape(image, (-1,)).detach().clone()
    aug_image[linear_idxs] = 0
    aug_image = torch.reshape(aug_image, image.shape)
    aug_image = aug_image.expand(3, aug_image.shape[0], aug_image.shape[1])
    
    # aug_image_pil = tvf.to_pil_image(aug_image)
    # aug_image_pil.save(f'corrupted_images_by_part/img_erased_{part_to_erase}.png')

    return aug_image


# add some noise (Gaussian - RandomPlasmaShadow / RandomGaussianNoise / RandomThinPlateSpline)
def get_noisy_image(image: torch.Tensor):
    aug = RandomThinPlateSpline(scale=0.4, align_corners=False, same_on_batch=False, p=1.0, keepdim=True)
    aug_image = aug(image)

    # idx = np.random.randint(10, size=1)[0]
    # aug_image_pil = tvf.to_pil_image(aug_image)
    # aug_image_pil.save(f'corrupted_images_hard/img_noisy_{idx}.png')
        
    return aug_image


def get_corrupted_image(image: torch.Tensor, mode: str):
    image = get_erased_image(image=image, mode=mode)
    if mode == 'slightly':
        return image
    elif mode == 'hard':
        return image
        # return get_noisy_image(image)


if __name__ == "__main__":
    # save slightly corrupted validation images
    for i in range(1, 6):
        validation_image_path = f"validation_set/conditioning_image_{i}.png"
        validation_image = Image.open(validation_image_path).convert("RGB")
        conditioning_image_transforms = transforms.Compose([transforms.ToTensor()])
        validation_image_tensor = conditioning_image_transforms(validation_image)
        validation_image_tensor = get_corrupted_image(validation_image_tensor, mode='slightly')
        validation_image = tvf.to_pil_image(validation_image_tensor)
        validation_image.save(f'validation_set-slightly_corrupted/conditioning_image_{i}.png')

    # save hard corrupted validation images
    for i in range(1, 6):
        validation_image_path = f"validation_set/conditioning_image_{i}.png"
        validation_image = Image.open(validation_image_path).convert("RGB")
        conditioning_image_transforms = transforms.Compose([transforms.ToTensor()])
        validation_image_tensor = conditioning_image_transforms(validation_image)
        validation_image_tensor = get_corrupted_image(validation_image_tensor, mode='hard')
        validation_image = tvf.to_pil_image(validation_image_tensor)
        validation_image.save(f'validation_set-hard_corrupted/conditioning_image_{i}.png')
