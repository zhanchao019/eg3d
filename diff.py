
from PIL import Image
import os
from tqdm import tqdm
def get_data_path(root='examples'):
    im_path = [os.path.join(root, i) for i in sorted(os.listdir(root)) if i.endswith('png') ]
    return im_path

def join(png1, png2, spacing, save_position, flag='horizontal'):
    """
    :param png1: png1 path 
    :param png2: png2 path
    :param spacing: spacing width, default 0
    :param save_position: save image
    :param flag: horizontal or vertical
    :return:
    """
    img1, img2 = Image.open(png1), Image.open(png2)
    size1, size2 = img1.size, img2.size
    if flag == 'horizontal':
        joint = Image.new('RGB', (size1[0]+size2[0] + spacing, size1[1]))
        loc1, loc2 = (0, 0), (size1[0] + spacing, 0)
        joint.paste(img1, loc1)
        joint.paste(img2, loc2)
        joint.save(save_position)
    elif flag == 'vertical':
        joint = Image.new('RGB', (size1[0], size1[1]+size2[1] + spacing))
        loc1, loc2 = (0, 0), (0, size1[1] + spacing)
        joint.paste(img1, loc1)
        joint.paste(img2, loc2)
        joint.save(save_position)


if __name__ == '__main__':
    generate=get_data_path('/home/zhanchao/Mdisk/ubuntu_workplace/github/eg3d/outImg')
    origin=get_data_path('/home/zhanchao/Mdisk/ubuntu_workplace/github/eg3d/dataset_preprocessing/drive/dataset')

    tmp = len(generate)
    for i in tqdm(range(tmp)):
        img1 = generate[i]
        img2 = origin[i]
        join(img1, img2, 10, f'./res/{i:04d}.png')
    # join(png, png, flag='vertical')
