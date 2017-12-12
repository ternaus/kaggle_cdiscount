from pathlib import Path
import cv2
from joblib import Parallel, delayed
from skimage import measure
from tqdm import tqdm
import numpy as np
import datetime

print('[{}] Creating train file list...'.format(str(datetime.datetime.now())))
train_imgs = list(Path('../data/train').glob('**/*.jpg'))


def crop_image(file_name):
    img = cv2.imread(str(file_name))
    mask = (img.mean(axis=2) < 250).astype(np.uint8)
    try:
        labeled_img = measure.label(mask)
    except:
        return None

    props = measure.regionprops(labeled_img)
    if len(props) == 0:
        return 0
    max_area_index = np.argmax([x.area for x in props])
    max_prop = props[max_area_index]
    y_min, x_min, y_max, x_max = max_prop.bbox
    height = y_max - y_min
    width = x_max - x_min

    if height > width:
        new_img = np.ones((height, height, 3)) * 255
        shift = int((height - width) / 2)
        new_img[:, shift:shift + width] = img[y_min:y_max, x_min:x_max]
    else:
        new_img = np.ones((width, width, 3)) * 255
        shift = int((width - height) / 2)
        new_img[shift:shift + height] = img[y_min:y_max, x_min:x_max]

    return new_img.astype(np.uint8)


def helper(file_name):
    new_img = crop_image(file_name)
    if not None:
        new_path = str(file_name).replace('train', 'train_cropped')
        Path(new_path).parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(new_path, new_img)


print('[{}] Cropping...'.format(str(datetime.datetime.now())))

result = [helper(x) for x in tqdm(train_imgs)]
# result = Parallel(n_jobs=8)(delayed(helper)(x) for x in train_imgs)
