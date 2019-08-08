import numpy as np

def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
    def eraser(input_img):
        img_h, img_w, img_c = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        else:
            c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w, :] = c

        return input_img

    return eraser

def standard_scaling(X_train, X_test):
    X_train_mean = np.mean(X_train, axis=(0,1,2))
    X_train_std = np.std(X_train, axis=(0,1,2))
    X_train = (X_train - X_train_mean) / X_train_std
    X_test = (X_test - X_train_mean) / X_train_std
    return X_train, X_test

def random_pad_crop(img, pad_size):
    # Note: image_data_format is 'channel_last'
    assert img.shape[2] == 3

    shape = img.shape
    x = np.pad(img, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='reflect')
    height, width = x.shape[0], x.shape[1]
    dy, dx = shape[0], shape[1]
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y+dy), x:(x+dx), :]

def cutout(img, max_holes=1, max_length=4):
    height = img.shape[1]
    width = img.shape[2]

    # normalize before adding the mask
    mean = img.mean(keepdims=True)
    img -= mean

    mask = np.ones((height, width), np.float32)
    nb_holes = np.random.randint(0, max_holes + 1)

    for i in range(nb_holes):
        y = np.random.randint(height)
        x = np.random.randint(width)
        length = 4 * np.random.randint(1, max_length + 1)

        y1 = np.clip(y - length // 2, 0, height)
        y2 = np.clip(y + length // 2, 0, height)
        x1 = np.clip(x - length // 2, 0, width)
        x2 = np.clip(x + length // 2, 0, width)

        mask[y1: y2, x1: x2] = 0.

    # apply mask
    img = img * mask

    # denormalize
    img += mean

    return img
