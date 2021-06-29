import tensorflow as tf

def augment_image(img, transforms, prct = 0.25, ** kwargs):
    """
        Augment `img` by applying sequentially each `transforms` each with `prct` probability
        
        Arguments :
            - img   : the image to augment
            - transforms    : (list of) str / callable, transformations to apply
            - prct  : the probability to apply each transformation
            - kwargs    : kwargs passed to each transformation function
        Return :
            - transformed : (maybe) transformed image
        
        Supported transformations' names are in the `_image_augmentations_fn` variable which associate name with function.
        All functions have an unused `kwargs` argument which allows to pass kwargs for each transformation function without disturbing other. 
        
        Note that the majority of these available functions are simply the application of 1 or multiple `tf.image.random_*` function
    """
    if not isinstance(transforms, (list, tuple)): transforms = [transforms]
    for transfo in transforms:
        assert callable(transfo) or transfo in _image_augmentations_fn, "Unknown transformation !\n  Accepted : {}\n  Got : {}".format(tuple(_image_augmentations_fn.keys()), transfo)
        
        fn = transfo if callable(transfo) else _image_augmentations_fn[transfo]
        
        img = tf.cond(
            tf.random.uniform(()) < prct,
            lambda: fn(img, ** kwargs),
            lambda: img
        )
    return img

def flip_vertical(img, ** kwargs):
    return tf.image.flip_up_down(img)

def flip_horizontal(img, ** kwargs):
    return tf.image.flip_left_right(img)

def rotate(img, n = None, ** kwargs):
    if n is None: n = tf.random.uniform((), minval = 0, maxval = 4, dtype = tf.int32)
    return tf.image.rot90(img, n)

def zoom(img, min_factor = 0.8, ** kwargs):
    scales = list(np.arange(0.8, 1.0, 0.01))
    boxes = np.zeros((len(scales), 4))
    
    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        box[i] = [x1, y1, x2, y2]
        
    def random_crop(img):
        crops = tf.image.crop_and_resize([img], boxes = boxes, box_ind = np.zeros(len(scales)), crop_size = (32, 32))
        return crops[tf.random_uniform(shape = [], minval=0, maxval=len(scales), dtype = tf.int32)]
    
    return random_crop(img)

def noise(img, factor = 25., ** kwargs):
    noise = tf.random.normal(tf.shape(img)) / factor
    return tf.clip_by_value(img + noise, 0., 1.)

def quality(img, min_jpeg_quality = 25, max_jpeg_quality = 75, ** kwargs):
    return tf.image.random_jpeg_quality(img, min_jpeg_quality, max_jpeg_quality)

def color(img, ** kwargs):
    img = hue(img, ** kwargs)
    img = saturation(img, ** kwargs)
    img = brightness(img, ** kwargs)
    img = contrast(img, ** kwargs)
    return img

def hue(img, max_delta = 0.15, ** kwargs):
    return tf.image.random_hue(img, max_delta)

def saturation(img, lower = 0.5, upper = 2., ** kwargs):
    return tf.image.random_saturation(img, lower, upper)

def brightness(img, max_delta = 0.15, ** kwargs):
    return tf.clip_by_value(tf.image.random_brightness(img, max_delta), 0., 1.)

def contrast(img, lower = 0.5, upper = 1.5, ** kwargs):
    return tf.clip_by_value(tf.image.random_contrast(img, lower, upper), 0., 1.)


_image_augmentations_fn = {
    'flip_vertical'     : flip_vertical,
    'flip_horizontal'   : flip_horizontal,
    'rotate'            : rotate,
    
    'noise'     : noise,
    'quality'   : quality,
    
    'color'     : color,
    'hue'       : hue,
    'saturation'    : saturation,
    'brightness'    : brightness,
    'contrast'      : contrast
}
