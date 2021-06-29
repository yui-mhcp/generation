import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from matplotlib import colors

from utils.generic_utils import to_json
from utils.plot_utils import plot, plot_multiple
from utils.image.mask_utils import apply_mask
from utils.image.image_utils import _normalize_color
from utils.image.image_io import load_image, get_image_size

NORMALIZE_NONE  = 0
NORMALIZE_01    = 1
NORMALIZE_WH    = 2

CIRCLE      = CERCLE    = 0
ELLIPSE     = OVALE     = 1
RECTANGLE   = RECT      = 2

class BoundingBox:
    def __init__(self, x1, y1, x2 = None, y2 = None, w = None, h = None, 
                 classes = None, conf = 1., labels = None):
        assert x2 is not None or w is not None
        assert y2 is not None or h is not None
        
        self.x1 = max(x1, 0.)
        self.y1 = max(y1, 0.)
        
        self.x2 = x2 if x2 is not None else x1 + w
        self.y2 = y2 if y2 is not None else y1 + h
        
        self.w  = self.x2 - self.x1
        self.h  = self.y2 - self.y1
        
        self.conf   = conf
        self.classes = classes
        
        self.c = np.argmax(classes) if isinstance(classes, (np.ndarray, tf.Tensor)) else classes
        if self.c is None: self.c = 0
        
        self.label  = labels[self.c] if labels is not None else self.c
    
    @property
    def p(self):
        if self.classes is None: return 1.
        return self.classes[self.c] if not isinstance(self.classes, (int, np.integer)) else 1.
    
    @property
    def area(self):
        return self.w * self.h
    
    @property
    def score(self):
        return self.p * self.conf
    
    @property
    def rectangle(self):
        return [self.x1, self.y1, self.x2, self.y2]
    
    @property
    def box(self):
        return [self.x1, self.y1, self.w, self.h, self.c]
    
    def __str__(self):
        return "{:.4f} {:.4f} {:.4f} {:.4f} {} {:.4f}".format(*self.box, self.score)
    
    def json(self, labels = None):
        infos = {
            'xmin' : self.x1,
            'ymin' : self.y1,
            'xmax' : self.x2,
            'ymax' : self.y2,
            'label' : self.label
        }
        label = labels[self.c] if labels is not None else self.label
        if label is not None: infos['label'] = label
        return to_json(infos)
    
    def to_image(self, image_w, image_h):
        return [self.x1 * image_w, self.y1 * image_h, self.x2 * image_w, self.y2 * image_h]
        
def bbox_iou(box1, box2):
    xmin1, ymin1, w1, h1 = get_box_pos(box1)
    xmax1, ymax1 = xmin1 + w1, ymin1 + h1
    xmin2, ymin2, w2, h2 = get_box_pos(box2)
    xmax2, ymax2 = xmin2 + w2, ymin2 + h2
    
    intersect_w = _interval_overlap([xmin1, xmax1], [xmin2, xmax2])
    intersect_h = _interval_overlap([ymin1, ymax1], [ymin2, ymax2])  
    
    intersect = intersect_w * intersect_h
    
    union = w1*h1 + w2*h2 - intersect
    
    return float(intersect) / union

def compute_overlap(a, b):
    """
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua  
    
def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap      
        
def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2,x4) - x3          

def _is_box_list(box):
    if isinstance(box, (BoundingBox, dict)): return False
    if isinstance(box, (tuple, list)):
        if len(box) not in (4, 6): return True
        if isinstance(box[0], (int, float, np.integer, np.floating)): return False
        elif isinstance(box[0], (list, dict, BoundingBox)): return True
    elif isinstance(box, np.ndarray):
        if box.ndim == 1: return False
        elif box.ndim == 2: return True
        else:
            raise ValueError("Invalid box shape : {}".format(box.shape))
    else:
        raise ValueError("Type de box inconnu : {}\n{}".format(type(box), box))
        
    
def get_box_pos(box, image = None, image_h = None, image_w = None,
                box_mode = 0, dezoom_factor = 1., 
                with_label = False, labels = None, normalize_mode = NORMALIZE_NONE):
    """
        arg : 
            - box : soit BoundBox, dict (avec 'xmin', 'ymin', 'xmax', 'ymax') ou liste
            - box_mode : Seulement utile si box est de type 'list'
                Si box_mode == 0: box = [x, y, w, h]
                Si box_mode == 1: box = [x0, y0, x1, y1]
        return : 
            tuple (x, y, w, h) si with_label == False
            tuple (x, y, w, h, label, score) si with_label == True
    """
    if image is not None:
        image_h, image_w = get_image_size(image)
    if isinstance(box, BoundingBox):
        x1, y1, x2, y2 = box.rectangle
        w, h = x2 - x1, y2 - y1
        score = box.score
        label = box.label
    elif isinstance(box, (list, tuple)):
        if box_mode == 0:
            x1, y1, w, h = box[:4]
        else:
            x1, y1, x2, y2 = box[:4]
            w, h = x2 - x1, y2 - y1
        label = box[4] if len(box) > 4 else None
        score = 1.
    elif isinstance(box, dict):
        x1, y1 = box['xmin'], box['ymin']
        x2, y2 = box['xmax'], box['ymax']
        w, h = x2 - x1, y2 - y1
        label = box['name'] if 'name' in label else box['label']
        score = 1
    else:
        print("Box {} n'est pas du bon format ! ".format(box))
        return 0, 0, 0, 0
    
    new_w = w * dezoom_factor
    new_h = h * dezoom_factor
    x1 = x1 - ((new_w - w) / 2.)
    y1 = y1 - ((new_h - h) / 2.)
    w, h = new_w, new_h
    
    x2, y2 = x1 + w, y1 + h
        
    if image_h is not None and image_w is not None and normalize_mode != NORMALIZE_NONE:
        is_01 = x1 <= 1.1 and x2 <= 1.1 and y1 <= 1.1 and y2 <= 1.1
        
        if normalize_mode == NORMALIZE_01 and not is_01:
            x1 = max(0., x1 / image_w)
            x2 = min(1., x2 / image_w)
            y1 = max(0., y1 / image_h)
            y2 = min(1., y2 / image_h)
        elif normalize_mode == NORMALIZE_WH:
            if is_01:
                x1 = max(0, int(x1 * image_w))
                x2 = min(image_w, int(x2 * image_w))
                y1 = max(0, int(y1 * image_h))
                y2 = min(image_h, int(y2 * image_h))
            else:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        else:
            raise ValueError("Mode de normalisation inconnu : {} !".format(normalize_mode))
    
    w, h = x2 - x1, y2 - y1
    
    if with_label:
        if labels is not None and isinstance(label, (int, np.integer)):
            label = labels[label] if label < len(labels) else None
        
        return x1, y1, w, h, label, float(score)
    return x1, y1, w, h
    
def get_box_area(box, **kwargs):
    _, _, w, h = get_box_pos(box, **kwargs)
    return w * h

def crop_box(filename, box, show = False, **kwargs):
    image = load_image(filename)
    image_h, image_w = get_image_size(image)
    
    x, y, w, h, label, score = get_box_pos(
        box, image = image, with_label = True, 
        normalize_mode = NORMALIZE_WH, **kwargs
    )
    
    box_image = image[y : y + h, x : x + w]
    
    if show:
        plot(box_image)
        
    return box_image, [x, y, w, h, label, score]

def draw_boxes(filename,
               boxes,
               shape    = RECTANGLE,
               color    = 'r',
               thickness    = 3, 
               use_label    = False,
               labels   = None, 
               vertical = True,
               ** kwargs
              ):
    assert shape in (CERCLE, ELLIPSE, RECTANGLE)

    image = load_image(filename).numpy()
    image_h, image_w, _ = image.shape
    
    if not isinstance(color, list): color = [color]
    color = [_normalize_color(c, image = image) for c in color]
    label_color = {}
    
    if not _is_box_list(boxes): boxes = [boxes]
    
    for i, box in enumerate(boxes):
        normalized_box = get_box_pos(
            box, image = image, with_label = use_label, labels = labels,
            normalize_mode = NORMALIZE_WH, ** kwargs
        )
        
        x, y, w, h = normalized_box[:4]
        center_x, center_y = int(x + w/2), int(y + h/2)
        c = color[i % len(color)]
        
        if use_label:
            label, conf = normalized_box[4:6]
            if label not in label_color: 
                label_color[label] = color[len(label_color) % len(color)]
            c = label_color[label]
            
            image = cv2.putText(
                image, "{} ({}%)".format(label, int(conf * 100)), 
                (x, y - 13), cv2.FONT_HERSHEY_SIMPLEX, 
                1e-3 * image_h, c, 3
            )
        
        if shape == RECTANGLE:
            image = cv2.rectangle(image, (x,y), (x+w, y+h), c, thickness)
        elif shape == CERCLE:
            image = cv2.circle(image, (center_x, center_y), min(w, h)//2, c, thickness)
        elif shape == ELLIPSE:
            axes = (w // 2, int(h / 1.5)) if vertical else (int(w / 1.5), h // 2)
            image = cv2.ellipse(
                image, angle = 0, startAngle = 0, endAngle = 360, 
                center = (center_x, center_y), axes = axes,
                color = c, thickness = thickness
            )
    
    return image

def box_as_mask(filename, boxes, mask_background = False, ** kwargs):
    image = load_image(filename).numpy()
    image_h, image_w, _ = image.shape
    
    mask = tf.zeros((image_h, image_w, 3), dtype = tf.float32)
    
    mask = draw_boxes(mask, boxes, color = 255, thickness = -1, ** kwargs)
    
    mask = mask[...,:1] > 0.
    
    if mask_background: mask = ~mask
    
    return mask

def mask_boxes(filename, boxes, shape = RECTANGLE, dezoom_factor = 1., **kwargs):
    image = load_image(filename)
    
    mask = box_as_mask(image, boxes, shape = shape, dezoom_factor = dezoom_factor)
    
    return apply_mask(image, mask, **kwargs)

def show_boxes(filename, boxes, labels = None, dezoom_factor = 1., ** kwargs):
    image = load_image(filename).numpy()
    image_h, image_w = get_image_size(image)
    
    pairs = []
    labels_nb = {}
    
    if not _is_box_list(boxes): boxes = [boxes]
    
    for box in boxes:
        box_img, (x1, y1, w, h, label, score) = crop_box(
            image, box, labels = labels,dezoom_factor = dezoom_factor
        )
                
        if isinstance(label, int) and labels is not None: 
            label = labels[label]
        
        if label not in labels_nb: labels_nb[label] = 0
        labels_nb[label] += 1
        
        box_name = "{}_{} ({:.2f}%)".format(label, labels_nb[label], score*100)
        
        pairs.append((box_name, box_img))
    
    plot_multiple(* pairs, use_subplots = True, plot_type='imshow', **kwargs)

def save_boxes(filename, boxes, labels, append = True, ** kwargs):
    """
        Save boxes to a .txt file with format `x y w h label`
        
        Arguments :
            - filename  : the image filename
            - boxes     : list of boxes
            - labels    : labels for boxes
            - append    : whether to overwrite or append at the end of the file
    """
    open_mode = 'a' if append else 'w'
    
    image_w, image_h = get_image_size(filename)

    description = '{}\n{}\n'.format(filename, len(boxes))
    for box in boxes:
        x, y, w, h, label, score = get_box_pos(
            box, image_h = image_h, image_w = image_w, with_label = True, labels = labels,
            normalize_mode = NORMALIZE_WH, ** kwargs
        )
            
                
        description += "{} {} {} {} {}\n".format(x, y, w, h, label)

    with open(filename, open_mode) as fichier:
        fichier.write(description)
