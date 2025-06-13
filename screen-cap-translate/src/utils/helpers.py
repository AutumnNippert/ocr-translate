def nms(rois, thr=0.3):
    if not rois:
        return []
    b = np.asarray(rois, float)
    x1, y1, x2, y2 = b.T
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idx = area.argsort()[::-1]
    keep = []
    while idx.size:
        i = idx[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[idx[1:]])
        yy1 = np.maximum(y1[i], y1[idx[1:]])
        xx2 = np.minimum(x2[i], x2[idx[1:]])
        yy2 = np.minimum(y2[i], y2[idx[1:]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        iou = (w * h) / (area[i] + area[idx[1:]] - w * h)
        idx = idx[1:][iou < thr]
    return [tuple(map(int, b[i])) for i in keep]

def preprocess_fast(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def preprocess_deep(gray):
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(gray, 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)