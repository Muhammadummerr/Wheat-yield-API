import cv2

def stitch_images(image_paths):
    images = [cv2.imread(p) for p in image_paths]
    if len(images) < 2:
        return images[0]
    stitcher = cv2.Stitcher_create()
    status, stitched = stitcher.stitch(images)
    if status != cv2.Stitcher_OK:
        raise RuntimeError("Stitching failed")
    return stitched
