import os, glob, argparse
import numpy as np, cv2 as cv
from ultralytics import YOLO
from pathlib import Path

def rasterize(img_shape, dets, pad=2, conf_thr=0.3, max_det=None):
    H,W = img_shape[:2]
    m = np.zeros((H,W), np.uint8); used = 0
    for b in dets:
        if float(b.conf) < conf_thr: continue
        x1,y1,x2,y2 = map(int, b.xyxy.cpu().numpy().ravel())
        x1 = max(0, x1-pad); y1 = max(0, y1-pad)
        x2 = min(W-1, x2+pad); y2 = min(H-1, y2+pad)
        m[y1:y2+1, x1:x2+1] = 255
        used += 1
        if max_det and used>=max_det: break
    return m

def main():
    ap = argparse.ArgumentParser("Predict UI boxes on cropped images -> raster masks")
    ap.add_argument("--images", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--pad", type=int, default=2)
    ap.add_argument("--dilate_k", type=int, default=3)
    ap.add_argument("--dilate_it", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model = YOLO(args.weights)
    files = []
    for e in ("*.png","*.jpg","*.jpeg","*.tif","*.tiff","*.bmp"):
        files += glob.glob(os.path.join(args.images, e))
    files.sort()

    for i, p in enumerate(files, 1):
        im = cv.imread(p, cv.IMREAD_COLOR)
        H,W = im.shape[:2]
        base = Path(p).stem
        res = model.predict(source=im, verbose=False, conf=args.conf, imgsz=640, max_det=50)[0]
        m = rasterize(im.shape, res.boxes, pad=args.pad, conf_thr=args.conf, max_det=None)
        if args.dilate_k>0 and args.dilate_it>0:
            ker = cv.getStructuringElement(cv.MORPH_ELLIPSE, (args.dilate_k, args.dilate_k))
            m = cv.dilate(m, ker, iterations=args.dilate_it)
        outp = os.path.join(args.out, base + "_ui_mask.png")
        cv.imwrite(outp, m)
        print(f"[{i}/{len(files)}] {base} -> {outp}")

if __name__ == "__main__":
    main()
