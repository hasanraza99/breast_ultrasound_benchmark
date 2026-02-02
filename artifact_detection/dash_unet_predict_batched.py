import os, argparse, time, glob
from pathlib import Path
import cv2 as cv, numpy as np
import torch, torch.nn as nn

def setup_fast():
    torch.backends.cudnn.benchmark = True
    try: torch.set_float32_matmul_precision("high")
    except: pass

def letterbox(im, size):
    H, W = im.shape[:2]
    s = min(size/H, size/W)
    nh, nw = int(round(H*s)), int(round(W*s))
    imr = cv.resize(im, (nw, nh), cv.INTER_LINEAR)
    top = (size-nh)//2; left = (size-nw)//2
    bottom = size-nh-top; right = size-nw-left
    imp = cv.copyMakeBorder(imr, top, bottom, left, right, cv.BORDER_CONSTANT, value=(0,0,0))
    return imp, (top,left,nh,nw), (H,W)

def conv_block(i,o):
    return nn.Sequential(
        nn.Conv2d(i,o,3,padding=1,bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o,o,3,padding=1,bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
    )

# ***** IMPORTANT: names match the training script *****
class UNetTiny(nn.Module):
    def __init__(self,in_ch=3,out_ch=1,base=32):
        super().__init__()
        self.enc1 = conv_block(in_ch, base);      self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(base, base*2);     self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(base*2, base*4);   self.pool3 = nn.MaxPool2d(2)
        self.enc4 = conv_block(base*4, base*8);   self.pool4 = nn.MaxPool2d(2)
        self.bott = conv_block(base*8, base*16)
        self.up4  = nn.ConvTranspose2d(base*16, base*8, 2, 2)
        self.dec4 = conv_block(base*16, base*8)
        self.up3  = nn.ConvTranspose2d(base*8, base*4, 2, 2)
        self.dec3 = conv_block(base*8, base*4)
        self.up2  = nn.ConvTranspose2d(base*4, base*2, 2, 2)
        self.dec2 = conv_block(base*4, base*2)
        self.up1  = nn.ConvTranspose2d(base*2, base, 2, 2)
        self.dec1 = conv_block(base*2, base)
        self.outc = nn.Conv2d(base, out_ch, 1)
    def forward(self,x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b  = self.bott(self.pool4(e4))
        d4 = self.up4(b); d4 = self.dec4(torch.cat([d4, e4], 1))
        d3 = self.up3(d4); d3 = self.dec3(torch.cat([d3, e3], 1))
        d2 = self.up2(d3); d2 = self.dec2(torch.cat([d2, e2], 1))
        d1 = self.up1(d2); d1 = self.dec1(torch.cat([d1, e1], 1))
        return self.outc(d1)

def list_images(folder):
    exts = (".png",".jpg",".jpeg",".bmp",".tif",".tiff")
    return [os.path.join(folder,f) for f in sorted(os.listdir(folder)) if f.lower().endswith(exts)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--th", type=float, default=0.35)
    ap.add_argument("--close_k", type=int, default=3)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--channels_last", action="store_true")
    args = ap.parse_args()

    setup_fast()
    if not os.path.isfile(args.weights):
        raise FileNotFoundError(f"weights not found: {args.weights}")
    os.makedirs(args.out, exist_ok=True)

    # load model with matching names
    ckpt = torch.load(args.weights, map_location="cpu")
    model = UNetTiny(base=32)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval().cuda()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    paths = list_images(args.images)
    if not paths:
        raise FileNotFoundError(f"No images found in {args.images}")
    print(f"[predict] found {len(paths)} images")
    print(f"[predict] out={args.out}  size={args.size}  batch={args.batch}  th={args.th}  close_k={args.close_k}")

    autocast = torch.amp.autocast
    t0 = time.time(); written = 0

    for i in range(0, len(paths), args.batch):
        chunk = paths[i:i+args.batch]
        ims, metas, origs, bases = [], [], [], []
        for p in chunk:
            im = cv.imread(p, cv.IMREAD_COLOR)
            if im is None: 
                print(f"[WARN] cannot read {p}"); 
                continue
            lb, meta, orig = letterbox(im, args.size)
            rgb = lb[:,:,::-1].astype(np.float32)/255.0
            t = torch.from_numpy(np.transpose(rgb,(2,0,1))).unsqueeze(0)
            if args.channels_last:
                t = t.to(memory_format=torch.channels_last)
            ims.append(t); metas.append(meta); origs.append(orig); bases.append(Path(p).stem)
        if not ims: 
            continue
        batch = torch.cat(ims,0).cuda(non_blocking=True)

        with torch.no_grad(), autocast("cuda", enabled=args.amp):
            probs = torch.sigmoid(model(batch)).detach().cpu().numpy()[:,0]

        for k, prob in enumerate(probs):
            mask = (prob > args.th).astype(np.uint8)*255
            if args.close_k > 0:
                K = cv.getStructuringElement(cv.MORPH_RECT,(args.close_k,args.close_k))
                mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, K, iterations=1)
            top,left,nh,nw = metas[k]; H0,W0 = origs[k]
            crop = mask[top:top+nh, left:left+nw]
            final = cv.resize(crop, (W0, H0), interpolation=cv.INTER_NEAREST)
            cv.imwrite(os.path.join(args.out, bases[k] + "_dash.png"), final)
            written += 1

    dt = time.time() - t0
    print(f"[predict] done: wrote {written} in {dt:.2f}s  ({dt/max(1,written):.4f}s/img)")
    sample = sorted(glob.glob(os.path.join(args.out, "*_dash.png")))[:5]
    for s in sample: print("  -", s)

if __name__ == "__main__":
    main()
