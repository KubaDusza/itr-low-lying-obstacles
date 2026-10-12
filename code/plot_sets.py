import cv2, json, numpy as np, os
OWN, PRED, OUT = 'data/OWN', 'data/OWN_pred', 'figures/dataset'
meta = {r['run']: r for r in json.load(open(f'{OWN}/gt_summary.json'))}
MODELS = [("YOLOv8n-seg","yolo"), ("U-Net","unet_mobilenet_v2"), ("DeepLabV3+","deeplabv3plus_mobilenet_v2")]
SETS = ["cables","tapes","hexkeys","random","screws","chess","rags"]
BLOCKS = [("h52","floor"),("h52","poster"),("h25","floor"),("h25","poster")]
DIST = [("075","0.87 m"),("150","1.56 m"),("250","2.60 m")]
SC = 0.62
def tile(im): return cv2.resize(im, (int(640*SC), int(362*SC)), interpolation=cv2.INTER_AREA)
for s in SETS:
    rows = []
    for h, bg in BLOCKS:
        for d, dl in DIST:
            run = f"{h}_{bg}_{s}_{d}"
            if not os.path.isdir(f'{OWN}/{run}/rgb'): continue
            ok = meta.get(run, {}).get('reliable', False)
            img = cv2.imread(f'{OWN}/{run}/rgb/2.png')
            fl = cv2.imread(f'{OWN}/{run}/floor.png', 0) > 0
            gt = (cv2.imread(f'{OWN}/{run}/gt/2.png', 0) > 0) if ok else None
            g = img.copy()
            if ok: g[gt] = (0, 0, 255)
            else:  cv2.putText(g, "mask not reliable", (12, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,215,255), 2)
            row = [tile(img), tile(g)]
            for _, k in MODELS:
                p = (cv2.imread(f'{PRED}/{k}/{run}/2.png', 0) > 0) & fl
                v = img.copy()
                if ok: v[p & gt]=(80,200,80); v[gt & ~p]=(0,0,255); v[p & ~gt]=(0,140,255)
                else:  v[p] = (230, 120, 20)
                row.append(tile(v))
            band = np.concatenate(row, 1)
            lab = np.full((band.shape[0], 150, 3), 248, np.uint8)
            cv2.putText(lab, f"{h[1:]} cm", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (26,26,26), 2)
            cv2.putText(lab, bg, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90,90,90), 1)
            cv2.putText(lab, dl, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (26,26,26), 2)
            rows.append(np.concatenate([lab, band], 1))
    if not rows: continue
    out = np.concatenate(rows, 0)
    hdr = np.full((38, out.shape[1], 3), 248, np.uint8)
    for j, c in enumerate(["image","ground truth"]+[m for m,_ in MODELS]):
        cv2.putText(hdr, c, (150 + j*int(640*SC) + 8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (26,26,26), 2)
    cv2.imwrite(f'{OUT}/pred_{s}.jpg', np.concatenate([hdr, out], 0), [cv2.IMWRITE_JPEG_QUALITY, 84])
    print(f'pred_{s}.jpg  {len(rows)} rows')
