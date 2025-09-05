import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def detect_pressure(img_path, min_value=0.0, max_value=1.6, show=True):
    # --- 画像読み込み ---
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- 円検出（メーター外枠） ---
    gb = cv2.GaussianBlur(gray, (5,5), 0)
    circles = cv2.HoughCircles(gb, cv2.HOUGH_GRADIENT,
                               dp=1.2, minDist=100,
                               param1=100, param2=30,
                               minRadius=0, maxRadius=360)
    x0, y0, r = np.round(circles[0,0]).astype(int)

    # ==============================================================
    # ① 最小値・最大値の検出（外周マスク + ヒストグラム）
    # ==============================================================
    edges = cv2.Canny(gray, 50, 150)
    mask_outer = np.zeros_like(edges)
    cv2.circle(mask_outer, (x0, y0), int(r * 0.91), 255, -1)
    cv2.circle(mask_outer, (x0, y0), int(r * 0.75), 0, -1)
    ticks_area = cv2.bitwise_and(edges, mask_outer)

    ys, xs = np.where(ticks_area > 0)
    angles = []
    for x, y in zip(xs, ys):
        dx, dy = x - x0, y0 - y
        ang = math.degrees(math.atan2(dy, dx))
        if ang < 0: ang += 360
        if ang >= 180:  # 下半分に限定
            angles.append(int(round(ang)) % 360)

    hist, _ = np.histogram(angles, bins=180, range=(180,360))
    kernel = np.ones(7)/7
    hist_smooth = np.convolve(hist, kernel, mode='same')

    # 谷検出（最大の空白区間）
    threshold = max(1, int(np.percentile(hist_smooth, 25)))
    below = hist_smooth <= threshold
    best_len, best_start, best_end = 0, None, None
    N = len(below)
    bb = np.concatenate([below, below])
    start=None
    for i,v in enumerate(bb):
        if v and start is None:
            start=i
        if (not v) and (start is not None):
            length=i-start
            if length>best_len:
                best_len=length; best_start=start; best_end=i-1
            start=None
    if start is not None:
        length=len(bb)-start
        if length>best_len:
            best_len=length; best_start=start; best_end=len(bb)-1

    min_angle_ccw=max_angle_ccw=None; gap=None
    if best_start is not None:
        a1=(best_start)%N; a2=(best_end)%N
        min_angle_ccw=180+min(a1,a2)
        max_angle_ccw=180+max(a1,a2)
        gap=(max_angle_ccw-min_angle_ccw)%360

    # ==============================================================
    # ② 針検出（内側マスク + HoughLinesP）
    # ==============================================================
    mask_inner = np.zeros_like(edges)
    cv2.circle(mask_inner, (x0, y0), int(r * 0.90), 255, -1)
    cv2.circle(mask_inner, (x0, y0), int(r * 0.35), 0, -1)
    needle_area = cv2.bitwise_and(edges, mask_inner)

    lines_needle = cv2.HoughLinesP(needle_area,1,np.pi/180,
                                   threshold=30,
                                   minLineLength=int(r*0.25),
                                   maxLineGap=15)
    needle_tip=None; max_dist=-1
    if lines_needle is not None:
        for l in lines_needle:
            x1,y1,x2,y2=l[0]
            d1=math.hypot(x1-x0,y1-y0); d2=math.hypot(x2-x0,y2-y0)
            root=(x1,y1) if d1<d2 else (x2,y2)
            tip=(x2,y2) if d1<d2 else (x1,y1)
            if math.hypot(root[0]-x0,root[1]-y0)<r*0.40:
                if max(d1,d2)>max_dist:
                    needle_tip=tip; max_dist=max(d1,d2)

    needle_angle_ccw=None
    if needle_tip is not None:
        dx,dy=needle_tip[0]-x0,y0-needle_tip[1]
        needle_angle_ccw=math.degrees(math.atan2(dy,dx))
        if needle_angle_ccw<0: needle_angle_ccw+=360

    # ==============================================================
    # ③ 圧力値算出（線形補間）
    # ==============================================================
    pressure_value=None
    if (needle_angle_ccw is not None) and (min_angle_ccw is not None) and (max_angle_ccw is not None):
        cw_range=(min_angle_ccw-max_angle_ccw)%360
        cw_from_min=(min_angle_ccw-needle_angle_ccw)%360
        if cw_range>0:
            frac=np.clip(cw_from_min/cw_range,0.0,1.0)
            pressure_value=min_value+frac*(max_value-min_value)

    # ==============================================================
    # ④ 結果の可視化
    # ==============================================================
    out=img.copy()
    def draw_angle_point(image,angle,color,scale=0.95):
        rad=math.radians(angle)
        px=int(x0+r*scale*math.cos(rad)); py=int(y0-r*scale*math.sin(rad))
        cv2.circle(image,(px,py),6,color,-1)

    if min_angle_ccw is not None: draw_angle_point(out,min_angle_ccw,(0,0,255))  # 青
    if max_angle_ccw is not None: draw_angle_point(out,max_angle_ccw,(255,0,0))  # 赤
    if needle_tip is not None: cv2.line(out,(x0,y0),needle_tip,(0,255,0),2)      # 緑

    if show:
        plt.figure(figsize=(8,8))
        plt.title(f"min={min_angle_ccw:.1f}°, max={max_angle_ccw:.1f}°, needle={needle_angle_ccw:.1f}°, value={pressure_value:.3f} MPa")
        plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

    return min_angle_ccw, max_angle_ccw, needle_angle_ccw, pressure_value

# 実行例
detect_pressure("/mnt/data/pic_0.png")
