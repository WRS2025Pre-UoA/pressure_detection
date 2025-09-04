import cv2
import numpy as np
import math

# --- 画像読み込み ---
img = cv2.imread("pic_1.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- 円検出（中心と半径） ---
gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
circles = cv2.HoughCircles(
    gray_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
    param1=100, param2=30, minRadius=50, maxRadius=200
)

if circles is None:
    raise RuntimeError("メーターの円が検出できませんでした")

circles = np.round(circles[0, :]).astype("int")
x_center, y_center, r = circles[0]

# 円と中心を描画
cv2.circle(img, (x_center, y_center), r, (0, 255, 0), 2)
cv2.circle(img, (x_center, y_center), 2, (0, 0, 255), 3)

# --- エッジ検出 ---
edges = cv2.Canny(gray, 50, 150)

# --- マスク（リング状） ---
mask = np.zeros_like(edges)
cv2.circle(mask, (x_center, y_center), int(r * 0.9), 255, -1)
cv2.circle(mask, (x_center, y_center), int(r * 0.35), 0, -1)
needle_area = cv2.bitwise_and(edges, mask)

# --- HoughLinesPで直線検出 ---
lines = cv2.HoughLinesP(
    needle_area, 1, np.pi / 180, threshold=30,
    minLineLength=int(r * 0.25), maxLineGap=15
)

if lines is not None:
    farthest_point = None
    max_dist = -1

    for l in lines:
        x1, y1, x2, y2 = l[0]
        d1 = math.hypot(x1 - x_center, y1 - y_center)
        d2 = math.hypot(x2 - x_center, y2 - y_center)

        root = (x1, y1) if d1 < d2 else (x2, y2)
        tip  = (x2, y2) if d1 < d2 else (x1, y1)

        if math.hypot(root[0] - x_center, root[1] - y_center) < r * 0.4:
            if max(d1, d2) > max_dist:
                farthest_point = tip
                max_dist = max(d1, d2)

    if farthest_point is not None:
        cv2.line(img, (x_center, y_center), farthest_point, (0, 0, 255), 2)

# --- 結果表示 ---
cv2.imshow("Needle Detection", img)
cv2.imshow("Edges", needle_area)
cv2.waitKey(0)
cv2.destroyAllWindows()
