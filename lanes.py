import cv2
import numpy as np
import matplotlib as plt


def make_coordinates(img, line_parameters):
    slope, intercept = line_parameters
    y1 = img.shape[0]
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def canny(img):  # tạo hàm canny để detect cạnh đường
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # chuyển ảnh màu thành ảnh xám
    blurr = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blurr, 50, 150)
    return canny


def display_lines(img, lines):
    line_img = np.zeros_like(img)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            # for line in lines:
            # x1, y1, x2, y2 = line.reshape(4)

            cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_img


def ROI(img):
    height = img.shape[0]
    polygons = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    mask_img = cv2.bitwise_and(img, mask)
    return mask_img


def average_slope_intercept(img, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(img, left_fit_average)
    right_line = make_coordinates(img, right_fit_average)
    return np.array([left_line, right_line])


"""img = cv2.imread("test_image.jpg")
lane_img = np.copy(img)
cany = canny(lane_img)
crop_img = ROI(cany)
lines = cv2.HoughLinesP(
    crop_img,
    2,
    np.pi / 180,
    100,
    np.array([]),
    minLineLength=40,
    maxLineGap=5,
)
average_lines = average_slope_intercept(lane_img, lines)
line_img = display_lines(lane_img, average_lines)
combo_img = cv2.addWeighted(lane_img, 0.8, line_img, 1, 1)
cv2.imshow("res", combo_img)
cv2.waitKey(0)"""

cap = cv2.VideoCapture("test2.mp4")
while cap.isOpened():
    _, frame = cap.read()
    cany = canny(frame)
    crop_img = ROI(cany)
    lines = cv2.HoughLinesP(
        crop_img,
        2,
        np.pi / 180,
        100,
        np.array([]),
        minLineLength=40,
        maxLineGap=5,
    )
    average_lines = average_slope_intercept(frame, lines)
    line_img = display_lines(frame, average_lines)
    combo_img = cv2.addWeighted((frame), 0.8, line_img, 1, 1)
    cv2.imshow("res", combo_img)
    cv2.waitKey(1)
