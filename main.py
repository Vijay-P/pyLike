#!/usr/bin/env python3

import numpy as np
import cv2
import math
import queue as pyq

eye_percent_top = 25
eye_percent_side = 13
eye_percent_height = 30
eye_percent_width = 35
smooth_face = True
sigma_factor = 0.005
gradient_threshold = 50
enable_weight = True
weight_divisor = 1
enable_post_process = False
post_process_threshold = 0.97
plot_vector_field = False
fast_eye_width = 50


def set_cap_props(cap):
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FPS, 30)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    return


def get_cap_props(cap):
    # print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # print(cap.get(cv2.CAP_PROP_FOURCC))
    # print(cap.get(cv2.CAP_PROP_FPS))
    return


def blur(frame, fw, fh):
    global sigma_factor
    sigma = sigma_factor * fw
    blur_frame = cv2.GaussianBlur(frame, (5, 5), sigma, sigma)
    return blur_frame


def matlab_gradient(frame, axis):
    # axis = x or y

    # MatLab gradient algorithm: [x(2)-x(1) (x(3:end)-x(1:end-2))/2 x(end)-x(end-1)],
    # where x is input matrix.

    # frame is a 3D numpy array (or matrix), where each row is a row in the
    # image and each column is a column in the image, and the contents of the
    # column is the pixel's RGB value

    if axis == "y":
        frame = cv2.transpose(frame)

    grad = np.ndarray(shape=(frame.shape), dtype=(frame.dtype))
    for row in range(np.size(frame, 0)):
        # x(2)-x(1)
        grad[0] = np.subtract(frame[1], frame[0])
        # (x(3:end)-x(1:end-2))/2
        for col in range(1, np.size(frame, 1) - 2):
            grad[row, col] = np.true_divide(np.subtract(
                frame[row, col + 1], frame[row, col - 1]), 2)
        # x(end)-x(end-1)
        grad[np.size(frame, 0) - 1] = np.subtract(frame[np.size(frame, 0) - 1],
                                                  frame[np.size(frame, 0) - 2])

    if axis == "y":
        return cv2.transpose(grad)
    else:
        return grad


def matrix_vector_magnitudes(x_gradient, y_gradient):
    mags = np.ndarray(shape=(x_gradient.shape), dtype=(x_gradient.dtype))
    for row in range(np.size(x_gradient, 0)):
        for col in range(np.size(y_gradient, 1)):
            gx = x_gradient[row, col]
            gy = y_gradient[row, col]
            magnitude = math.sqrt(math.pow(gx, 2) + math.pow(gy, 2))
            mags[row, col] = magnitude
    return mags


def compute_dynamic_threshold(magnitudes, std_dev_factor):
    mean_magn_grad, std_magn_grad = cv2.meanStdDev(magnitudes)
    std_dev = std_magn_grad[0][0] / math.sqrt(np.size(magnitudes, 0) * np.size(magnitudes, 1))
    return std_dev_factor * std_dev + mean_magn_grad[0][0]


def test_possible_centers_formula(x, y, weight, x_grad_val, y_grad_val, out_sum):
    global enable_weight
    global weight_divisor
    for row in range(np.size(out_sum, 0)):
        for col in range(np.size(out_sum, 1)):
            if((x == col) and (y == row)):
                continue
            displacement_x = x - col
            displacement_y = y - row
            magnitude = math.sqrt(math.pow(displacement_x, 2) + math.pow(displacement_y, 2))
            displacement_x = displacement_x / magnitude
            displacement_y = displacement_y / magnitude
            dot_product = displacement_x * x_grad_val + displacement_y * y_grad_val
            if (dot_product < 0):
                dot_product = 0.0
            if(enable_weight):
                out_sum[row, col] = dot_product * dot_product * \
                    (np.true_divide(weight[row, col], weight_divisor))
            else:
                out_sum[row, col] = dot_product * dot_product
    return out_sum


def scale_to_fast_size(roi):
    global fast_eye_width
    return cv2.resize(roi, (fast_eye_width, int((fast_eye_width / np.size(roi, 1)) * np.size(roi, 0))))


def unscale_point(point, rect):
    global fast_eye_width

    x = rect[0]
    y = rect[1]
    w = rect[2]
    h = rect[3]

    ratio = fast_eye_width / w
    unscale_x = point[0] / ratio
    unscale_y = point[1] / ratio

    return (unscale_x, unscale_y)


def flood_should_push_point(point, mask):
    if((point[1] < np.size(mask, 0)) and (point[0] < np.size(mask, 1))):
        return True
    else:
        return False


def flood_kill_edges(flood):
    mask = np.ndarray(shape=(flood.shape), dtype=(np.uint8))
    mask.fill(255)
    Q = pyq.Queue()
    Q.put((0.0, 0.0))
    while not Q.empty():
        point = Q.get()
        if(point == 0.0):
            continue
        point_np = (point[0] + 1, point[1])
        if(flood_should_push_point(point_np, mask)):
            Q.put(point_np)
        point_np = (point[0] - 1, point[1])
        if(flood_should_push_point(point_np, mask)):
            Q.put(point_np)
        point_np = (point[0], point[1] + 1)
        if(flood_should_push_point(point_np, mask)):
            Q.put(point_np)
        point_np = (point[0], point[1] - 1)
        if(flood_should_push_point(point_np, mask)):
            Q.put(point_np)
        mask[point[0], point[1]] = 0
    return mask


# do not have MathGL, non-functional
def plot_vec_field(x_gradient, y_gradient, eye_roi):
    return eye_roi


def find_eye_center(face_roi, eye_roi, eye_rect, name_string):
    print("starting find " + name_string)
    global gradient_threshold
    global enable_post_process
    global post_process_threshold
    global plot_vector_field

    eye_copy = np.copy(eye_roi)

    eye_roi = scale_to_fast_size(eye_roi)

    x_gradient = matlab_gradient(eye_roi, "x")
    y_gradient = matlab_gradient(eye_roi, "y")
    # cv2.imwrite("x_gradient.jpg", x_gradient)
    # cv2.imwrite("y_gradient.jpg", y_gradient)

    print("got gradients")

    mags = matrix_vector_magnitudes(x_gradient, y_gradient)

    grad_thresh = compute_dynamic_threshold(mags, gradient_threshold)

    # normalize
    for row in range(np.size(eye_roi, 0)):
        for col in range(np.size(eye_roi, 1)):
            magnitude = mags[row, col]
            if(magnitude > grad_thresh):
                binarized_x = np.true_divide(x_gradient[row, col], magnitude)
                binarized_y = np.true_divide(y_gradient[row, col], magnitude)
                x_gradient[row, col] = binarized_x * 255
                y_gradient[row, col] = binarized_y * 255
            else:
                x_gradient[row, col] = 0
                y_gradient[row, col] = 0

    # cv2.imwrite("normalized_x.jpg", x_gradient)
    # cv2.imwrite("normalized_y.jpg", y_gradient)

    print("normalized")

    #blur and invert
    weight = blur(eye_roi, 0, 0)
    # cv2.imwrite("blur.jpg", weight)
    for row in range(np.size(weight, 0)):
        for col in range(np.size(weight, 1)):
            weight[row, col] = 255 - weight[row, col]

    # cv2.imwrite("invert.jpg", weight)

    print("blurred and inverted")

    out_sum = np.zeros(shape=(eye_roi.shape), dtype=(eye_roi.dtype))

    # test each point as a possible center
    for row in range(np.size(weight, 0)):
        for col in range(np.size(weight, 1)):
            if((x_gradient[row, col] == 0) and (y_gradient[row, col] == 0)):
                continue
            out_sum = test_possible_centers_formula(row, col, weight, x_gradient[
                row, col], y_gradient[row, col], out_sum)

    print("points tested")

    num_gradients = (np.size(weight, 0) * np.size(weight, 1))
    out = np.multiply(out_sum, (1.0 / num_gradients))
    out = out.astype(np.float32)
    min_val, max_val, min_pt, max_pt = cv2.minMaxLoc(out)

    print("minimax done")

    if(enable_post_process):
        flood_threshold = max_val * post_process_threshold
        retval, flood_clone = cv2.threshold(out, flood_threshold, 0.0, cv2.THRESH_TOZERO)
        if(plot_vector_field):
            vector_field = plot_vec_field(x_gradient, y_gradient, flood_clone)
            # imwrite("vector_field.png", vector_field)
        mask = flood_kill_edges(flood_clone)
        min_val, max_val, min_pt, max_pt = cv2.minMaxLoc(mask)

    print("postprocessing done")

    eye_point = unscale_point(max_pt, eye_rect)

    print("unscaled")

    # cv2.circle(eye_copy, (int(eye_point[0]), int(eye_point[1])), 3, (0, 255, 0))
    # cv2.imwrite(name_string + ".jpg", eye_copy)

    return eye_point


def find_eyes(frame_gray, rect_face):
    global smooth_face
    global eye_percent_width
    global eye_percent_height
    global eye_percent_top
    global eye_percent_side

    x, y, w, h = rect_face
    roi_gray = frame_gray[y:y + h, x:x + w]

    if(smooth_face):
        roi_gray = blur(roi_gray, w, h)

    # roi_gray2 = roi_gray

    eye_roi_width = w * (eye_percent_width / 100)
    eye_roi_height = w * (eye_percent_height / 100)
    eye_roi_top = h * (eye_percent_top / 100)

    left_x = w * (eye_percent_side / 100)
    left_y = eye_roi_top

    right_x = w - eye_roi_width - (w * (eye_percent_side / 100))
    right_y = eye_roi_top

    roi_left_eye = roi_gray[left_y:left_y + eye_roi_height, left_x:left_x + eye_roi_width]
    roi_right_eye = roi_gray[right_y:right_y + eye_roi_height, right_x:right_x + eye_roi_width]

    rect_left_eye = (left_x, left_y, eye_roi_width, eye_roi_height)
    rect_right_eye = (right_x, right_y, eye_roi_width, eye_roi_height)

    # cv2.imwrite("left.jpg", roi_left_eye)
    # cv2.imwrite("right.jpg", roi_right_eye)
    left_point = find_eye_center(roi_gray, roi_left_eye, rect_left_eye, "left eye")
    right_point = find_eye_center(roi_gray, roi_right_eye, rect_right_eye, "right eye")
    left_point = ((left_point[0] + left_x), (left_point[1] + left_y))
    right_point = ((right_point[0] + right_x), (right_point[1] + right_y))
    cv2.circle(roi_gray, (int(left_point[0]), int(left_point[1])), 3, (255, 0, 0))
    cv2.circle(roi_gray, (int(right_point[0]), int(right_point[1])), 3, (255, 0, 0))
    # cv2.imshow("eye_face", roi_gray)
    return roi_gray


def face_detect(frame):
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    img = frame
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 2)
    if(np.size(faces, 0) > 0):
        return find_eyes(gray, faces[0])


if __name__ == '__main__':
    cap = cv2.VideoCapture(1)

    set_cap_props(cap)
    get_cap_props(cap)

    while True:
        ret, frame = cap.read()
        # frame = cv2.imread('selfie.jpg')
        frame = cv2.flip(frame, 1)  # horizontal flip
        frame = face_detect(frame)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
