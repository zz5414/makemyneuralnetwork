import os
import cv2
import numpy as np
import argparse
import pandas as pd
import random
from enum import IntEnum
from tqdm import tqdm
import math

class OBJECT(IntEnum):
    RECT = 0
    CIRCLE = 1
    TRIANLE = 2


_HEIGHT = 200
_WIDTH = 200
_TRAIN_SAMPLE = 60000
_VALID_SAMPLE = 12000

_MIN_WIDTH = 140
_MAX_WIDTH = 150
_PADDING = 10

# 도형이 묘사하는 파라미터는 총 5개이다.
# 도형의 타입, 도형의 중점 좌표(x, y), 길이, 선의 두께

# 도형을 그릴 떄 사용하는 parameter는 중점 좌표이다.
# 사각형과 원의 중점은 원래의 정의대로 정하였으며, 정사각형과 원이다.
# 직사각형과 타원은 나중에 할 예정

# 삼각형은 밑변이 이미지의 u축과 평행한 정삼각형이다.
# 삼각형을 묘사하는 중점은 밑변의 중점으로 정하였다.
# 직사각형이나 다른 형태는 나중에 할 예정
# 삼각형의 높이는 (cos30도)를 이용해 구함

# training Set, validation Set 중 하나라도 겹치는 일이 없도록
# is_unique함수를 구현하였다.

def degree_to_rad(degree):
    return degree * math.pi / 180


def is_unique(records, fig_type, cx, cy, w, tick, lt, rb):
    for idx, record in enumerate(records):
        if (record[0] == fig_type) and \
                (record[1] == cx) and \
                (record[2] == cy) and \
                (record[3] == w) and \
                (record[4] == tick):
            return False
    return True


def gen_parameter(fig_type):
    tick = random.randint(0, 3)
    if tick == 0: tick -= 1

    w = random.randint(_MIN_WIDTH, _MAX_WIDTH)

    if fig_type in (OBJECT.RECT, OBJECT.CIRCLE):
        cx = random.randint(int(_PADDING + (w / 2)), int(_WIDTH - _PADDING - (w / 2)))
        cy = random.randint(int(_PADDING + (w / 2)), int(_HEIGHT - _PADDING - (w / 2)))

        # left top, right bottom
        lt = (int(cx - w / 2), int(cy - w / 2))
        rb = (int(cx + w / 2), int(cy + w / 2))

        return tuple((fig_type, cx, cy, w, tick, lt, rb))
    else:
        cx = random.randint(int(_PADDING + (w / 2)), int(_WIDTH - _PADDING - (w / 2)))
        triangle_height = w * math.cos(degree_to_rad(30))
        cy = random.randint(int(_PADDING + triangle_height), int(_HEIGHT - _PADDING))

        # left top, right bottom
        lt = (int(cx - w / 2), int(cy - triangle_height))
        rb = (int(cx + w / 2), int(cy))

        return tuple((fig_type, cx, cy, w, tick, lt, rb))


def gen_rect(idx, records, dst_path):
    fig_type = OBJECT.RECT

    while True:
        parameter = gen_parameter(fig_type)
        ret = is_unique(records, *parameter)
        if ret: break

    _, cx, cy, w, tick, lt, rb = parameter


    img = np.zeros((_HEIGHT, _WIDTH), np.uint8)
    cv2.rectangle(img, lt, rb, (255, 255, 255), tick)

    # cv2.imshow('hello', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    cv2.imwrite(os.path.join(dst_path, f'{idx:05}.png'), img)

    return parameter


def gen_circle(idx, records, dst_path):
    fig_type = OBJECT.CIRCLE

    while True:
        parameter = gen_parameter(fig_type)
        ret = is_unique(records, *parameter)
        if ret: break

    _, cx, cy, w, tick, lt, rb = parameter

    img = np.zeros((_HEIGHT, _WIDTH), np.uint8)
    cv2.circle(img, (cx, cy), int(w/2), (255,255,255), tick)

    # cv2.rectangle(img, lt, rb, (255, 0, 0))
    # cv2.imshow('hello', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    cv2.imwrite(os.path.join(dst_path, f'{idx:05}.png'), img)

    return parameter


def gen_triangle(idx, records, dst_path):
    fig_type = OBJECT.TRIANLE

    while True:
        parameter = gen_parameter(fig_type)
        ret = is_unique(records, *parameter)
        if ret: break

    _, cx, cy, w, tick, lt, rb = parameter

    img = np.zeros((_HEIGHT, _WIDTH), np.uint8)
    p1 = [int(cx - (w / 2)), int(cy)]
    p2 = [int(cx + (w / 2)), int(cy)]
    p3 = [int(cx), int(cy - (w * math.cos(degree_to_rad(30))))]
    pts = np.array([p1, p3, p2])

    if tick == -1:
        cv2.fillPoly(img, [pts], (255,255,255))
    else:
        cv2.polylines(img, [pts], True, (255,255,255), tick)

    # cv2.rectangle(img, lt, rb, (255, 0, 0))
    # cv2.imshow('hello', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    cv2.imwrite(os.path.join(dst_path, f'{idx:05}.png'), img)

    return parameter


def generate_training_sample(dst_path):
    records = []
    idx = 0
    for _ in tqdm(range(int(_TRAIN_SAMPLE/3)), desc="RECT"):
        records.append(gen_rect(idx, records, dst_path))
        idx += 1

    for _ in tqdm(range(int(_TRAIN_SAMPLE/3), int(_TRAIN_SAMPLE*2/3)), desc="CIRCLE"):
        records.append(gen_circle(idx, records, dst_path))
        idx += 1

    for _ in tqdm(range(int(_TRAIN_SAMPLE*2/3), _TRAIN_SAMPLE), desc="TRIANGLE"):
        records.append(gen_triangle(idx, records, dst_path))
        idx += 1

    df = pd.DataFrame.from_records(records, columns=['fig_type', 'cx', 'cy', 'w', 'tick', 'lt', 'rb'])
    df.to_csv(os.path.join(dst_path, "training_sample.csv"), index=False)

    return records


def generate_validation_sample(dst_path, records):
    initial_len = len(records)
    idx = 0
    for _ in tqdm(range(int(_VALID_SAMPLE/3)), desc="RECT"):
        records.append(gen_rect(idx, records, dst_path))
        idx += 1

    for _ in tqdm(range(int(_VALID_SAMPLE/3), int(_VALID_SAMPLE*2/3)), desc="CIRCLE"):
        records.append(gen_circle(idx, records, dst_path))
        idx += 1

    for _ in tqdm(range(int(_VALID_SAMPLE*2/3), _VALID_SAMPLE), desc="TRIANGLE"):
        records.append(gen_triangle(idx, records, dst_path))
        idx += 1

    df = pd.DataFrame.from_records(records[initial_len:], columns=['fig_type', 'cx', 'cy', 'w', 'tick', 'lt', 'rb'])
    df.to_csv(os.path.join(dst_path, "validation_sample.csv"), index=False)

    return records


def main(args):
    if not os.path.exists(args.training_sample_path):
        os.makedirs(args.training_sample_path)
    if not os.path.exists(args.validation_sample_path):
        os.makedirs(args.validation_sample_path)

    records = generate_training_sample(args.training_sample_path)
    generate_validation_sample(args.validation_sample_path, records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('training_sample_path')
    parser.add_argument('validation_sample_path')
    args = parser.parse_args()
    main(args)
