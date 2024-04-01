#!/usr/bin/python3

import cv2
import numpy as np
import subprocess
import time
from enum import Enum
import os
import sys
import glob
import socket

# The different LED regions in the mask, (Upper Left, Lower Right)
REG_0 = ((460, 677), (465, 687))
REG_1 = ((540, 701), (547, 714))
REG_2 = ((679, 732), (693, 747))
REG_3 = ((1116, 738), (1136, 757))

REGIONS = (REG_0, REG_1, REG_2, REG_3)


class LEDState(Enum):
    OFF = 1
    NORM = 2
    LOCATE = 3
    FAULT = 4
    LOCATE_FAULT = 5
    UNKNOWN = 6


MASK = "mask.png"

# Theory
# - Take N samples as fast as we can
# - Calculate the colors for each of the regions over the N samples
# - See if there is any significant change in each of the regions from sample to sample, where the
#   sample turns into similar value of RGB to indicate a gray, indicates the LED is off, thus
#   blinking.  If the LED is blinking == LOCATE enabled for that region
#
#   Steady Green == NORMAL
#   Steady Pinkish == FAILURE
#   Blink Green == LOCATE
#   Blink Pink == LOCATE & FAILURE


def find_non_black_average(img_data, r):
    """
	Returns (R, G, B) average for region
	:param img_data:
	:param r: Region of img_data
	:return: (R, G, B)
	"""
    r_sum = 0
    g_sum = 0
    b_sum = 0
    samples = 0

    (upper_left, lower_right) = r
    for x in range(upper_left[0], lower_right[0]):
        for y in range(upper_left[1], lower_right[1]):
            b, g, r = img_data[y, x]
            if b != 0 or g != 0 or r != 0:
                samples += 1
                r_sum += r
                g_sum += g
                b_sum += b

    return (r_sum // samples, g_sum // samples, b_sum // samples)


def region_colors(img_data):
    rc = []

    for r in REGIONS:
        rc.append(find_non_black_average(img_data, r))
    return rc


def acquire_image(device, mask_img):

    image_name = "/tmp/" + time.strftime("%Y%m%d-%H%M%S") + ".jpg"

    if "megadeth" in socket.gethostname():
        # local
        cmd = [
            "fswebcam", "-q", "-d", device, "-r", "1920x1080", "--jpeg", "100",
            "-S", "20", image_name
        ]
        subprocess.run(cmd, check=True, capture_output=False, shell=False)
    else:
        # remote
        cmd = f'ssh root@megadeth "fswebcam -q -d {device} -r 1920x1080 --jpeg 100  -S 20 -" > {image_name}'
        subprocess.run([cmd], check=True, capture_output=False, shell=True)
    return image_name


def quantization(img, n_colors=5):
    # https://docs.opencv.org/4.x/d1/d5c/tutorial_py_kmeans_opencv.html
    pixels = np.float32(img.reshape(-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    ret, label, center = cv2.kmeans(pixels, n_colors, None, criteria, 10,
                                    cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    f = label.flatten()
    res = center[f]
    return res.reshape((img.shape))


def process_image(image_file, mask_img=MASK, n_colors=5):
    org = cv2.imread(image_file)
    mask = cv2.imread(mask_img)
    img = cv2.bitwise_and(org, mask)
    #return quantization(img, n_colors)
    return img


def max_diff(sample):
    s = list(sample)
    s.sort()
    return abs(s[0] - s[2])


def led_state(sample):
    # This is very fragile and prone to error and depends on ambient light etc.
    # red, green, blue, +- range, name
    red = (242, 157, 190, 30, LEDState.FAULT)
    green = (104, 225, 151, 30, LEDState.NORM)

    states = [red, green]

    for s in states:
        state_chk = s[4]
        r = s[3]
        found = True

        for i in range(3):
            low = max(0, s[i] - r)
            hi = s[i] + r
            if sample[i] not in range(low, hi):
                found = False
                break

        if found:
            return state_chk

    if max_diff(sample) <= 25:
        return LEDState.OFF

    return LEDState.UNKNOWN


def get_region_numbers(files=None, count=10):

    samples = []

    if files is not None:
        for f in files:
            q_img = process_image(f)
            rc = region_colors(q_img)
            rc.append(f)
            samples.append(rc)
    else:
        for _ in range(0, count):
            fn = acquire_image("/dev/video0", "mask.png")
            q = process_image(fn)
            if INTERACTIVE:
                cv2.imshow('Quantization', q)

            colors = region_colors(q)
            colors.append(fn)
            samples.append(colors)

    return samples


def get_samples(n=10):
    global INTERACTIVE

    samples = []

    for _ in range(0, n):
        (aoi, fn) = acquire_image("/dev/video0", "mask.png")
        if INTERACTIVE:
            cv2.imshow("Image and Mask", aoi)
        q = quantization(aoi)
        if INTERACTIVE:
            cv2.imshow('Quantization', q)

        rc = region_colors(q)

        if INTERACTIVE:
            print(f"{rc}")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        samples.append((led_state(rc[0]), led_state(rc[1]), led_state(rc[2]),
                        led_state(rc[3]), fn))
    return samples


def min_max_calc(samples):
    """
	Returns a tuple of min/max values for r, g, b
	:param samples:
	:return:
	"""
    min_max = [[255, 0], [255, 0], [255, 0]]

    for s in samples:
        if s[0] < min_max[0][0]:
            min_max[0][0] = s[0]
        if s[0] > min_max[0][1]:
            min_max[0][1] = s[0]

        if s[1] < min_max[1][0]:
            min_max[1][0] = s[1]
        if s[1] > min_max[1][1]:
            min_max[1][1] = s[1]

        if s[2] < min_max[2][0]:
            min_max[2][0] = s[2]
        if s[2] > min_max[2][1]:
            min_max[2][1] = s[2]

    #print(f"min_max = {min_max}")

    return min_max


def led_state_mm(sample, r, g):

    # Some delta +- to account for some randomness in data
    # The problem is that the "R" and "G" ranges in some
    # cases overlap.  Thus when we fall into the bucket of
    # norm or fault, we do a simple check to see if R > G
    # and R < G and use that to determine the state.
    #
    # In testing this code seems to work, but could
    # it simplified?
    d = 10


    if sample[0] in range(g[0][0]-d, g[0][1]+d) and \
     sample[1] in range(g[1][0]-d, g[1][1]+d) and \
     sample[2] in range(g[2][0]-d, g[2][1]+d):

        if sample[0] > sample[1]:
            return LEDState.FAULT

        return LEDState.NORM
    elif sample[0] in range(r[0][0]-d, r[0][1]+d) and \
     sample[1] in range(r[1][0]-d, r[1][1]+d) and \
     sample[2] in range(r[2][0]-d, r[2][1]+d):

        if sample[0] < sample[1]:
            return LEDState.NORM

        return LEDState.FAULT
    else:
        if max_diff(sample) <= 25:
            return LEDState.OFF
        return None


def build_rgb_ranges():

    with open("data.learn", "r") as FH:
        data = FH.readlines()

    db = [
        dict(R=[], G=[]),
        dict(R=[], G=[]),
        dict(R=[], G=[]),
        dict(R=[], G=[])
    ]

    for line in data:
        ls = line.strip()
        (t, region, r, g, b) = ls.split(",")
        region = int(region)
        color = (int(r), int(g), int(b))
        db[region][t].append(color)

    rc = [
        dict(R=[], G=[]),
        dict(R=[], G=[]),
        dict(R=[], G=[]),
        dict(R=[], G=[])
    ]

    for r in range(len(REGIONS)):
        rc[r]["G"] = min_max_calc(db[r]["G"])
        rc[r]["R"] = min_max_calc(db[r]["R"])

    for i, v in enumerate(rc):
        print(f"Region[{i}]['R'] = {v['R']}")
        print(f"Region[{i}]['G'] = {v['G']}")
    return rc


def observed(led_states):
    norm = 0
    fault = 0
    unknown = 0
    for ls in led_states:
        if ls == LEDState.NORM:
            norm += 1
        elif ls == LEDState.FAULT:
            fault += 1
        else:
            unknown += 1

    if unknown > 0:
        # We likely have blinking
        if norm > fault:
            return LEDState.LOCATE
        if fault > norm:
            return LEDState.LOCATE_FAULT
    else:
        if norm > fault:
            return LEDState.NORM
        if fault > norm:
            return LEDState.FAULT

    return LEDState.UNKNOWN


def interpret(r):
    rc = []

    for i in range(len(REGIONS)):
        rt = [sublist[i] for sublist in r]
        rc.append(observed(rt))

    return rc


INTERACTIVE = False
USE_FILES = False

if __name__ == "__main__":

    if len(sys.argv) > 1 and len(sys.argv) != 6:
        print(
            f"syntax: {sys.argv[0]} collect [G|R][4], eg. led_determine.py collect G G G G"
        )
        sys.exit(1)

    if len(sys.argv) == 1:
        # Open the file with rgb values for each of the regions with known settings and build the needed
        # data to interpret the LED statuses.
        mmr = build_rgb_ranges()

        samples = get_region_numbers(count=10)

        results = []

        for s in samples:
            sr = []
            print(f"sample = {s}")
            for i, v in enumerate(s[0:4]):
                sr.append(led_state_mm(v, mmr[i]["R"], mmr[i]["G"]))
            print(f"mm result = {sr}")
            results.append(sr)

        # We have a list of states for each of the sample, we now need to determine what the LEDs are doing
        # steady color, or flashing.
        print(f"Best guess ... {interpret(results)}")

    elif sys.argv[1] == 'collect':
        # Loop collecting data and dump to stdout
        agv = sys.argv[2:6]
        for v in agv:
            if v not in ["G", "R"]:
                print(f"Invalid expected state for {v}")

        if USE_FILES:
            files = glob.glob("*.jpg")
            color_numbers = get_region_numbers(files)
        else:
            color_numbers = get_region_numbers(count=30)

        for c in color_numbers:
            for i, v in enumerate(c[0:4]):
                print(f"{agv[i]}, {i}, {v[0]}, {v[1]}, {v[2]}")
    else:
        print(f"Invalid option {sys.argv[1]}")
        sys.exit(2)
