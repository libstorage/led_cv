#!/usr/bin/python3

import cv2
import subprocess
import time
from enum import Enum
import sys
import glob
import socket
import os
import yaml

# The different LED regions in the mask, (Upper Left, Lower Right)
# Note: These are a small region of the LED locations
REG_0 = ((463, 680), (464, 685))
REG_1 = ((542, 706), (546, 710))
REG_2 = ((682, 737), (688, 742))
REG_3 = ((1123, 748), (1129, 754))

REGIONS = (REG_0, REG_1, REG_2, REG_3)

DEBUG = bool(os.getenv("LED_DETERMINE_CV_DEBUG", ""))

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

class LEDState(Enum):
    OFF = 1
    NORM = 2
    LOCATE = 3
    FAULT = 4
    LOCATE_FAULT = 5
    UNKNOWN = 6

    @staticmethod
    def y():
        return "{OFF: 1, NORM: 2, LOCATE: 3, FAULT: 4, LOCATE_FAULT: 5, UNKNOWN: 6}"


class LED:

    def __init__(self, rgb, intensity, fn):
        self._rgb = rgb
        self._intensity = intensity
        self._fn = fn

    def is_red(self):
        return self._rgb[0] > self._rgb[1]

    def is_green(self):
        return self._rgb[1] > self._rgb[0]

    def intensity(self):
        return self._intensity

    def __str__(self):
        return f"({self._rgb[0]}, {self._rgb[1]}, {self._rgb[2]}), {self._intensity}, {self._fn}"

    def fn(self):
        return self._fn


def debug(msg):
    if DEBUG:
        print(msg)

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
    pixel_samples = 0

    (upper_left, lower_right) = r
    for x in range(upper_left[0], lower_right[0]):
        for y in range(upper_left[1], lower_right[1]):
            b, g, r = img_data[y, x]
            if b != 0 or g != 0 or r != 0:
                pixel_samples += 1
                r_sum += r
                g_sum += g
                b_sum += b

    return (r_sum // pixel_samples, g_sum // pixel_samples, b_sum // pixel_samples)


def find_intensity(img_data, r):
    """
    Returns 0-255 representing intensity
    :param img_data:
    :param r: Region of img_data
    :return: 0-255 intensity
    """
    intensity_sum = 0
    pixed_samples = 0

    (upper_left, lower_right) = r
    for x in range(upper_left[0], lower_right[0]):
        for y in range(upper_left[1], lower_right[1]):
            intensity_sum += img_data[y, x]
            pixed_samples += 1

    return (intensity_sum // pixed_samples)


def region_colors(img_data):
    rc = []

    for r in REGIONS:
        rc.append(find_non_black_average(img_data, r))
    return rc


def region_intensity(img_data):
    rc = []

    gray_scale = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)

    for r in REGIONS:
        rc.append(find_intensity(gray_scale, r))
    return rc


def acquire_image(device):
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


def process_image(image_file):
    return cv2.imread(image_file)


def get_region_numbers(count=10):
    region_numbers = []

    for _ in range(0, count):
        fn = acquire_image("/dev/video0")
        q = process_image(fn)

        colors = region_colors(q)
        intensity = region_intensity(q)

        leds = []
        for idx in range(len(colors)):
            leds.append(LED(colors[idx], intensity[idx], fn))

        region_numbers.append(leds)

    return region_numbers


def min_max_calc(led_samples):
    """
    Returns a tuple of intensity (min, max)
    :param led_samples:
    :return:
    """

    debug(f"samples = {led_samples}")
    min_max = [255, 0]

    for ls in led_samples:
        if ls < min_max[0]:
            min_max[0] = ls
        if ls > min_max[1]:
            min_max[1] = ls

    debug(f"min_max = {min_max}")

    return min_max


def led_state_mm(sample, on_threshold):
    d = 10

    debug(f"{sample} - {on_threshold}")

    if sample.intensity() >= (on_threshold - d):
        if sample.is_red():
            return LEDState.FAULT
        else:
            return LEDState.NORM
    else:
        return LEDState.OFF


def build_numbers():
    with open(os.path.join(__location__, "data.learn"), "r") as FH:
        data = FH.readlines()

    db = [
        dict(R=[], G=[]),
        dict(R=[], G=[]),
        dict(R=[], G=[]),
        dict(R=[], G=[])
    ]

    for line in data:
        ls = line.strip()
        (t, region, it) = ls.split(",")
        region = int(region)
        intensity = int(it)
        db[region][t].append(intensity)

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
        debug(f"Region[{i}]['R'] = {v['R']}")
        debug(f"Region[{i}]['G'] = {v['G']}")

    region_mins = []

    for i, v in enumerate(rc):
        r_min = v['R'][0]
        g_min = v['G'][0]
        region_mins.append(min(r_min, g_min))

    debug(f"Region mins = {region_mins}")
    return region_mins


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



if __name__ == "__main__":

    # File with wwn for each 'slot' we are monitoring via USB camera
    with open(os.path.join(__location__, "config.yaml"), "r") as FH:
        slot_ids = yaml.load(FH, Loader=yaml.Loader)

    if len(sys.argv) > 1 and len(sys.argv) != 6:
        print(
            f"syntax: {sys.argv[0]} collect [G|R][4], eg. led_determine.py collect G G G G"
        )
        sys.exit(1)

    if len(sys.argv) == 1:
        # Open the file with the intensity values for each of the regions with known settings
        # and build the needed data to interpret the LED statuses.
        mmr = build_numbers()
        samples = get_region_numbers(count=10)

        results = []

        for s in samples:
            sr = []
            for i, v in enumerate(s):
                sr.append(led_state_mm(v, mmr[i]))
            debug(f"mm result = {sr}")
            results.append(sr)

        # We have a list of states for each of the sample, we now need to determine what
        # the LEDs are doing steady color, or flashing.
        results = interpret(results)

        output = dict(statekey=LEDState.y(), results=[])
        for i, v in enumerate(results):
            output['results'].append(dict(wwn=slot_ids['slots'][i], state=v.value))

        print(yaml.dump(output, Dumper=yaml.Dumper))
        sys.exit(0)
    elif sys.argv[1] == 'collect':
        # Loop collecting data and dump to stdout
        agv = sys.argv[2:6]
        for v in agv:
            if v not in ["G", "R"]:
                print(f"Invalid expected state for {v}")

        numbers = get_region_numbers(count=30)

        for c in numbers:
            for i, v in enumerate(c):
                print(f"{agv[i]}, {i}, {v.intensity()}")
    else:
        print(f"Invalid option {sys.argv[1]}")
        sys.exit(2)
