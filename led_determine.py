#!/usr/bin/python3

# SPDX-License-Identifier: GPL-2.0-or-later

import cv2
import subprocess
import time
from enum import Enum
import sys
import glob
import socket
import os
import yaml
import pathlib
import argparse
import csv
import statistics
from collections import defaultdict
import numpy as np
from sklearn.mixture import GaussianMixture

# Theory
# - Take N samples as fast as we can
# - Calculate the intensity for each of the regions over the N samples.
# - Using the intensity determine if the LED is on or off.
# - If the LED is on, we determine the color of the LED.
# - Walk through all samples to see if the LED state is consistent or toggling on/off which indicates
#   blinking.  If the LED is blinking == LOCATE enabled for that region

#   NOTE: We are using the hard coded value of 80 for on threshold for on.  This is a magic number that
#   was determined by studying the data and seeing what value was most likely to be on.
#   We're not using the existing code of calculating this with the 'collect' command.
#
#   Steady Green == NORMAL
#   Steady Pinkish/Red == FAILURE
#   Blink Green == LOCATE
#   Blink Pink == LOCATE & FAILURE

# The different LED regions in the mask, (Upper Left, Lower Right)
# Note: These are a small region of the LED locations
# TODO: move this to the config.yaml
REG_0 = ((601, 211), (604, 215))
REG_1 = ((673, 322), (677, 326))
REG_2 = ((809, 515), (814, 520))
REG_3 = ((1124, 826), (1129, 827))

REGIONS = (REG_0, REG_1, REG_2, REG_3)

# Define to dump out debug
DEBUG = bool(os.getenv("LED_DETERMINE_CV_DEBUG", ""))

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

FILE_PREFIX = "/tmp/led_determine_cv_"

ON_THRESHOLD = 80


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


def validate_rg_args(values):
    if len(values) != 4:
        raise argparse.ArgumentTypeError("Exactly 4 arguments are required.")
    for val in values:
        if val not in ("R", "G"):
            raise argparse.ArgumentTypeError(
                f"Invalid character '{val}': Only 'R' or 'G' allowed."
            )
    return values


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
                r_sum += int(r)
                g_sum += int(g)
                b_sum += int(b)

    return (r_sum // pixel_samples, g_sum // pixel_samples, b_sum // pixel_samples)


def find_intensity(img_data, r):
    """
    Returns 0-255 representing intensity
    :param img_data:
    :param r: Region of img_data
    :return: 0-255 intensity
    """
    intensity_sum = int(0)
    pixed_samples = int(0)

    (upper_left, lower_right) = r

    for x in range(upper_left[0], lower_right[0]):
        for y in range(upper_left[1], lower_right[1]):
            intensity_sum += int(img_data[y, x])
            pixed_samples += 1

    v = intensity_sum // pixed_samples
    debug(
        f"intensity:upper_left:{upper_left} lower_right:{lower_right} intensity_sum:{intensity_sum} pixed_samples:{pixed_samples} v:{v}"
    )
    return v


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
    image_name = FILE_PREFIX + time.strftime("%Y%m%d-%H%M%S") + ".jpg"

    if "megadeth" in socket.gethostname():
        # local
        cmd = [
            "fswebcam",
            "-q",
            "-d",
            device,
            "-r",
            "1920x1080",
            "--jpeg",
            "100",
            "-S",
            "20",
            image_name,
        ]
        subprocess.run(cmd, check=True, capture_output=False, shell=False)
    else:
        # remote
        cmd = f'ssh root@megadeth "fswebcam -q -d {device} -r 1920x1080 --jpeg 100  -S 20 -" > {image_name}'
        subprocess.run([cmd], check=True, capture_output=False, shell=True)
    return image_name


def process_image(image_file):
    return cv2.imread(image_file)


def get_region_numbers(count=10, replay=False):
    region_numbers = []

    if not replay:
        for _ in range(0, count):
            fn = acquire_image("/dev/video0")
            q = process_image(fn)

            colors = region_colors(q)
            intensity = region_intensity(q)

            leds = []
            for idx in range(len(colors)):
                leds.append(LED(colors[idx], intensity[idx], fn))

            region_numbers.append(leds)
    else:
        for f in get_capture_files():
            q = process_image(f)
            colors = region_colors(q)
            intensity = region_intensity(q)

            debug(f"colors:{colors}")
            debug(f"intensity values:{intensity}")

            leds = []
            for idx in range(len(colors)):
                leds.append(LED(colors[idx], intensity[idx], f))
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

    debug(f"Sample:{sample} - On Threshold:{on_threshold}")

    # if sample.intensity() >= (on_threshold - d):
    if sample.intensity() >= ON_THRESHOLD:
        if sample.is_red():
            return LEDState.FAULT
        else:
            return LEDState.NORM
    else:
        return LEDState.OFF


def build_numbers():
    with open(os.path.join(__location__, "data.learn"), "r") as FH:
        data = FH.readlines()

    db = [dict(R=[], G=[]), dict(R=[], G=[]), dict(R=[], G=[]), dict(R=[], G=[])]

    for line in data:
        ls = line.strip()
        (t, region, it) = ls.split(",")
        region = int(region)
        intensity = int(it)
        db[region][t].append(intensity)

    rc = [dict(R=[], G=[]), dict(R=[], G=[]), dict(R=[], G=[]), dict(R=[], G=[])]

    for r in range(len(REGIONS)):
        rc[r]["G"] = min_max_calc(db[r]["G"])
        rc[r]["R"] = min_max_calc(db[r]["R"])

    for i, v in enumerate(rc):
        debug(f"Region[{i}]['R'] = {v['R']}")
        debug(f"Region[{i}]['G'] = {v['G']}")

    region_mins = []

    for i, v in enumerate(rc):
        r_min = v["R"][0]
        g_min = v["G"][0]
        debug(f"R min:{r_min} G min:{g_min}")
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


def get_capture_files():
    """
    Returns a list of all files that match the FILE_PREFIX pattern
    :return: List of file paths
    """
    return glob.glob(FILE_PREFIX + "*")


def delete_captures():
    for f in get_capture_files():
        pathlib.Path(f).unlink()


def process_data_file(filename):
    grouped_data = defaultdict(list)

    # Read and group the data
    with open(filename, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            group_key = (row[0].strip(), int(row[1].strip()))
            value = int(row[2].strip())
            grouped_data[group_key].append(value)

    # Compute stats and print header
    print("group_0,group_1,average,median,stddev,lower_3stddev,min")

    for (label, index), values in sorted(grouped_data.items()):
        mean_val = statistics.mean(values)
        median_val = statistics.median(values)
        stddev_val = statistics.stdev(values) if len(values) > 1 else 0.0
        lower_3stddev = mean_val - 3 * stddev_val
        min_val = min(values)

        print(
            f"{label},{index},{mean_val:.2f},{median_val:.2f},{stddev_val:.2f},{lower_3stddev:.2f},{min_val}"
        )


def process_data_file_with_bimodal_check(filename):
    grouped_data = defaultdict(list)

    # Read and group the data
    with open(filename, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            group_key = (row[0].strip(), int(row[1].strip()))
            value = int(row[2].strip())
            grouped_data[group_key].append(value)

    # Header
    print(
        "group_0,group_1,average,median,stddev,lower_3stddev,min,bimodal,subgroup1_min,subgroup1_max,subgroup2_min,subgroup2_max"
    )

    for (label, index), values in sorted(grouped_data.items()):
        mean_val = statistics.mean(values)
        median_val = statistics.median(values)
        stddev_val = statistics.stdev(values) if len(values) > 1 else 0.0
        lower_3stddev = mean_val - 3 * stddev_val
        min_val = min(values)

        # Bimodal check using Gaussian Mixture Model
        data = np.array(values).reshape(-1, 1)
        gmm1 = GaussianMixture(n_components=1, random_state=0).fit(data)
        gmm2 = GaussianMixture(n_components=2, random_state=0).fit(data)
        is_bimodal = gmm2.bic(data) < gmm1.bic(data)

        subgroup1_min = subgroup1_max = subgroup2_min = subgroup2_max = ""

        if is_bimodal:
            labels = gmm2.predict(data)
            cluster1 = data[labels == 0].flatten()
            cluster2 = data[labels == 1].flatten()
            subgroup1_min = int(cluster1.min())
            subgroup1_max = int(cluster1.max())
            subgroup2_min = int(cluster2.min())
            subgroup2_max = int(cluster2.max())

        print(
            f"{label},{index},{mean_val:.2f},{median_val:.2f},{stddev_val:.2f},{lower_3stddev:.2f},{min_val},{is_bimodal},{subgroup1_min},{subgroup1_max},{subgroup2_min},{subgroup2_max}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process one of: collect, collect-replay, replay, evaluate, process-data"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # collect
    collect_parser = subparsers.add_parser(
        "collect", help="Run collect with 4 R/G args"
    )
    collect_parser.add_argument(
        "colors", nargs=4, type=str, metavar="C", help="4 color chars (R or G)"
    )
    collect_parser.add_argument(
        "--keep-images", action="store_true", help="Retain intermediate image files, to allow for analysis"
    )
    collect_parser.set_defaults(func=validate_rg_args)

    # collect-replay
    collect_replay_parser = subparsers.add_parser(
        "collect-replay", help="Run collect-replay with 4 R/G args"
    )
    collect_replay_parser.add_argument(
        "colors", nargs=4, type=str, metavar="C", help="4 color chars (R or G)"
    )
    collect_replay_parser.set_defaults(func=validate_rg_args)

    # replay
    replay_parser = subparsers.add_parser("replay", help="Run replay (no args)")

    # evaluate
    evaluate_parser = subparsers.add_parser("evaluate", help="Run evaluate (no args)")
    evaluate_parser.add_argument(
        "--keep-images", action="store_true", help="Retain intermediate image files, to allow for analysis"
    )

    # process-data
    process_data_parser = subparsers.add_parser(
        "process-data", help="Process data file"
    )
    process_data_parser.add_argument("filename", type=str, help="Data file to process")

    args = parser.parse_args()

    # Validate RG arguments for collect and collect-replay
    if args.command in ("collect", "collect-replay"):
        try:
            args.colors = validate_rg_args(args.colors)
        except argparse.ArgumentTypeError as e:
            parser.error(str(e))

    # File with wwn for each 'slot' we are monitoring via USB camera
    with open(os.path.join(__location__, "config.yaml"), "r") as FH:
        slot_ids = yaml.load(FH, Loader=yaml.Loader)

    if args.command == "evaluate":
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

        if LEDState.UNKNOWN not in results:
            if not args.keep_images:
                delete_captures()

        output = {"statekey": LEDState.y(), "results": []}
        for i, v in enumerate(results):
            output["results"].append(dict(wwn=slot_ids["slots"][i], state=v.value))

        print(yaml.dump(output, Dumper=yaml.Dumper))
        sys.exit(0)

    elif args.command == "collect":
        # Loop collecting data and dump to stdout
        agv = sys.argv[2:6]
        for v in agv:
            if v not in ["G", "R"]:
                print(f"Invalid expected state for {v}")

        numbers = get_region_numbers(count=30)

        if not args.keep_images:
            delete_captures()

        for c in numbers:
            for i, v in enumerate(c):
                print(f"{agv[i]}, {i}, {v.intensity()}", file=sys.stderr)
        sys.exit(0)
    elif args.command == "collect-replay":
        # Loop collecting data and dump to stdout
        agv = sys.argv[2:6]
        for v in agv:
            if v not in ["G", "R"]:
                print(f"Invalid expected state for {v}")

        numbers = get_region_numbers(count=30, replay=True)

        for c in numbers:
            for i, v in enumerate(c):
                print(f"{agv[i]}, {i}, {v.intensity()}", file=sys.stderr)
        sys.exit(0)
    elif args.command == "replay":
        mmr = build_numbers()
        # Replay the last 10 captures
        samples = get_region_numbers(count=10, replay=True)
        results = []

        for s in samples:
            sr = []
            for i, v in enumerate(s):
                debug(f"enumerating: {i} , {v}")
                sr.append(led_state_mm(v, mmr[i]))
            debug(f"mm result = {sr}")
            results.append(sr)

        # We have a list of states for each of the sample, we now need to determine what
        # the LEDs are doing steady color, or flashing.
        results = interpret(results)

        if LEDState.UNKNOWN not in results:
            if not args.keep_images:
                delete_captures()

        output = {"statekey": LEDState.y(), "results": []}
        for i, v in enumerate(results):
            output["results"].append(dict(wwn=slot_ids["slots"][i], state=v.value))

        print(yaml.dump(output, Dumper=yaml.Dumper))
        sys.exit(0)
    elif args.command == "process-data":
        process_data_file_with_bimodal_check(args.filename)
        sys.exit(0)
