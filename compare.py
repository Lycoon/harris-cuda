import os
import subprocess
import sys

import cv2


def parse_points(file):
    points = []
    with open(file, "r") as f:
        line = f.readline()
        while line:
            values = line.split("|")
            x = values[0].strip().removeprefix("x: ").strip()
            y = values[1].strip().removeprefix("y: ").strip()
            points.append((int(x), int(y)))
            line = f.readline()
    return points

def main():
    if len(sys.argv) != 2:
        print("Usage: ./compare <image>")
        exit()

    image_path = sys.argv[1]

    python = subprocess.Popen(("python", "solution/harris.py", image_path), stdout=subprocess.DEVNULL)
    python.wait()
    os.rename("output.txt", "python.txt")
    print("Python version done.")

    ref = subprocess.Popen(("ref/build/harris", image_path), stdout=subprocess.DEVNULL)
    ref.wait()
    os.rename("output.txt", "cpu.txt")
    print("CPU version done.")

    cuda = subprocess.Popen(("cuda/build/harris", image_path), stdout=subprocess.DEVNULL)
    cuda.wait()
    os.rename("output.txt", "gpu.txt")
    print("GPU version done.")

    python_points = parse_points("python.txt")
    cpu_points = parse_points("cpu.txt")
    gpu_points = parse_points("gpu.txt")

    image = cv2.imread(image_path)

    for p in python_points:
        cv2.circle(image, (p[0], p[1]), 3, (0, 255, 0), -1)
    for p in cpu_points:
        cv2.circle(image, (p[0], p[1]), 3, (255, 255, 0), -1)
    for p in gpu_points:
        cv2.circle(image, (p[0], p[1]), 3, (255, 0, 255), -1)

    cv2.imwrite("comparison.png", image)


if __name__ == "__main__":
    main()
