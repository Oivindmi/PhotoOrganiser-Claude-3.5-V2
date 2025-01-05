import pkg_resources
import sys


def check_packages():
    print("Installed packages:")
    installed = {pkg.key: pkg.version for pkg in pkg_resources.working_set}

    opencv_packages = {
        name: version for name, version in installed.items()
        if "opencv" in name.lower()
    }

    if opencv_packages:
        print("\nOpenCV related packages:")
        for name, version in opencv_packages.items():
            print(f"  {name}: {version}")
    else:
        print("\nNo OpenCV packages found")

    try:
        import cv2
        print(f"\nOpenCV in sys.modules: {cv2.__file__}")
        print(f"OpenCV version: {cv2.__version__}")
    except ImportError:
        print("\nCannot import cv2")


if __name__ == "__main__":
    check_packages()