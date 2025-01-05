import winreg
import os
import sys
import ctypes
from typing import List, Optional


def is_admin() -> bool:
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False


def get_cuda_path() -> Optional[str]:
    return os.environ.get('CUDA_PATH')


def add_to_system_path(paths: List[str]) -> bool:
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            'SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Environment',
                            0, winreg.KEY_ALL_ACCESS) as key:
            current_path = winreg.QueryValueEx(key, 'Path')[0]

            # Add new paths if they don't exist
            new_paths = [p for p in paths if p.lower() not in current_path.lower()]
            if new_paths:
                new_path_str = current_path + ';' + ';'.join(new_paths)
                winreg.SetValueEx(key, 'Path', 0, winreg.REG_EXPAND_SZ, new_path_str)
                return True
            return False
    except Exception as e:
        print(f"Error modifying PATH: {e}")
        return False


def main():
    if not is_admin():
        print("This script needs administrator privileges.")
        print("Please run it as administrator.")
        return

    cuda_path = get_cuda_path()
    if not cuda_path:
        print("CUDA_PATH environment variable not found.")
        print("Please ensure CUDA is installed correctly.")
        return

    paths_to_add = [
        os.path.join(cuda_path, 'bin'),
        os.path.join(cuda_path, 'lib', 'x64'),
        os.path.join(cuda_path, 'libnvvp')
    ]

    print("\nChecking paths to add:")
    for path in paths_to_add:
        exists = os.path.exists(path)
        print(f"{path}: {'Exists' if exists else 'Not Found'}")

    proceed = input("\nDo you want to proceed with PATH modification? (y/n): ")
    if proceed.lower() != 'y':
        print("Operation cancelled.")
        return

    if add_to_system_path(paths_to_add):
        print("\nPATH successfully updated!")
        print("Please restart your computer for changes to take effect.")
        print("\nAfter restart, run these commands:")
        print("pip uninstall opencv-python opencv-python-headless opencv-contrib-python")
        print("pip install opencv-python-cuda==4.8.0.76")
    else:
        print("\nNo changes were necessary - paths already exist in PATH")


if __name__ == "__main__":
    main()