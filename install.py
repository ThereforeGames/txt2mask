from launch import is_installed, run_pip

if not is_installed("clipseg"):
    run_pip(f"install git+https://github.com/timojl/clipseg#656e0c662bd1c9a5ae511011642da5b7d8503312", "clipseg")
