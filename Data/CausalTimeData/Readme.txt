CausalTime dataset setup
========================

This dataset is not bundled with the repository and is not auto-downloaded.

1. Download the generated data from: https://www.causaltime.cc/dataset/
   (the "Dataset Download" link of https://github.com/jarrycyx/UNN; the site
   is sometimes unreachable -- try again later or from another network, and
   keep a backup of the six .npy files, ~12 MB in total).
2. Place each domain's files in its own subfolder right here, under
   Dataset/<domain>/, so the final layout looks like:

       Data/CausalTimeData/Dataset/medical/gen_data.npy
       Data/CausalTimeData/Dataset/medical/graph.npy
       Data/CausalTimeData/Dataset/pm25/gen_data.npy
       Data/CausalTimeData/Dataset/pm25/graph.npy
       Data/CausalTimeData/Dataset/traffic/gen_data.npy
       Data/CausalTimeData/Dataset/traffic/graph.npy

See Data/README.md for every other dataset, and Data/causaltime.py's module docstring ("HOW TO GET THE DATA") for
details, and for how to point to a different location (base_dir parameter,
CAUSALTIME_DIR_PATH in Config/execution.py, or the CAUSALTIME_DIR
environment variable -- the latter has no effect on main.py and the other
entry points, which always pass CAUSALTIME_DIR_PATH).
