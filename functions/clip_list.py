import numpy as np

clip_names = np.array(
    [
        "1000-2022-12-14-09-43-56-0fcac6d3",
        "1002-2022-12-14-11-43-58-23e05b8c",
        "1004-2022-12-14-13-14-14-c8a509b9",
        "1005-2022-12-14-15-07-31-ba8d94d5",
        "1005-2022-12-14-15-07-31-ba8d94d5",  # BE CAREFUL THIS IS DOULBE!!!!
    ]
)

np.save("/cluster/users/tom/git/neon_blink_detection/clip_list.npy", clip_names)
