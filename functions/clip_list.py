import numpy as np

clip_names = np.array(
    [
        "1000-2022-12-14-09-43-56-0fcac6d3",
        "1002-2022-12-14-11-43-58-23e05b8c",
        "1004-2022-12-14-13-14-14-c8a509b9",
        "1005-2022-12-14-15-07-31-ba8d94d5",
        "1010-2022-12-15-13-27-31-f46dcdd8",
        "1140-2023-01-12-13-15-56-2f0172d2",
        "1141-2023-01-12-14-17-58-470c61da",
        "1142-2023-01-12-14-27-07-34f1fccf",
        "1144-2023-01-12-16-36-04-2c1ecc99",
        "1151-2023-01-13-12-03-16-bca271ec",
        "1152-2023-01-13-13-03-33-ddabe2a5",
        "1156-2023-01-13-15-15-36-93d791d5",
        "1167-2023-01-16-15-28-05-761a32fb",
        "1287-2023-01-31-11-44-32-c6118754",
        "1190-2023-01-18-11-47-36-4663855c",
        "1199-2023-01-19-10-39-37-cbad3d47",
        "1201-2023-01-19-11-43-16-41f36271",
        "1202-2023-01-19-13-17-50-3651315a",
        # "2023-01-27_15-59-54-49a115d5",
        # "2023-01-27_16-10-14-a2a8cbe1",
        # "2023-01-27_16-15-26-57802f75",
        # "2023-01-27_16-24-04-eb4305b1",
        # "2023-01-27_16-31-52-5f743ed0",
    ]
)

np.save("/cluster/users/tom/git/neon_blink_detection/clip_list.npy", clip_names)
