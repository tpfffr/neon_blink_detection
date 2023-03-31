import numpy as np

clip_names = np.array(
    [
        # "1000-2022-12-14-09-43-56-0fcac6d3",
        # "1002-2022-12-14-11-43-58-23e05b8c",
        # "1004-2022-12-14-13-14-14-c8a509b9",
        # "1005-2022-12-14-15-07-31-ba8d94d5",
        # "1010-2022-12-15-13-27-31-f46dcdd8",
        # "1140-2023-01-12-13-15-56-2f0172d2",
        # "1141-2023-01-12-14-17-58-470c61da",
        # "1142-2023-01-12-14-27-07-34f1fccf",
        # "1144-2023-01-12-16-36-04-2c1ecc99",
        # "1151-2023-01-13-12-03-16-bca271ec",
        # "1152-2023-01-13-13-03-33-ddabe2a5",
        # "1156-2023-01-13-15-15-36-93d791d5",
        # "1167-2023-01-16-15-28-05-761a32fb",
        # "1287-2023-01-31-11-44-32-c6118754",
        # "1190-2023-01-18-11-47-36-4663855c",
        # "1199-2023-01-19-10-39-37-cbad3d47",
        # "1201-2023-01-19-11-43-16-41f36271",
        # "1202-2023-01-19-13-17-50-3651315a",
        # "1219-2023-01-23-10-52-53-f0f1506f",
        # "1230-2023-01-24-10-34-24-8e573035",
        # "1250-2023-01-25-15-28-28-e49f5c02",
        # "1253-2023-01-26-11-39-39-c0de66cb",
        # "1280-2023-01-30-14-24-40-d879dc2a",
        # "1306-2023-02-02-10-44-25-4fec6abd",
        # "1304-2023-02-01-17-38-36-66aceb25",
        # "1303-2023-02-01-17-30-01-d3ee799c",
        # "1302-2023-02-01-16-38-55-bc30b710",
        # "1301-2023-02-01-15-38-42-70991a8d",
        # "1300-2023-02-01-15-30-06-808aa97c",
        # "1299-2023-02-01-14-37-00-65f38d85",
        # "1298-2023-02-01-14-29-40-08017935",
        # "1297-2023-02-01-13-37-41-668a3c8e",
        # "1294-2023-02-01-11-11-04-442d1d78",
        # "1293-2023-02-01-10-13-34-0c9638f8",
        # "1292-2023-01-31-17-44-04-c0a47014",
        # "1291-2023-01-31-17-39-55-a701abd4",
        # "1290-2023-01-31-13-22-58-3ef10981",
        # "1289-2023-01-31-13-15-32-56e437ef",
        # "1288-2023-01-31-12-15-03-89605fe1",
        # "1286-2023-01-31-10-43-45-210d7b9a",
        # "1284-2023-01-30-17-57-14-9d3b1e31",
        # "1283-2023-01-30-16-42-51-e08daf16",
        # "1282-2023-01-30-16-34-00-1249f6de",
        # "1308-2023-02-02-13-13-29-b929df4c",
        # "1279-2023-01-30-13-06-13-e24bbdcf",
        # "1276-2023-01-30-09-36-22-b48621f1",
        # "1275-2023-01-27-17-25-45-4122dd25",
        # "1274-2023-01-27-16-44-58-962f6780",
        # "1273-2023-01-27-16-37-52-f2d08d55",
        # "1267-2023-01-27-11-38-27-264bce1d",
        # "1266-2023-01-27-10-32-47-144fd890",
        # "1264-2023-01-27-09-36-12-df2f3ab1",
        # "1265-2023-01-27-10-37-53-f3b4f617",
        # "1262-2023-01-26-17-22-05-a4b97078",
        # "1261-2023-01-26-16-33-22-32c2c928",
        # "1260-2023-01-26-16-16-34-0165fc4e",
        # "1259-2023-01-26-15-23-42-69af6c5a",
        # "1258-2023-01-26-14-29-04-23d8a6a8",
        # "1255-2023-01-26-13-20-00-5137ed82",
        # "1254-2023-01-26-11-39-01-7f79f26c",
        # "1252-2023-01-26-09-38-13-5b1af4ce",
        # "1251-2023-01-25-16-37-59-6778eeb5",
        # "1249-2023-01-25-14-29-18-e54693e4",
        # "1248-2023-01-25-12-59-30-e6407072",
        # "1328-2023-02-06-09-45-58-f56cc84e",
        # "1327-2023-02-06-09-40-56-d5cf1243",
        # "1326-2023-02-03-17-45-57-8a41f628",
        # "1325-2023-02-03-17-46-35-11ad4b90",
        # "1322-2023-02-03-14-25-41-7883a14f",
        # "1318-2023-02-03-10-41-29-03c72c25",
        # "1317-2023-02-03-09-41-14-30f69b87",
        # "1316-2023-02-03-09-34-09-31ef7938",
        # "1311-2023-02-02-14-30-40-96be0928",
        # "1312-2023-02-02-16-21-47-cd28e8e0",
        # "1313-2023-02-02-16-31-53-67b1bdbe",
        "1329-2023-02-06-10-40-29-c4308424",
        "1332-2023-02-06-11-58-03-8fc75e25",
        "1333-2023-02-06-13-17-50-ebc1e021",
        "1336-2023-02-06-15-28-30-adb9dbfb",
        "1337-2023-02-06-15-36-31-211278ea",
        "1338-2023-02-06-16-29-00-adceb23d",
        "1339-2023-02-06-17-24-49-5b24f5ba",
        "1340-2023-02-06-17-37-00-e6218648",
        "1341-2023-02-07-10-38-42-d828e5ac",
        "1342-2023-02-07-10-43-55-b5709e1f",
        "1343-2023-02-07-11-37-02-214c1078",
        "1345-2023-02-07-13-19-29-f5930ed0",
        "1344-2023-02-07-13-19-49-d1517542",
        "1346-2023-02-07-14-22-59-b1384544",
        # "2023-03-01_09-59-07-2ea49126",  # kai bike
        # "2023-01-27_15-59-54-49a115d5",  # tom computer
        # "2023-02-01_11-45-11-7621531e",  # kai computer
        # "2023-01-27_16-10-14-a2a8cbe1",  # ryan discussing
        # "2023-01-27_16-15-26-57802f75",  # tom walking
        # "2023-01-27_16-24-04-eb4305b1",  # kai walking
        # "2023-01-27_16-31-52-5f743ed0",  # moritz snowboarding
        # "padel_tennis_neon_01-b922b245",  # mgg padel
        # "padel_tennis_neon_03-2ded8f56",  # mgg partner padel
    ]
)

np.save("/users/tom/git/neon_blink_detection/clip_list.npy", clip_names)
