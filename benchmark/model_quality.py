import matplotlib.pyplot as plt
import argparse

cfg = [
    1.5,
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
    7.0,
    8.0,
    10.0,
    15.0,
    20.0,
]

fid_512 = [
    56.13683,
    48.3625,
    43.13792,
    42.07286,
    41.21331,
    41.21309,
    40.76164,
    40.51427,
    40.22781,
    39.66504,
    38.57083,
]

clip_512 = [
    23.168075,
    24.3268,
    25.29295,
    25.67775,
    25.93075,
    26.068925,
    26.15145,
    26.151175,
    26.26665,
    26.3845,
    26.402225,
]

isc_512 = [
    20.32828279489911,
    23.092083811105134,
    25.34707454898865,
    25.782333543568505,
    26.779519535473717,
    26.72532414371535,
    26.8378182891666,
    27.02354446351334,
    27.235757940256587,
    27.461719798190302,
    27.37252925955596,
]

fid_256 = [43.64503, 40.57112, 39.38306, 39.29915, 40.10225, 41.97274, 45.10721, 49.11104, 59.13854, 81.46585, 96.3426]

clip_256 = [
    24.191875,
    25.035825,
    25.689725,
    26.0217,
    26.1032,
    26.048225,
    25.90045,
    25.691,
    25.319,
    24.49525,
    23.915725,
]

isc_256 = [
    21.247120913990408,
    23.008063867685443,
    23.49288416726619,
    24.13530452474164,
    23.197031957136875,
    21.741427950979876,
    20.435789339047123,
    18.84057076723702,
    15.793238717380486,
    10.74857386855099,
    8.62769427725863,
]

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--fid", action="store_true")
    args.add_argument("--isc", action="store_true")
    args.add_argument("--clip", action="store_true")
    args = args.parse_args()

    if args.fid:
        plt.title(f"FID")
        plt.ylabel("FID Score (10k)")
        plt.plot(cfg, fid_256, marker="o", label="muse-256")
        plt.plot(cfg, fid_512, marker="o", label="muse-512")
    elif args.isc:
        plt.title(f"Inception Score")
        plt.ylabel("Inception Score (10k)")
        plt.plot(cfg, isc_256, marker="o", label="muse-256")
        plt.plot(cfg, isc_512, marker="o", label="muse-512")
    elif args.clip:
        plt.title(f"CLIP Score")
        plt.ylabel("CLIP Score (10k)")
        plt.plot(cfg, clip_256, marker="o", label="muse-256")
        plt.plot(cfg, clip_512, marker="o", label="muse-512")
    else:
        assert False

    plt.xlabel("cfg scale")
    plt.legend()

    # Show grid (optional)
    plt.grid(True)

    # Display the plot
    if args.fid:
        plt.savefig("./benchmark/artifacts/fid.png")
    elif args.isc: 
        plt.savefig("./benchmark/artifacts/isc.png")
    elif args.clip: 
        plt.savefig("./benchmark/artifacts/clip.png")
    else:
        assert False
