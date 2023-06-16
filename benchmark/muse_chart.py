import matplotlib.pyplot as plt
import pandas as pd
from argparse import ArgumentParser

df = pd.read_csv("artifacts/all.csv")

# round to GB
df["Max Memory"] = df["Max Memory"].apply(lambda x: round(x / 10**9, 2))

df["Median"] = df["Median"].apply(lambda x: round(x, 2))
df["Mean"] = df["Mean"].apply(lambda x: round(x, 2))

bar_width = 0.25

model_names = [
    "openMUSE/muse-laiona6-uvit-clip-220k",
    "williamberman/laiona6plus_uvit_clip_f8",
    "runwayml/stable-diffusion-v1-5",
]


def chart(device, component, compiled, plot_on, legend, y_axis_key, y_label, timesteps):
    filter = (
        (df["Device"] == device)
        & (df["Component"] == component)
        & (df["Compilation Type"] == compiled)
    )

    if timesteps is not None:
        filter = filter & (df["Timesteps"] == timesteps)

    fdf = df[filter]

    placement = range(6)

    def inc_placement():
        nonlocal placement
        placement = [x + bar_width for x in placement]

    for model_name in model_names:
        filter_ = fdf["Model Name"] == model_name

        ffdf = fdf[filter_]

        y_axis = ffdf[y_axis_key].tolist()

        for _ in range(6 - len(y_axis)):
            y_axis.append(0)

        bars = plot_on.bar(placement, y_axis, width=bar_width, label=f"{model_name}")

        for bar in bars:
            yval = bar.get_height()
            plot_on.text(
                bar.get_x() + bar.get_width() / 2,
                yval + 0.05,
                yval,
                ha="center",
                va="bottom",
                rotation=80,
            )

        inc_placement()

    plot_on.set_xlabel("Batch Size")
    plot_on.set_ylabel(y_label)
    plot_on.set_xticks([r + bar_width for r in range(6)], [1, 2, 4, 8, 16, 32])
    plot_on.set_title(f"{device}, {component}, compiled: {compiled}")

    if legend:
        plot_on.legend(fontsize="x-small")


"""
python muse_chart.py --component full --graphing time --timesteps 12
python muse_chart.py --component full --graphing time --timesteps 20
python muse_chart.py --component full --graphing memory --timesteps 12
python muse_chart.py --component full --graphing memory --timesteps 20

python muse_chart.py --component backbone --graphing time
python muse_chart.py --component backbone --graphing memory

python muse_chart.py --component vae --graphing time
python muse_chart.py --component vae --graphing memory
"""
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--component", required=True)
    parser.add_argument("--graphing", required=True)
    parser.add_argument("--timesteps", required=False, default=None)

    args = parser.parse_args()

    assert args.component in ["full", "backbone", "vae"]

    if args.component == "full":
        assert args.timesteps is not None
        args.timesteps = int(args.timesteps)

    if args.graphing == "time":
        y_axis_key = "Median"
        y_label = "Median Time (ms)"
    elif args.graphing == "memory":
        y_axis_key = "Max Memory"
        y_label = "Max Memory (GB)"
    else:
        assert False, args.graphing

    fig, axs = plt.subplots(4, 3, sharey="row")

    for row_idx, device in enumerate(["a100", "4090", "t4", "cpu"]):
        for col_idx, compiled in enumerate(["None", "default", "reduce-overhead"]):
            legend = row_idx == 0 and col_idx == 2
            chart(
                device,
                args.component,
                compiled,
                axs[row_idx, col_idx],
                legend,
                y_axis_key,
                y_label,
                args.timesteps,
            )

    plt.subplots_adjust(hspace=0.75, wspace=0.50)

    plt.show()
