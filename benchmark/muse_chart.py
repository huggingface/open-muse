from argparse import ArgumentParser

import matplotlib.pyplot as plt
import pandas as pd

"""
python benchmark/muse_chart.py --device a100
python benchmark/muse_chart.py --device 4090
"""

bar_width = 0.10


def main():
    parser = ArgumentParser()
    parser.add_argument("--device", choices=["4090", "a100"], required=True)

    args = parser.parse_args()

    y_axis_key = "Median"
    y_label = "Median Time (s)"

    df = pd.read_csv("benchmark/artifacts/all.csv")

    df["Median"] = df["Median"].apply(lambda x: round(x / 1000, 2))

    timesteps = [12, 20]
    resolutions = [256, 512]
    force_down_up_samples = [False, True]

    num_rows = len(timesteps) * len(force_down_up_samples)
    num_cols = len(resolutions)

    fig, axs = plt.subplots(num_rows, num_cols, sharey="row")

    for row_idx_1, timesteps_ in enumerate(timesteps):
        for row_idx_2, force_down_up_sample in enumerate(force_down_up_samples):
            row_idx = row_idx_1 * len(timesteps) + row_idx_2

            for col_idx, resolution in enumerate(resolutions):
                plot_on = axs[row_idx, col_idx]

                chart(
                    df=df,
                    device=args.device,
                    resolution=resolution,
                    force_down_up_sample=force_down_up_sample,
                    plot_on=plot_on,
                    y_axis_key=y_axis_key,
                    y_label=y_label,
                    timesteps=timesteps_,
                )

                if row_idx == 3 and col_idx == 1:
                    plot_on.legend(bbox_to_anchor=(1, -0.1), fontsize="x-small")

    plt.subplots_adjust(hspace=0.75, wspace=0.50)

    plt.show()


def chart(df, device, resolution, force_down_up_sample, plot_on, y_axis_key, y_label, timesteps):
    filter = (df["Device"] == device) & (df["Resolution"] == resolution)

    if timesteps is not None:
        filter = filter & (df["Timesteps"] == timesteps)

    fdf = df[filter]

    placement = range(2)

    def inc_placement():
        nonlocal placement
        placement = [x + bar_width + 0.05 for x in placement]

    (fdf["Model Name"] == "stable_diffusion_1_5") & (fdf["Use Xformers"] == False)

    for use_xformers in [False, True]:
        filter_ = (fdf["Model Name"] == "stable_diffusion_1_5") & (fdf["Use Xformers"] == use_xformers)

        plot_one_bar(
            fdf=fdf,
            filter_=filter_,
            plot_on=plot_on,
            placement=placement,
            label=f"stable_diffusion_1_5, use_xformers: {use_xformers}",
            y_axis_key=y_axis_key,
        )

        inc_placement()

    for use_xformers, use_fused_mlp, use_fused_residual_norm in [
        [False, False, False],
        [True, False, False],
        [True, True, True],
    ]:
        filter_ = (
            (fdf["Model Name"] == "muse")
            & (fdf["Use Xformers"] == use_xformers)
            & (fdf["Use Fused MLP"] == use_fused_mlp)
            & (fdf["Use Fused Residual Norm"] == use_fused_residual_norm)
            & (df["Force Down Up Sample"] == force_down_up_sample)
        )

        plot_one_bar(
            fdf=fdf,
            filter_=filter_,
            plot_on=plot_on,
            placement=placement,
            label=(
                f"muse, use_xformers: {use_xformers}, use_fused_mlp: {use_fused_mlp}, use_fused_residual_norm:"
                f" {use_fused_residual_norm}"
            ),
            y_axis_key=y_axis_key,
        )

        inc_placement()

    plot_on.set_xlabel("Batch Size")
    plot_on.set_ylabel(y_label)
    plot_on.set_xticks([r + bar_width for r in range(2)], [1, 8])
    plot_on.set_title(
        f"{device}, timesteps: {timesteps}, resolution: {resolution} muse downsampled {force_down_up_sample}"
    )


def plot_one_bar(fdf, filter_, plot_on, placement, label, y_axis_key):
    ffdf = fdf[filter_]

    y_axis = ffdf[y_axis_key].tolist()

    for _ in range(2 - len(y_axis)):
        y_axis.append(0)

    bars = plot_on.bar(placement, y_axis, width=bar_width, label=label)

    for bar in bars:
        yval = bar.get_height()
        plot_on.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.05,
            yval,
            ha="center",
            va="bottom",
            rotation=80,
            fontsize="small",
        )


if __name__ == "__main__":
    main()
