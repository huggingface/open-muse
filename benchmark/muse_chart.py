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
    parser.add_argument("--batch-size", type=int, choices=[1, 8], required=True)

    args = parser.parse_args()

    df = pd.read_csv("benchmark/artifacts/all.csv")

    df["Median"] = df["Median"].apply(lambda x: round(x / 1000, 2))

    fig, axs = plt.subplots(1, 1, sharey="row")

    chart(df=df, device=args.device, batch_size=args.batch_size, plot_on=axs)

    axs.set_ylabel("Median Time (s)")
    axs.set_title(f"{args.device} Batch size: {args.batch_size}")

    plt.show()


def chart(df, device, batch_size, plot_on):
    fdf = df[
        (df["Device"] == device)
        & (df["Use Xformers"] == True)
        & ((df["Use Fused Residual Norm"] == True) | (df["Use Fused Residual Norm"].isna()))
        & (df["Batch Size"] == batch_size)
    ]

    chart_values = {
        # "stable diffusion 1.5; timesteps 12": fdf[
        #     (fdf["Model Name"] == "stable_diffusion_1_5") & (fdf["Timesteps"] == "12")
        # ].iloc[0]["Median"],
        "stable diffusion 1.5; resolution 512; timesteps 20": fdf[
            (fdf["Model Name"] == "stable_diffusion_1_5") & (fdf["Timesteps"] == "20")
        ].iloc[0]["Median"],
        "sdxl; resolution 1024; timesteps 12": fdf[(fdf["Model Name"] == "sdxl") & (fdf["Timesteps"] == "12")].iloc[0]["Median"],
        "sdxl; resolution 1024; timesteps 20": fdf[(fdf["Model Name"] == "sdxl") & (fdf["Timesteps"] == "20")].iloc[0]["Median"],
        "ssd 1b; resolution 1024; timesteps 12": fdf[(fdf["Model Name"] == "ssd_1b") & (fdf["Timesteps"] == "12")].iloc[0]["Median"],
        "ssd 1b; resolution 1024; timesteps 20": fdf[(fdf["Model Name"] == "ssd_1b") & (fdf["Timesteps"] == "20")].iloc[0]["Median"],
        "wurst; resolution 1024": fdf[(fdf["Model Name"] == "wurst")].iloc[0]["Median"],
        "lcm; resolution 512; timesteps 4": fdf[(fdf["Model Name"] == "lcm") & (fdf["Timesteps"] == "4")].iloc[0]["Median"],
        "lcm; resolution 512; timesteps 8": fdf[(fdf["Model Name"] == "lcm") & (fdf["Timesteps"] == "8")].iloc[0]["Median"],
        "muse; resolution 256; timesteps 12": fdf[
            (fdf["Model Name"] == "muse") & (fdf["Resolution"] == 256) & (fdf["Timesteps"] == "12")
        ].iloc[0]["Median"],
        # "muse; resolution 256; timesteps 20": fdf[
        #     (fdf["Model Name"] == "muse") & (fdf["Resolution"] == 256) & (fdf["Timesteps"] == "20")
        # ].iloc[0]["Median"],
        "muse; resolution 512; timesteps 12": fdf[
            (fdf["Model Name"] == "muse") & (fdf["Resolution"] == 512) & (fdf["Timesteps"] == "12")
        ].iloc[0]["Median"],
        # "muse; resolution 512; timesteps 20": fdf[
        #     (fdf["Model Name"] == "muse") & (fdf["Resolution"] == 512) & (fdf["Timesteps"] == "20")
        # ].iloc[0]["Median"],
        "sd-turbo; resolution 512; timesteps 1": fdf[
            (fdf["Model Name"] == "sd_turbo")
        ].iloc[0]['Median'],
        "sdxl-turbo; resolution 1024; timesteps 1": fdf[
            (fdf["Model Name"] == "sdxl_turbo")
        ].iloc[0]['Median'],
    }

    # Gives consistent colors from chart to chart
    colors = [
        # "b",  # Blue
        "g",  # Green
        "r",  # Red
        "c",  # Cyan
        "m",  # Magenta
        "y",  # Yellow
        "k",  # Black
        "purple",
        "#FF5733",  # Hex code for a shade of orange
        (0.2, 0.4, 0.6),  # RGB tuple for a shade of blue
        "lime",  # Named color
        "navy",  # Named color
        "hotpink",  # Named color
    ]

    colors = {x: y for x, y in zip(chart_values.keys(), colors)}

    chart_values = [x for x in chart_values.items()]
    chart_values = sorted(chart_values, key=lambda x: x[1])

    placement = 0

    for label, value in chart_values:
        color = colors[label]

        # gross but we barely run this code
        label = f"{label}; {value} s"
        label = "\n".join(label.split(";"))

        bars = plot_on.bar(placement, value, width=bar_width, label=label, color=color)
        bar = bars[0]
        yval = bar.get_height()
        plot_on.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.05,
            label,  # f"{label}; {value}", # yval,
            ha="center",
            va="bottom",
            fontsize="small",
        )
        placement = placement + bar_width + 0.05


if __name__ == "__main__":
    main()
