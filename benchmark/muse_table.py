r"""
\begin{tabular}{|l|c|c|c|}
\hline
\textbf{ } & \textbf{Heading 1} & \textbf{Heading 2} & \textbf{Heading 3} \\ \hline
\textbf{Row 1} & Data 1 & Data 2 & Data 3 \\ \hline
\textbf{Row 2} & Data 4 & Data 5 & Data 6 \\ \hline
\textbf{Row 3} & Data 7 & Data 8 & Data 9 \\ \hline
\end{tabular}
"""

from argparse import ArgumentParser

import pandas as pd

"""
python benchmark/muse_table.py --device a100
python benchmark/muse_table.py --device 4090
"""


def main():
    parser = ArgumentParser()
    parser.add_argument("--device", choices=["4090", "a100"], required=True)
    parser.add_argument("--batch-size", type=int, choices=[1, 8], required=True)

    args = parser.parse_args()

    df = pd.read_csv("benchmark/artifacts/all.csv")

    df["Median"] = df["Median"].apply(lambda x: round(x / 1000, 2))

    print(table(df=df, device=args.device, batch_size=args.batch_size))


def table(df, device, batch_size):
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
        "sdxl; resolution 1024; timesteps 12": fdf[(fdf["Model Name"] == "sdxl") & (fdf["Timesteps"] == "12")].iloc[0][
            "Median"
        ],
        "sdxl; resolution 1024; timesteps 20": fdf[(fdf["Model Name"] == "sdxl") & (fdf["Timesteps"] == "20")].iloc[0][
            "Median"
        ],
        "ssd 1b; resolution 1024; timesteps 12": fdf[
            (fdf["Model Name"] == "ssd_1b") & (fdf["Timesteps"] == "12")
        ].iloc[0]["Median"],
        "ssd 1b; resolution 1024; timesteps 20": fdf[
            (fdf["Model Name"] == "ssd_1b") & (fdf["Timesteps"] == "20")
        ].iloc[0]["Median"],
        "wurst; resolution 1024; timesteps TODO": fdf[(fdf["Model Name"] == "wurst")].iloc[0]["Median"],
        "lcm; resolution 512; timesteps 4": fdf[(fdf["Model Name"] == "lcm") & (fdf["Timesteps"] == "4")].iloc[0][
            "Median"
        ],
        "lcm; resolution 512; timesteps 8": fdf[(fdf["Model Name"] == "lcm") & (fdf["Timesteps"] == "8")].iloc[0][
            "Median"
        ],
        "muse-256; resolution 256; timesteps 12": fdf[
            (fdf["Model Name"] == "muse") & (fdf["Resolution"] == 256) & (fdf["Timesteps"] == "12")
        ].iloc[0]["Median"],
        # "muse; resolution 256; timesteps 20": fdf[
        #     (fdf["Model Name"] == "muse") & (fdf["Resolution"] == 256) & (fdf["Timesteps"] == "20")
        # ].iloc[0]["Median"],
        "muse-512; resolution 512; timesteps 12": fdf[
            (fdf["Model Name"] == "muse") & (fdf["Resolution"] == 512) & (fdf["Timesteps"] == "12")
        ].iloc[0]["Median"],
        # "muse; resolution 512; timesteps 20": fdf[
        #     (fdf["Model Name"] == "muse") & (fdf["Resolution"] == 512) & (fdf["Timesteps"] == "20")
        # ].iloc[0]["Median"],
        "sd-turbo; resolution 512; timesteps 1": fdf[(fdf["Model Name"] == "sd_turbo")].iloc[0]["Median"],
        "sdxl-turbo; resolution 1024; timesteps 1": fdf[(fdf["Model Name"] == "sdxl_turbo")].iloc[0]["Median"],
    }

    chart_values = [x for x in chart_values.items()]
    chart_values = sorted(chart_values, key=lambda x: x[1])

    table = r"""
\begin{tabular}{|l|c|c|c|}
\hline
\textbf{ } & \textbf{inference time} & \textbf{timesteps} & \textbf{resolution} \\ \hline
"""

    for label, value in chart_values:
        # gross but we barely run this code
        model, resolution, timesteps = label.split(";")
        resolution = resolution.split(" ")[-1]
        timesteps = timesteps.split(" ")[-1]

        table += r"\textbf{" + f"{model}}} & {value} s & {timesteps} & {resolution}" + r" \\ \hline" + "\n"

    table += r"\end{tabular}"

    return table


if __name__ == "__main__":
    main()
