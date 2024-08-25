from pathlib import Path
import json
import os
# Analytics
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

# CLI
from typer import Typer

# Graphics
from PIL import Image
import dash
from dash import ctx
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go

uuid_version = "f25277ca-a65b-41c3-97b9-67884dca1aaf/5"
uuid_version = "a4680205-fce5-4b9d-b9c9-86d96f3f2d5c/5"
DATASET_CACHE_DIR = os.path.join(os.environ['HOME'], f".computer_vision_dataset_cache/{uuid_version}/images")
OUTPUT_DIR = './output'
app = Typer()

cli = Typer()

@cli.command()
def main(captions_path: Path='data/iot/caption_comparison.csv'):
    # Load context
    print("Loading captions and images...")
    captions = pd.read_csv(captions_path)

    print("Initializing webapp...")
    app = dash.Dash("Caption Clusters")
    app.layout = html.Div(
        style={"display": "flex"},
        children=[
            html.Div(
                style={"flex": 2, "padding": "10px"},
                children=[
                    html.Div(
                        style={"display": "flex"},
                        children=[
                            html.Button(
                                "previous",
                                id="previous-image",
                                n_clicks=0,
                                style={"flex": 1, "padding": "10px"},
                            ),
                            html.Button(
                                "next",
                                id="next-image",
                                n_clicks=0,
                                style={"flex": 1, "padding": "10px"},
                            ),
                        ],
                    ),
                    html.Div(id="image-container", style={"textAlign": "center"}),
                ],
            ),
            dcc.Store(id="image-index", data=0),
        ],
    )

    @app.callback(
        Output("image-index", "data"),
        [
            Input("next-image", "n_clicks"),
            Input("previous-image", "n_clicks"),
        ],
        [State("image-index", "data")],
        prevent_initial_call=True,
    )
    def update_image_index(next_clicks, previous_clicks, current_index):
        trigger = ctx.triggered_id
        if trigger == "next-image":
            return current_index + 1
        elif trigger == "previous-image":
            return max(current_index - 1, 0)
        return current_index

    @app.callback(
        Output("image-container", "children"),
        Input("image-index", "data"),
    )
    def display_image(image_index):
        row = captions.iloc[image_index]
        image_path = os.path.join(DATASET_CACHE_DIR, row['file_name'])

        image = Image.open(image_path)
        return [
            html.H5(str(captions_path)),
            html.H5(row['file_name']),
            html.Img(src=image, style={"width": "640px", "height": "320px"}),
            html.Plaintext('GT:    '+row['target']),
            html.Plaintext('VGG19: '+row['vgg19']),
            html.Plaintext('RN152: '+row['rn152']),
            html.Plaintext('RN152 FT LR1e-5 EP1: '+row['rn152_ft_1']),
            html.Plaintext('RN152 FT LR1e-5 EP5: '+row['rn152_ft_5']),
            html.Plaintext('RN152 FT LR1e-5 EP10: '+row['rn152_ft_10']),
            html.Plaintext('RN152 FT LR1e-4 EP1: '+row['rn152_ft_highlr_1']),
            html.Plaintext('RN152 FT LR1e-4 EP5: '+row['rn152_ft_highlr_5']),
        ]


    app.run_server(
        debug=True,
    )


if __name__ == "__main__":
    cli()
