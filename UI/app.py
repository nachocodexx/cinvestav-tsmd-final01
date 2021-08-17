import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import numpy as np
import pickle
import os
# import path

# models_base_path = os.path.join(os.getcwd(), "UI", "models")
models_base_path = os.path.join(os.getcwd(), "models")

external_stylesheets = []
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
# ________________________________________________

models = os.listdir(models_base_path)
MODELS = {}
for model in models:
    modelName = model.split(".")[0]
    MODELS[modelName] = None
    modelPath = os.path.join(models_base_path, model)
    pipeline = pickle.load(open(modelPath, "rb"))
    MODELS[modelName] = pipeline
# print(MODELS)
# ___________________________________________________


def getData(**kwargs):
    x = kwargs.get("x", np.array([[0, 0]]))
    # print(x)
    test_df = pd.DataFrame()
    for i, pipeline in MODELS.items():
        classes = pipeline.classes_
        modelName = ["{}".format(i)] * len(classes)
        proba = pipeline.predict_proba(x)[0]
        tmp_df = pd.DataFrame(
            {
                "MODEL_NAME": modelName,
                "PROBABILITY": proba,
                "PREDICTION": classes
            }
        )
        test_df = pd.concat([tmp_df, test_df])
    return test_df
# __________________________________________________
# model0 = ["MODEL-0", "MODEL-0", "MODEL-0"]
# model0_proba = [0, .4, .6]
# model0_pred = ["GAMMA", "BETA", "ALPHA"]
# #
# model1 = ["MODEL-1", "MODEL-1", "MODEL-1"]
# model1_proba = [.2, .1, .7]
# model1_pred = ["GAMMA", "BETA", "ALPHA"]
# dff = pd.DataFrame({
#     "MODEL_NAME": model0 + model1,
#     "PROBABILITY": model0_proba + model1_proba,
#     "PREDICTION": model0_pred + model1_pred
# })

# fig = px.bar(dff, x="MODEL_NAME", y="PROBABILITY",
#              color="PREDICTION", barmode="group")


app.layout = html.Div([
    html.Div(
        id="header",
        children=[html.H2("Storage System - User classification"), ]
    ),
    html.Div(
        id="main",
        children=[
            html.Div(id="container", children=[
                html.Div([
                    "USER_VOLUME: ",
                    dcc.Input(id="volume", value="100", type="number")
                ]),
                html.Div([
                    "USER_DENSITY: ",
                    dcc.Input(id="density", value="100", type="number")
                ])
            ])
        ]
    ),
    html.Br(),
    # dcc.Graph(id="graph", figure=fig)
    dcc.Graph(id="graph")
])


@app.callback(
    Output(component_id="graph", component_property="figure"),
    Input(component_id="volume", component_property="value"),
    Input(component_id="density", component_property="value")
)
def update_out_div(value, value1):
    x = np.array([[float(value), float(value1)]])
    df = getData(x=x)
    # prediction = model.predict_proba(x)
    # prediction = prediction[0]
    # print(prediction)
    # df = pd.DataFrame(
    #     {"USER_ROLE": ["GAMMA", "BETA", "ALPHA"], "PROBABILITY": prediction})
    # fig = px.bar(df, x="USER_ROLE", y="PROBABILITY")
    fig = px.bar(df, x="MODEL_NAME", y="PROBABILITY",
                 color="PREDICTION", barmode="group")
    return fig


if __name__ == '__main__':
    app.run_server(host="0.0.0.0", port=5000, debug=True)
