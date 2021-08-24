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
print(models)
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
    x = kwargs.get("x", np.array([[0, 0, 0, 0]]))
    x = pd.DataFrame(
        x, columns=["USER_VOLUME", "USER_DENSITY", "PRODUCTION", "CONSUME_OTHERS"])
    # print(x)
    test_df = pd.DataFrame()
    classes = ['Alpha', 'Beta', 'Gamma']
    for i, pipeline in MODELS.items():
        # print(pipeline.na )
        modelName = ["{}".format(i)] * len(classes)
        proba = pipeline.predict_proba(x)[0]
        print(proba)
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
                ]),
                html.Div([
                    "PRODUCTION: ",
                    dcc.Input(id="production", value="100", type="number")
                ]),
                html.Div([
                    "CONSUME_OTHERS: ",
                    dcc.Input(id="consumer_others", value="100", type="number")
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
    Input(component_id="density", component_property="value"),
    Input(component_id="production", component_property="value"),
    Input(component_id="consumer_others", component_property="value")
)
def update_out_div(v0, v1, v2, v3):
    x = np.array([[float(v0), float(v1), float(v2), float(v3)]])
    df = getData(x=x)
    print(df)
    # df = pd.DataFrame({"MODEL_NAME": ["LOG"], "PROBABILITY": [1]})
    fig = px.bar(df, x="MODEL_NAME", y="PROBABILITY",
                 color="PREDICTION",
                 barmode="group"
                 )
    return fig

    # prediction = model.predict_proba(x)
    # prediction = prediction[0]
    # print(prediction)
    # df = pd.DataFrame(
    #     {"USER_ROLE": ["GAMMA", "BETA", "ALPHA"], "PROBABILITY": prediction})
    # fig = px.bar(df, x="USER_ROLE", y="PROBABILITY")
if __name__ == '__main__':
    app.run_server(host="0.0.0.0", port=5000, debug=True)
