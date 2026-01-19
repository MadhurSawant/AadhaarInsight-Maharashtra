# =========================================================
# Aadhaar Activity Trends – Maharashtra
# FINAL DASH APPLICATION
# (LGD SAFE + CLICKABLE MAP + UX POLISHED)
# =========================================================

import json
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import joblib
from datetime import datetime
from dash.exceptions import PreventUpdate


# ================= PREDICTION MODELS =================
ENROL_MODEL = joblib.load("enrol_agewise_model.pkl")
DEMO_MODEL  = joblib.load("demo_agewise_model.pkl")
BIO_MODEL   = joblib.load("bio_agewise_model.pkl")

ENROL_RAW = pd.read_csv("MHEnrol_clean_agg.csv")
ENROL_RAW.columns = ENROL_RAW.columns.str.lower()
ENROL_RAW["district_lgd_code"] = ENROL_RAW["district_lgd_code"].astype(str)
ENROL_RAW["pincode"] = ENROL_RAW["pincode"].astype(str)

ENROL_FEATURES = [
    "month_index",
    "district_lgd_code",
    "pincode_enc",
    "district_label_enc",
    "age_0_5_lag1",
    "age_5_17_lag1",
    "age_18_greater_lag1",
]

def get_last_lag_values(lgd, pincode, metric):
    """
    Fetch last available lag values for a pincode
    """
    hist = df[df["district_lgd_code"] == lgd].copy()

    if hist.empty:
        return None

    hist = hist.sort_values("month_dt")

    if metric == "new_enrolments":
        return {
            "age_0_5_lag1": hist["age_0_5"].iloc[-1],
            "age_5_17_lag1": hist["age_5_17"].iloc[-1],
            "age_18_greater_lag1": hist["age_18_greater"].iloc[-1],
        }

    elif metric == "demographic_updates":
        return {
            "demo_age_5_17_lag1": hist["demo_age_5_17"].iloc[-1],
            "demo_age_17__lag1": hist["demo_age_17_"].iloc[-1],
        }

    else:  # biometric_updates
        return {
            "bio_age_5_17_lag1": hist["bio_age_5_17"].iloc[-1],
            "bio_age_17__lag1": hist["bio_age_17_"].iloc[-1],
        }


# ================= SHARED ENCODERS ==================
PIN_ENCODER = joblib.load("pincode_encoder.pkl")
DIST_LABEL_ENCODER = joblib.load("district_label_encoder.pkl")

UIDAI_COLORS = {
    "primary": "#0B3C5D",   # Navy – trust, authority
    "accent": "#F4A261",    # Saffron – national identity
    "success": "#2A9D8F",   # Green – positive / growth
    "bg": "#F7F9FC",        # Light grey background
    "card": "#FFFFFF",     # Card background
    "text": "#1F2933",     # Main text
    "muted": "#6B7280",    # Secondary labels
}
# =========================================================
# HELPERS
# =========================================================
def format_number(n):
    try:
        n = float(n)
    except Exception:
        return "0"
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.2f}K"
    return f"{int(n):,}"

def kpi(title, value, color):
    return html.Div(
        style={
            "background": UIDAI_COLORS["card"],   # was "white"
            "borderRadius": "14px",
            "padding": "14px",
            "boxShadow": "0 6px 18px rgba(0,0,0,0.08)",
            "textAlign": "center",
        },
        children=[
            html.H2(value, style={"margin": 0, "color": color}),
            html.P(
                title,
                style={"margin": 0, "color": UIDAI_COLORS["muted"]}  # was "#666"
            ),
        ],
    )

OFFICIAL_LABELS = {
    # Age groups
    "age_0_5": "Children (0–5 years)",
    "age_5_17": "Children & Adolescents (5–17 years)",
    "age_18_greater": "Adults (18+ years)",

    # Demographic update ages
    "demo_age_5_17": "Demographic Updates (5–17 years)",
    "demo_age_17_": "Demographic Updates (18+ years)",

    # Biometric update ages
    "bio_age_5_17": "Biometric Updates (5–17 years)",
    "bio_age_17_": "Biometric Updates (18+ years)",

    # Metrics
    "new_enrolments": "New Aadhaar Enrolments",
    "demographic_updates": "Demographic Updates",
    "biometric_updates": "Biometric Updates",
}

# =========================================================
# UIDAI / GOVERNMENT COLOR TOKENS (COLOR ONLY)
# =========================================================


# =========================================================
# LOAD DATA (LGD SAFE)
# =========================================================
def load_data():
    enrol = pd.read_csv("MHEnrol_clean_agg.csv")
    demo  = pd.read_csv("MHDemo_clean_agg.csv")
    bio   = pd.read_csv("MHBio_clean_agg.csv")

    for d in (enrol, demo, bio):
        d.columns = d.columns.str.lower()
        d["district_lgd_code"] = d["district_lgd_code"].astype(str)
        d["month_year"] = d["month_year"].astype(str)
        d["month_dt"] = pd.to_datetime(d["month_year"], format="%b-%y", errors="coerce")

    # ---------------- ENROL AGG ----------------
    enrol["new_enrolments"] = enrol.filter(regex="^age").sum(axis=1)

    enrol_agg = (
        enrol.groupby(["month_year","district_lgd_code","district_label"], as_index=False)
        .agg(
            new_enrolments=("new_enrolments","sum"),
            age_0_5=("age_0_5","sum"),
            age_5_17=("age_5_17","sum"),
            age_18_greater=("age_18_greater","sum"),
            month_dt=("month_dt","first"),
        )
    )

    # ---------------- DEMO AGG ----------------
    demo_agg = (
        demo.groupby(["month_year","district_lgd_code"], as_index=False)
        .agg(
            demographic_updates=("demo_age_5_17","sum"),
            demo_age_5_17=("demo_age_5_17","sum"),
            demo_age_17_=("demo_age_17_","sum"),
        )
    )

    # ---------------- BIO AGG ----------------
    bio_agg = (
        bio.groupby(["month_year","district_lgd_code"], as_index=False)
        .agg(
            biometric_updates=("bio_age_5_17","sum"),
            bio_age_5_17=("bio_age_5_17","sum"),
            bio_age_17_=("bio_age_17_","sum"),
        )
    )

    # ---------------- SAFE MERGE ----------------
    df = (
        enrol_agg
        .merge(demo_agg, on=["month_year","district_lgd_code"], how="left")
        .merge(bio_agg,  on=["month_year","district_lgd_code"], how="left")
        .fillna(0)
    )

    return df


df = load_data()

# =========================================================
# MASTER DATA
# =========================================================
DISTRICTS = (
    df[["district_lgd_code","district_label"]]
    .drop_duplicates()
    .sort_values("district_label")
)

LABEL_TO_LGD = dict(zip(DISTRICTS["district_label"], DISTRICTS["district_lgd_code"]))
LGD_TO_LABEL = dict(zip(DISTRICTS["district_lgd_code"], DISTRICTS["district_label"]))
MONTHS = sorted(df["month_year"].unique())

# =========================================================
# GEOJSON
# =========================================================
with open("maharashtra_districts_lgd_ready.geojson", encoding="utf-8") as f:
    MH_GEOJSON = json.load(f)

# =========================================================
# APP
# =========================================================
app = Dash(__name__ ,suppress_callback_exceptions=True)
app.title = "Aadhaar Activity Trends – Maharashtra"

# =========================================================
# LAYOUT (POSITION UNCHANGED)                
# =========================================================
def dashboard_layout():
    return html.Div(
        style={"backgroundColor": "#f4f6fb", "padding": "18px"},
        children=[
            html.H2(
                "Aadhaar Activity Trends – Maharashtra",
                style={"color": UIDAI_COLORS["primary"]}
            ),

            # ---------- KPIs ----------
            html.Div(
                style={"display":"grid","gridTemplateColumns":"repeat(3,1fr)","gap":"14px"},
                children=[
                    html.Div(id="kpi-enrol"),
                    html.Div(id="kpi-demo"),
                    html.Div(id="kpi-bio"),
                ],
            ),

            html.Br(),

            # ---------- FILTERS ----------
            html.Div(
                style={"display":"grid","gridTemplateColumns":"repeat(4,1fr)","gap":"12px"},
                children=[
                    dcc.Dropdown(
                        id="metric",
                        value="new_enrolments",
                        clearable=False,
                        options=[
                            {"label":"New Enrolments","value":"new_enrolments"},
                            {"label":"Demographic Updates","value":"demographic_updates"},
                            {"label":"Biometric Updates","value":"biometric_updates"},
                        ],
                    ),
                    dcc.Dropdown(
                        id="month",
                        value="ALL",
                        clearable=False,
                        options=[{"label":"All Months","value":"ALL"}] +
                                [{"label":m,"value":m} for m in MONTHS],
                    ),
                    dcc.Dropdown(
                        id="district",
                        value="ALL",
                        clearable=False,
                        options=[{"label":"All Maharashtra","value":"ALL"}] +
                                [{"label":d,"value":d} for d in DISTRICTS["district_label"]],
                    ),
                    dcc.Dropdown(
                        id="compare_district",
                        placeholder="Compare with…",
                        options=[{"label":d,"value":d} for d in DISTRICTS["district_label"]],
                    ),
                ],
            ),

            html.Br(),

            # ---------- LINE ----------
            dcc.Loading(type="cube", children=dcc.Graph(id="line-chart")),

            html.Br(),

            # ---------- BAR + PIE ----------
            html.Div(
                style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"18px"},
                children=[
                    dcc.Loading(type="cube", children=dcc.Graph(id="bar-chart")),
                    dcc.Loading(type="cube", children=dcc.Graph(id="pie-chart")),
                ],
            ),

            html.Br(),

            # ---------- MAP ----------
            dcc.Loading(type="cube", children=dcc.Graph(id="district-map")),
        ],
    )

def prediction_layout():
    return html.Div(
        style={"backgroundColor": "#f4f6fb", "padding": "18px"},
        children=[
            html.H2(
                "Future Aadhaar Activity Prediction",
                style={"color": UIDAI_COLORS["primary"]}
            ),

            html.Div(
                style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"14px"},
                children=[
                    dcc.Dropdown(
                        id="pred-district",
                        options=[{"label":d,"value":d} for d in DISTRICTS["district_label"]],
                        placeholder="Select District"
                    ),
                    dcc.Input(
                        id="pred-month",
                        placeholder="Future Month (YYYY-MM)",
                        type="text"
                    ),
                ],
            ),

            html.Br(),
            html.Button("Run Prediction", id="run-pred", n_clicks=0),
            html.Br(), html.Br(),

            html.Div(id="pred-output"),
        ],
    )


app.layout = html.Div(
    style={"backgroundColor": UIDAI_COLORS["bg"], "minHeight": "100vh"},
    children=[

        # ======= TOP PRODUCT HEADING =======
        html.Div(
            style={
                "padding": "18px 24px 6px 24px",
                "background": UIDAI_COLORS["card"],
                "borderBottom": "1px solid #E5E7EB",
            },
            children=[
                html.H1(
                    "AadhaarInsight",
                    style={
                        "margin": 0,
                        "color": UIDAI_COLORS["primary"],
                        "fontWeight": "700",
                        "letterSpacing": "0.5px",
                    },
                ),
                html.P(
                    "Aadhaar Activity Trends & Predictive Insights – Maharashtra",
                    style={
                        "margin": "4px 0 0 0",
                        "color": UIDAI_COLORS["muted"],
                        "fontSize": "14px",
                    },
                ),
            ],
        ),

        # ======= TABS =======
        dcc.Tabs(
            id="main-tabs",
            value="dashboard",
            children=[
                dcc.Tab(label="Dashboard", value="dashboard"),
                dcc.Tab(label="Prediction", value="prediction"),
            ],
        ),

        html.Div(id="tab-content"),
    ],
)

@app.callback(
    Output("tab-content", "children"),
    Input("main-tabs", "value")
)
def render_tab(tab):
    if tab == "prediction":
        return prediction_layout()
    return dashboard_layout()



# =========================================================
# CALLBACK
# =========================================================
@app.callback(
    Output("kpi-enrol","children"),
    Output("kpi-demo","children"),
    Output("kpi-bio","children"),
    Output("line-chart","figure"),
    Output("bar-chart","figure"),
    Output("pie-chart","figure"),
    Output("district-map","figure"),
    Output("district","value"),
    Input("metric","value"),
    Input("month","value"),
    Input("district","value"),
    Input("compare_district","value"),
    Input("district-map","clickData"),
)





def update_dashboard(metric, month, district, compare, clickData):

    # ---------- SAFETY GUARD ----------
    if not metric:
        raise PreventUpdate

    # ---------- DISTRICT SELECTION ----------
    selected = district
    if clickData and "location" in clickData["points"][0]:
        selected = LGD_TO_LABEL.get(
            str(clickData["points"][0]["location"]),
            district
        )

    dff = df.copy()
    if month != "ALL":
        dff = dff[dff["month_year"] == month]
    if selected != "ALL":
        dff = dff[dff["district_lgd_code"] == LABEL_TO_LGD[selected]]

    # ---------- KPIs ----------
    kpi_enrol = kpi(
        "New Enrolments",
        format_number(dff["new_enrolments"].sum()),
        "#2563eb"
    )
    kpi_demo = kpi(
        "Demographic Updates",
        format_number(dff["demographic_updates"].sum()),
        "#f59e0b"
    )
    kpi_bio = kpi(
        "Biometric Updates",
        format_number(dff["biometric_updates"].sum()),
        "#7c3aed"
    )

    # ---------- LINE ----------
     # ---------- LINE (SAFE + TWO COLORS) ----------

    base = dff.copy()
    base["series"] = selected if selected != "ALL" else "Maharashtra"

    if compare:
        cmp = df[df["district_lgd_code"] == LABEL_TO_LGD[compare]].copy()
        if month != "ALL":
            cmp = cmp[cmp["month_year"] == month]
        cmp["series"] = compare
        base = pd.concat([base, cmp])

    line_df = (
        base.groupby(["month_dt", "month_year", "series"], as_index=False)[metric]
        .sum()
        .sort_values("month_dt")
    )

    # 🎨 COLOR MAP (KEY FIX)
    COLOR_MAP = {
        "Maharashtra": UIDAI_COLORS["primary"],
    }

    if selected != "ALL":
        COLOR_MAP[selected] = UIDAI_COLORS["primary"]

    if compare:
        COLOR_MAP[compare] = UIDAI_COLORS["accent"]

    line_fig = px.line(
        line_df,
        x="month_year",
        y=metric,
        color="series",
        markers=True,
        color_discrete_map=COLOR_MAP,   # ✅ IMPORTANT
    )

    # ---------- STYLE (KEEP YOUR DESIGN) ----------
    line_fig.update_traces(
        marker=dict(
            size=8,
            line=dict(width=1, color="white")
        ),
        line=dict(width=3),
        hovertemplate=(
            "<b>%{fullData.name}</b><br>"
            "Month: %{x}<br>"
            f"{metric.replace('_',' ').title()}: %{{y:,.0f}}"
            "<extra></extra>"
        )
    )

    line_fig.update_layout(
        hovermode="x unified",
        template="plotly_white",
        yaxis_tickformat="~s",
        title=f"{metric.replace('_',' ').title()} Trend – {selected}",
    )

    # ---------- BAR ----------
    bar_df = (
        df.groupby("district_label", as_index=False)[metric]
        .sum()
        .sort_values(metric, ascending=False)
        .head(10)
    )

    bar_fig = px.bar(
        bar_df,
        x=metric,
        y="district_label",
        orientation="h",
        text=bar_df[metric].apply(format_number),
        color_discrete_sequence=[UIDAI_COLORS["primary"]],
        title="Top Districts (Overall)",
    )

    bar_fig.update_traces(
        hovertemplate=(
            "<b>%{y}</b><br>"
            f"{metric.replace('_',' ').title()}: %{{x:,.0f}}"
            "<extra></extra>"
        )
    )

    bar_fig.update_layout(
        template="plotly_white",
        yaxis=dict(autorange="reversed"),
        xaxis_tickformat="~s",
    )

    # ---------- PIE ----------
    age_cols = {
        "new_enrolments": ["age_0_5", "age_5_17", "age_18_greater"],
        "demographic_updates": ["demo_age_5_17", "demo_age_17_"],
        "biometric_updates": ["bio_age_5_17", "bio_age_17_"],
    }[metric]

    values = dff[age_cols].sum()
    total = values.sum()

    pie_fig = px.pie(
        names=values.index,
        values=values.values,
        hole=0.55,
        color_discrete_sequence=[
            UIDAI_COLORS["primary"],
            UIDAI_COLORS["accent"],
            UIDAI_COLORS["success"],
        ],
    )

    pie_fig.update_traces(
        labels=[OFFICIAL_LABELS.get(c, c) for c in values.index],
        textinfo="percent+label",
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Share: %{percent}<br>"
            "Count: %{value:,.0f}"
            "<extra></extra>"
        ),
    )

    pie_fig.update_layout(
        template="plotly_white",
        annotations=[dict(
            text=f"<b>{format_number(total)}</b><br>Total<br>{selected}",
            x=0.5, y=0.5, showarrow=False
        )]
    )

    # ---------- MAP ----------
    map_df = df.groupby(
        ["district_lgd_code", "district_label"], as_index=False
    )[metric].sum()

    map_fig = px.choropleth(
        map_df,
        geojson=MH_GEOJSON,
        locations="district_lgd_code",
        featureidkey="properties.district_lgd_code",
        color=metric,
        hover_name="district_label",
        color_continuous_scale=[
            UIDAI_COLORS["bg"],
            UIDAI_COLORS["primary"],
        ],
    )

    map_fig.update_traces(
        hovertemplate=(
            "<b>%{hovertext}</b><br>"
            f"{metric.replace('_',' ').title()}: %{{z:,.0f}}"
            "<extra></extra>"
        )
    )

    map_fig.update_geos(visible=False, fitbounds="locations")
    map_fig.update_layout(
        template="plotly_white",
        title=f"{metric.replace('_',' ').title()} by District",
    )

    return (
        kpi_enrol,
        kpi_demo,
        kpi_bio,
        line_fig,
        bar_fig,
        pie_fig,
        map_fig,
        selected,
    )



@app.callback(
    Output("pred-output", "children"),
    Input("run-pred", "n_clicks"),
    Input("pred-district", "value"),
    Input("pred-month", "date"),
    Input("pred-metric", "value"),
)
def run_prediction(n, district, month, metric):

    if not n:
        return ""

    if not district or not month or not metric:
        return html.P("⚠️ Please select district, metric, and month")

    # ---------- MONTH ----------
    try:
        dt = pd.to_datetime(month)
        month_index = dt.year * 12 + dt.month
    except Exception:
        return html.P("❌ Invalid month format (YYYY-MM)")

    # ---------- DISTRICT ----------
    lgd = LABEL_TO_LGD.get(district)
    if not lgd:
        return html.P("❌ Invalid district")

    # ---------- PINCODES ----------
    sub = (
        ENROL_RAW[ENROL_RAW["district_lgd_code"] == lgd]
        [["pincode", "district_label"]]
        .drop_duplicates()
    )

    rows = []
    for _, r in sub.iterrows():
        try:
            lag_vals = get_last_lag_values(lgd, r["pincode"], metric)
            if lag_vals is None:
                continue

            row = {
                "month_index": month_index,
                "district_lgd_code": lgd,
                "pincode_enc": PIN_ENCODER.transform([str(r["pincode"])])[0],
                "district_label_enc": DIST_LABEL_ENCODER.transform([r["district_label"]])[0],
            }
            row.update(lag_vals)
            rows.append(row)

        except Exception:
            continue

    if not rows:
        return html.P("❌ No valid pincodes found")

    X = pd.DataFrame(rows)

    # ---------- MODEL ----------
    if metric == "new_enrolments":
        model = ENROL_MODEL
        labels = ["0–5", "5–17", "18+"]
    elif metric == "demographic_updates":
        model = DEMO_MODEL
        labels = ["5–17", "18+"]
    else:
        model = BIO_MODEL
        labels = ["5–17", "18+"]

    # 🔑 Align features exactly as trained
    X = X[model.feature_names_in_]

    preds = model.predict(X)
    totals = preds.sum(axis=0)

    # ---------- UI OUTPUT ----------
    return html.Div(
        style={
            "background": UIDAI_COLORS["card"],
            "padding": "18px",
            "borderRadius": "14px",
            "boxShadow": "0 6px 18px rgba(0,0,0,0.08)",
            "marginTop": "12px",
        },
        children=[
            html.H4(
                f"Predicted {metric.replace('_',' ').title()} – {district}",
                style={"color": UIDAI_COLORS["primary"], "marginBottom": "14px"},
            ),

            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr 1fr",
                    "gap": "16px",
                    "alignItems": "center",
                },
                children=[
                    # ---------- PIE ----------
                    dcc.Graph(
                        figure=px.pie(
                            names=labels,
                            values=totals,
                            hole=0.6,
                            color_discrete_sequence=[
                                UIDAI_COLORS["primary"],
                                UIDAI_COLORS["accent"],
                                UIDAI_COLORS["success"],
                            ],
                        )
                        .update_traces(
                            textinfo="label+percent",
                            hovertemplate=(
                                "<b>%{label}</b><br>"
                                "Count: %{value:,.0f}<br>"
                                "Share: %{percent}"
                                "<extra></extra>"
                            ),
                        )
                        .update_layout(
                            showlegend=False,  # ✅ remove unnecessary legend
                            margin=dict(t=10, b=10, l=10, r=10),
                            template="plotly_white",
                        ),
                        config={"displayModeBar": False},
                        style={"height": "260px"},
                    ),

                    # ---------- VALUES ----------
                    html.Ul(
                        style={
                            "listStyle": "none",
                            "padding": 0,
                            "margin": "0 0 0 10px",
                            "fontSize": "16px",
                            "lineHeight": "1.8",
                            "color": UIDAI_COLORS["text"],
                        },
                        children=[
                            html.Li(
                                f"{labels[i]} : {format_number(totals[i])}",
                                style={"fontWeight": "600"},
                            )
                            for i in range(len(labels))
                        ],
                    ),
                ],
            ),
        ],
    )


def prediction_layout():
    return html.Div(
        style={"backgroundColor": "#f4f6fb", "padding": "18px"},
        children=[

            html.H2(
                "Future Aadhaar Activity Prediction",
                style={"color": UIDAI_COLORS["primary"]}
            ),

            # -------- INPUT ROW --------
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr 1fr 1fr",
                    "gap": "14px",
                },
                children=[

                    # District
                    dcc.Dropdown(
                        id="pred-district",
                        options=[
                            {"label": d, "value": d}
                            for d in DISTRICTS["district_label"]
                        ],
                        placeholder="Select District",
                        clearable=False,
                    ),

                    # Metric
                    dcc.Dropdown(
                        id="pred-metric",
                        options=[
                            {"label": "New Enrolments", "value": "new_enrolments"},
                            {"label": "Demographic Updates", "value": "demographic_updates"},
                            {"label": "Biometric Updates", "value": "biometric_updates"},
                        ],
                        placeholder="Select Metric",
                        clearable=False,
                    ),

                    # Calendar-style Month Picker
                    dcc.DatePickerSingle(
                        id="pred-month",
                        display_format="YYYY-MM",
                        placeholder="Select Month",
                        min_date_allowed=datetime.today(),
                    ),
                ],
            ),

            html.Br(),

            html.Button(
                "Run Prediction",
                id="run-pred",
                n_clicks=0,
                style={
                    "background": UIDAI_COLORS["primary"],
                    "color": "white",
                    "border": "none",
                    "padding": "10px 18px",
                    "borderRadius": "8px",
                    "cursor": "pointer",
                },
            ),

            html.Br(),
            html.Br(),

            html.Div(id="pred-output"),
        ],
    )


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    app.run(debug=True)
