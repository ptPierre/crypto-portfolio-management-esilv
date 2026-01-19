from __future__ import annotations

import os
import sys
import json
import pandas as pd

from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px

# --------------------------------------------------
# Fix PYTHONPATH pour que "src" soit visible
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # .../src/ui
SRC_DIR = os.path.dirname(BASE_DIR)                     # .../src
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from analytics.metrics import compute_metrics

# --------------------------------------------------
# Paths
# --------------------------------------------------
LOGS = "logs"
portfolio_csv = os.path.join(LOGS, "portfolio_history.csv")
trades_csv = os.path.join(LOGS, "trades.csv")
state_json = os.path.join(LOGS, "portfolio_state.json")
status_json = os.path.join(LOGS, "bot_status.json")
prices_csv = os.path.join(LOGS, "prices_long.csv")

# --------------------------------------------------
# Helpers robustes (lecture fichiers)
# --------------------------------------------------
def safe_read_csv(path: str, required_cols):
    if not os.path.exists(path):
        return pd.DataFrame(columns=required_cols)

    try:
        df = pd.read_csv(path)
    except Exception:
        # fichier en cours d’écriture / ligne partielle
        return pd.DataFrame(columns=required_cols)

    for c in required_cols:
        if c not in df.columns:
            df[c] = pd.NA

    return df


def safe_read_json(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def load_prices_for_symbol(symbol: str) -> pd.DataFrame:
    if symbol is None or not os.path.exists(prices_csv):
        return pd.DataFrame(columns=["datetime", "close", "symbol"])

    df = safe_read_csv(prices_csv, ["timestamp", "datetime", "symbol", "close"])
    df = df[df["symbol"] == symbol].copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).sort_values("datetime")
    return df


# --------------------------------------------------
# Dash app
# --------------------------------------------------
app = Dash(__name__)

GRAPH_HEIGHT = 320  # ✅ hauteur fixe pour éviter tout "growth" infini

app.layout = html.Div(
    style={"fontFamily": "Arial", "padding": "16px"},
    children=[
        html.H2("Crypto Portfolio Dashboard (Paper Trading)"),
        dcc.Interval(id="refresh", interval=2000, n_intervals=0),

        html.Div(id="bot-status", style={"background": "#f7f7f7", "padding": "10px"}),

        html.Div(style={"display": "flex", "gap": "18px"}, children=[
            html.Div(id="kpis", style={"flex": "1", "padding": "10px", "border": "1px solid #eee"}),

            # ✅ Graph déclaré UNE SEULE FOIS ici (au lieu de le recréer dans le callback)
            html.Div(
                style={"flex": "1", "padding": "10px", "border": "1px solid #eee"},
                children=[
                    html.H4("Allocation (USD)"),
                    dcc.Graph(
                        id="alloc-graph",
                        style={"height": f"{GRAPH_HEIGHT}px"},
                        config={"responsive": True, "displayModeBar": False},
                    ),
                ],
            ),
        ]),

        html.Hr(),

        html.Div(style={"display": "flex", "gap": "18px"}, children=[
            html.Div([
                html.H4("Equity Curve"),
                dcc.Graph(
                    id="equity-graph",
                    style={"height": f"{GRAPH_HEIGHT}px"},
                    config={"responsive": True, "displayModeBar": False},
                ),
            ], style={"flex": "2"}),

            html.Div([
                html.H4("Trade Log"),
                dash_table.DataTable(
                    id="trades-table",
                    page_size=12,
                    style_table={"overflowX": "auto"},
                    style_cell={"textAlign": "left", "fontSize": "12px"},
                )
            ], style={"flex": "2"}),
        ]),

        html.Hr(),

        html.Div(style={"display": "flex", "gap": "18px"}, children=[
            html.Div([
                html.H4("Current Positions"),
                dash_table.DataTable(
                    id="pos-table",
                    page_size=12,
                    style_table={"overflowX": "auto"},
                    style_cell={"textAlign": "left", "fontSize": "12px"},
                )
            ], style={"flex": "2"}),

            html.Div([
                html.H4("Price Chart"),
                dcc.Dropdown(id="asset-dropdown", options=[], value=None),
                dcc.Graph(
                    id="price-graph",
                    style={"height": f"{GRAPH_HEIGHT}px"},
                    config={"responsive": True, "displayModeBar": False},
                ),
            ], style={"flex": "2"}),
        ]),
    ],
)

# --------------------------------------------------
# Callback principal (ROBUSTE)
# --------------------------------------------------
@app.callback(
    Output("bot-status", "children"),
    Output("kpis", "children"),
    Output("alloc-graph", "figure"),          # ✅ figure only
    Output("equity-graph", "figure"),         # ✅ figure only
    Output("trades-table", "data"),
    Output("trades-table", "columns"),
    Output("pos-table", "data"),
    Output("pos-table", "columns"),
    Output("asset-dropdown", "options"),
    Output("asset-dropdown", "value"),
    Output("price-graph", "figure"),          # ✅ figure only
    Input("refresh", "n_intervals"),
    Input("asset-dropdown", "value"),
)
def refresh(_, selected_asset):
    try:
        # ---------- Lecture données ----------
        portfolio = safe_read_csv(
            portfolio_csv,
            ["timestamp", "account_value", "total_unrealized_pnl", "num_positions"]
        )
        trades = safe_read_csv(
            trades_csv,
            ["timestamp", "symbol", "side", "qty", "price", "success", "reason"]
        )
        prices_all = safe_read_csv(
            prices_csv,
            ["timestamp", "datetime", "symbol", "close"]
        )

        state = safe_read_json(state_json, {"cash": 0.0, "positions": {}})
        status = safe_read_json(status_json, {})

        # ---------- Bot status ----------
        running = status.get("running", False)
        mode = status.get("mode", "N/A")
        exchange = status.get("exchange", "N/A")
        timeframe = status.get("timeframe", "N/A")

        step = status.get("step", 0)
        max_steps = status.get("max_steps", 1)
        progress = 100 * step / max_steps if max_steps and max_steps > 0 else 0.0

        rebalance = status.get("rebalance_period", "N/A")
        dca_rate = status.get("dca_rate", 0.0)
        use_momentum = status.get("use_momentum", False)
        momentum_k = status.get("momentum_top_k", None)

        bot_status_block = html.Div([
            html.H4("Bot Status"),
            html.Div("🟢 Running" if running else "🔴 Stopped"),
            html.Div(f"Mode: {mode}"),
            html.Div(f"Exchange: {exchange}"),
            html.Div(f"Timeframe: {timeframe}"),
            html.Div(f"Progress: {step} / {max_steps} ({progress:.1f}%)"),
            html.Div(f"Rebalancing: {rebalance}"),
            html.Div(f"DCA: {'ON' if dca_rate > 0 else 'OFF'} ({dca_rate*100:.1f}%)"),
            html.Div(
                f"Momentum: {'ON' if use_momentum else 'OFF'}"
                + (f" (Top {momentum_k})" if momentum_k else "")
            ),
        ])

        # ---------- KPIs ----------
        tf = status.get("timeframe", "1h")
        if len(portfolio) > 2:
            equity_series = portfolio["account_value"].astype(float)
            metrics = compute_metrics(equity_series, tf)
            last_eq = float(equity_series.iloc[-1])
            last_unrl = float(portfolio["total_unrealized_pnl"].astype(float).iloc[-1])
        else:
            metrics = {"return_total": 0.0, "vol": 0.0, "max_drawdown": 0.0, "sharpe": 0.0}
            last_eq = 0.0
            last_unrl = 0.0

        cash = float(state.get("cash", 0.0))
        positions = state.get("positions", {})

        kpis = html.Div([
            html.Div(f"Equity: {last_eq:.2f}"),
            html.Div(f"Cash: {cash:.2f}"),
            html.Div(f"Unrealized PnL: {last_unrl:.2f}"),
            html.Hr(),
            html.Div(f"Total Return: {metrics['return_total']*100:.2f}%"),
            html.Div(f"Volatility: {metrics['vol']*100:.2f}%"),
            html.Div(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%"),
            html.Div(f"Sharpe: {metrics['sharpe']:.2f}"),
        ])

        # ---------- Equity curve ----------
        fig_eq = go.Figure()
        if len(portfolio) > 1:
            fig_eq.add_trace(go.Scatter(y=portfolio["account_value"].astype(float), mode="lines", name="Equity"))
        fig_eq.update_layout(
            height=GRAPH_HEIGHT,          # ✅ verrouille la taille
            autosize=False,               # ✅ évite resize cumulatif
            margin=dict(l=20, r=20, t=30, b=20),
        )

        # ---------- Last prices ----------
        last_price = {}
        if not prices_all.empty:
            tmp = prices_all.sort_values("timestamp").groupby("symbol").tail(1)
            for _, r in tmp.iterrows():
                last_price[str(r["symbol"])] = float(r["close"])

        # ---------- Allocation pie ----------
        alloc_vals = {}
        for sym, p in positions.items():
            qty = float(p.get("qty", 0.0))
            price = float(last_price.get(sym, 0.0))
            if qty > 0 and price > 0:
                alloc_vals[sym] = qty * price

        if alloc_vals:
            fig_alloc = px.pie(
                names=list(alloc_vals.keys()),
                values=list(alloc_vals.values()),
                title=None,
            )
        else:
            fig_alloc = go.Figure()

        fig_alloc.update_layout(
            height=GRAPH_HEIGHT,          # ✅ verrouille la taille
            autosize=False,
            margin=dict(l=20, r=20, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )

        # ---------- Trades table ----------
        trades_data = trades.tail(200).to_dict("records")
        trades_cols = [{"name": c, "id": c} for c in trades.columns]

        # ---------- Positions table ----------
        pos_rows = []
        for sym, p in positions.items():
            qty = float(p.get("qty", 0.0))
            entry = float(p.get("entry_price", 0.0))
            price = float(last_price.get(sym, 0.0))
            pos_rows.append({
                "symbol": sym,
                "qty": qty,
                "entry_price": entry,
                "price": price,
                "value_usd": qty * price if price > 0 else 0.0,
            })
        pos_df = pd.DataFrame(pos_rows)
        pos_data = pos_df.to_dict("records")
        pos_cols = [{"name": c, "id": c} for c in pos_df.columns] if not pos_df.empty else [{"name": "symbol", "id": "symbol"}]

        # ---------- Dropdown assets ----------
        trade_syms = trades["symbol"].dropna().astype(str).tolist() if "symbol" in trades else []
        assets = sorted(set(trade_syms + list(positions.keys()) + list(last_price.keys())))
        options = [{"label": a, "value": a} for a in assets]

        if selected_asset not in assets:
            selected_asset = assets[0] if assets else None

        # ---------- Price chart ----------
        price_df = load_prices_for_symbol(selected_asset)
        fig_price = go.Figure()
        if len(price_df) > 2:
            fig_price.add_trace(
                go.Scatter(x=price_df["datetime"], y=price_df["close"].astype(float), mode="lines", name="Price")
            )
        fig_price.update_layout(
            title=f"Price: {selected_asset}",
            height=GRAPH_HEIGHT,          # ✅ verrouille la taille
            autosize=False,
            margin=dict(l=20, r=20, t=30, b=20),
        )

        return (
            bot_status_block,
            kpis,
            fig_alloc,          # ✅ figure (pas dcc.Graph)
            fig_eq,             # ✅ figure
            trades_data,
            trades_cols,
            pos_data,
            pos_cols,
            options,
            selected_asset,
            fig_price,          # ✅ figure
        )

    except Exception as e:
        empty = go.Figure().update_layout(height=GRAPH_HEIGHT, autosize=False)
        return (
            html.Div(f"⚠️ Dashboard error: {e}"),
            html.Div("Error"),
            empty,
            empty,
            [],
            [],
            [],
            [],
            [],
            None,
            empty,
        )


# --------------------------------------------------
# Run
# --------------------------------------------------
if __name__ == "__main__":
    # debug=False évite certains comportements de resize/refresh bizarres en devtools
    app.run(debug=False)