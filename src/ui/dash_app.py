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
# Fix PYTHONPATH
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(BASE_DIR)
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
# Helpers
# --------------------------------------------------
def safe_read_csv(path: str, required_cols):
    if not os.path.exists(path):
        return pd.DataFrame(columns=required_cols)
    try:
        df = pd.read_csv(path)
    except Exception:
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


def fmt_num(val, decimals=2):
    """Format number with thousands separator."""
    if val is None:
        return "—"
    return f"{val:,.{decimals}f}"


def fmt_pct(val):
    """Format as percentage."""
    if val is None:
        return "—"
    return f"{val * 100:.2f}%"


# --------------------------------------------------
# Styles
# --------------------------------------------------
COLORS = {
    "bg": "#0f1117",
    "card": "#1a1d24",
    "border": "#2d323c",
    "text": "#e6e9ef",
    "text_muted": "#8b929e",
    "accent": "#3b82f6",
    "green": "#22c55e",
    "red": "#ef4444",
    "yellow": "#eab308",
}

FONT = "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"

# --------------------------------------------------
# Dash app
# --------------------------------------------------
app = Dash(__name__)

app.layout = html.Div(
    style={
        "fontFamily": FONT,
        "backgroundColor": COLORS["bg"],
        "color": COLORS["text"],
        "minHeight": "100vh",
        "padding": "20px",
        "boxSizing": "border-box",
    },
    children=[
        # Header
        html.Div(
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "center",
                "marginBottom": "20px",
                "paddingBottom": "16px",
                "borderBottom": f"1px solid {COLORS['border']}",
            },
            children=[
                html.H1(
                    "Portfolio Dashboard",
                    style={
                        "margin": 0,
                        "fontSize": "24px",
                        "fontWeight": "600",
                        "letterSpacing": "-0.5px",
                    },
                ),
                html.Div(id="status-badge"),
            ],
        ),
        
        dcc.Interval(id="refresh", interval=2000, n_intervals=0),

        # Top row: KPIs + Status
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr 1fr 280px", "gap": "12px", "marginBottom": "16px"},
            children=[
                html.Div(id="kpi-equity", style={"backgroundColor": COLORS["card"], "borderRadius": "8px", "padding": "16px", "border": f"1px solid {COLORS['border']}"}),
                html.Div(id="kpi-return", style={"backgroundColor": COLORS["card"], "borderRadius": "8px", "padding": "16px", "border": f"1px solid {COLORS['border']}"}),
                html.Div(id="kpi-drawdown", style={"backgroundColor": COLORS["card"], "borderRadius": "8px", "padding": "16px", "border": f"1px solid {COLORS['border']}"}),
                html.Div(id="kpi-sharpe", style={"backgroundColor": COLORS["card"], "borderRadius": "8px", "padding": "16px", "border": f"1px solid {COLORS['border']}"}),
                html.Div(id="bot-status", style={"backgroundColor": COLORS["card"], "borderRadius": "8px", "padding": "16px", "border": f"1px solid {COLORS['border']}"}),
            ],
        ),

        # Middle row: Equity + Allocation
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "2fr 1fr", "gap": "12px", "marginBottom": "16px"},
            children=[
                html.Div(
                    style={"backgroundColor": COLORS["card"], "borderRadius": "8px", "padding": "16px", "border": f"1px solid {COLORS['border']}"},
                    children=[
                        html.Div("Equity Curve", style={"fontSize": "14px", "fontWeight": "500", "marginBottom": "12px", "color": COLORS["text_muted"]}),
                        dcc.Graph(id="equity-graph", style={"height": "220px"}, config={"displayModeBar": False}),
                    ],
                ),
                html.Div(
                    style={"backgroundColor": COLORS["card"], "borderRadius": "8px", "padding": "16px", "border": f"1px solid {COLORS['border']}"},
                    children=[
                        html.Div("Allocation", style={"fontSize": "14px", "fontWeight": "500", "marginBottom": "12px", "color": COLORS["text_muted"]}),
                        dcc.Graph(id="alloc-graph", style={"height": "220px"}, config={"displayModeBar": False}),
                    ],
                ),
            ],
        ),

        # Bottom row: Positions + Price Chart + Trades
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "12px"},
            children=[
                html.Div(
                    style={"backgroundColor": COLORS["card"], "borderRadius": "8px", "padding": "16px", "border": f"1px solid {COLORS['border']}", "overflow": "hidden"},
                    children=[
                        html.Div("Positions", style={"fontSize": "14px", "fontWeight": "500", "marginBottom": "12px", "color": COLORS["text_muted"]}),
                        html.Div(
                            style={"maxHeight": "240px", "overflowY": "auto"},
                            children=[
                                dash_table.DataTable(
                                    id="pos-table",
                                    page_size=50,
                                    style_table={"overflowX": "hidden"},
                                    style_header={
                                        "backgroundColor": COLORS["bg"],
                                        "color": COLORS["text_muted"],
                                        "fontWeight": "500",
                                        "fontSize": "11px",
                                        "textTransform": "uppercase",
                                        "letterSpacing": "0.5px",
                                        "border": "none",
                                        "padding": "8px 12px",
                                    },
                                    style_cell={
                                        "backgroundColor": COLORS["card"],
                                        "color": COLORS["text"],
                                        "border": "none",
                                        "fontSize": "12px",
                                        "padding": "8px 12px",
                                        "textAlign": "left",
                                        "maxWidth": "120px",
                                        "overflow": "hidden",
                                        "textOverflow": "ellipsis",
                                    },
                                    style_data_conditional=[
                                        {"if": {"row_index": "odd"}, "backgroundColor": COLORS["bg"]},
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    style={"backgroundColor": COLORS["card"], "borderRadius": "8px", "padding": "16px", "border": f"1px solid {COLORS['border']}"},
                    children=[
                        html.Div(
                            style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "12px"},
                            children=[
                                html.Div("Price Chart", style={"fontSize": "14px", "fontWeight": "500", "color": COLORS["text_muted"]}),
                                dcc.Dropdown(
                                    id="asset-dropdown",
                                    options=[],
                                    value=None,
                                    style={"width": "120px"},
                                    className="dark-dropdown",
                                ),
                            ],
                        ),
                        dcc.Graph(id="price-graph", style={"height": "220px"}, config={"displayModeBar": False}),
                    ],
                ),
                html.Div(
                    style={"backgroundColor": COLORS["card"], "borderRadius": "8px", "padding": "16px", "border": f"1px solid {COLORS['border']}", "overflow": "hidden"},
                    children=[
                        html.Div("Recent Trades", style={"fontSize": "14px", "fontWeight": "500", "marginBottom": "12px", "color": COLORS["text_muted"]}),
                        html.Div(
                            style={"maxHeight": "240px", "overflowY": "auto"},
                            children=[
                                dash_table.DataTable(
                                    id="trades-table",
                                    page_size=50,
                                    style_table={"overflowX": "hidden"},
                                    style_header={
                                        "backgroundColor": COLORS["bg"],
                                        "color": COLORS["text_muted"],
                                        "fontWeight": "500",
                                        "fontSize": "11px",
                                        "textTransform": "uppercase",
                                        "letterSpacing": "0.5px",
                                        "border": "none",
                                        "padding": "8px 12px",
                                    },
                                    style_cell={
                                        "backgroundColor": COLORS["card"],
                                        "color": COLORS["text"],
                                        "border": "none",
                                        "fontSize": "12px",
                                        "padding": "8px 12px",
                                        "textAlign": "left",
                                        "maxWidth": "100px",
                                        "overflow": "hidden",
                                        "textOverflow": "ellipsis",
                                    },
                                    style_data_conditional=[
                                        {"if": {"row_index": "odd"}, "backgroundColor": COLORS["bg"]},
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)

# Custom CSS for dropdown
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Portfolio Dashboard</title>
        {%favicon%}
        {%css%}
        <style>
            * { box-sizing: border-box; }
            body { margin: 0; background: #0f1117; }
            
            .dark-dropdown .Select-control {
                background-color: #1a1d24 !important;
                border-color: #2d323c !important;
            }
            .dark-dropdown .Select-menu-outer {
                background-color: #1a1d24 !important;
                border-color: #2d323c !important;
            }
            .dark-dropdown .Select-value-label,
            .dark-dropdown .Select-placeholder {
                color: #e6e9ef !important;
            }
            .dark-dropdown .Select-option {
                background-color: #1a1d24 !important;
                color: #e6e9ef !important;
            }
            .dark-dropdown .Select-option:hover {
                background-color: #2d323c !important;
            }
            
            /* Scrollbar styling */
            ::-webkit-scrollbar { width: 6px; height: 6px; }
            ::-webkit-scrollbar-track { background: #1a1d24; }
            ::-webkit-scrollbar-thumb { background: #2d323c; border-radius: 3px; }
            ::-webkit-scrollbar-thumb:hover { background: #3d424c; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''


def create_kpi_card(label: str, value: str, sub: str = None, color: str = None):
    """Create a KPI card content."""
    return html.Div([
        html.Div(label, style={"fontSize": "11px", "color": COLORS["text_muted"], "textTransform": "uppercase", "letterSpacing": "0.5px", "marginBottom": "4px"}),
        html.Div(value, style={"fontSize": "22px", "fontWeight": "600", "color": color or COLORS["text"], "letterSpacing": "-0.5px"}),
        html.Div(sub, style={"fontSize": "11px", "color": COLORS["text_muted"], "marginTop": "4px"}) if sub else None,
    ])


def create_plot_layout():
    """Base layout for all plots."""
    return {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": COLORS["text_muted"], "size": 10},
        "margin": {"l": 40, "r": 16, "t": 8, "b": 32},
        "xaxis": {"gridcolor": COLORS["border"], "zerolinecolor": COLORS["border"], "tickfont": {"size": 9}},
        "yaxis": {"gridcolor": COLORS["border"], "zerolinecolor": COLORS["border"], "tickfont": {"size": 9}},
    }


# --------------------------------------------------
# Callback
# --------------------------------------------------
@app.callback(
    Output("status-badge", "children"),
    Output("kpi-equity", "children"),
    Output("kpi-return", "children"),
    Output("kpi-drawdown", "children"),
    Output("kpi-sharpe", "children"),
    Output("bot-status", "children"),
    Output("alloc-graph", "figure"),
    Output("equity-graph", "figure"),
    Output("trades-table", "data"),
    Output("trades-table", "columns"),
    Output("pos-table", "data"),
    Output("pos-table", "columns"),
    Output("asset-dropdown", "options"),
    Output("asset-dropdown", "value"),
    Output("price-graph", "figure"),
    Input("refresh", "n_intervals"),
    Input("asset-dropdown", "value"),
)
def refresh(_, selected_asset):
    try:
        # Load data
        portfolio = safe_read_csv(portfolio_csv, ["timestamp", "account_value", "total_unrealized_pnl", "num_positions"])
        trades = safe_read_csv(trades_csv, ["timestamp", "symbol", "side", "qty", "price", "success", "reason"])
        prices_all = safe_read_csv(prices_csv, ["timestamp", "datetime", "symbol", "close"])
        state = safe_read_json(state_json, {"cash": 0.0, "positions": {}})
        status = safe_read_json(status_json, {})

        # Status
        running = status.get("running", False)
        mode = status.get("mode", "N/A")
        step = status.get("step", 0)
        max_steps = status.get("max_steps", 1)
        progress = 100 * step / max_steps if max_steps and max_steps > 0 else 0.0

        # Status badge
        badge_color = COLORS["green"] if running else COLORS["red"]
        status_badge = html.Div(
            style={"display": "flex", "alignItems": "center", "gap": "8px"},
            children=[
                html.Div(style={"width": "8px", "height": "8px", "borderRadius": "50%", "backgroundColor": badge_color}),
                html.Span("Running" if running else "Stopped", style={"fontSize": "13px", "color": COLORS["text_muted"]}),
            ],
        )

        # Metrics
        tf = status.get("timeframe", "1h")
        if len(portfolio) > 2:
            equity_series = portfolio["account_value"].astype(float)
            metrics = compute_metrics(equity_series, tf)
            last_eq = float(equity_series.iloc[-1])
            last_unrl = float(portfolio["total_unrealized_pnl"].astype(float).iloc[-1])
        else:
            metrics = {"return_total": 0.0, "vol": 0.0, "max_drawdown": 0.0, "sharpe": 0.0}
            last_eq = float(state.get("cash", 0.0))
            last_unrl = 0.0

        cash = float(state.get("cash", 0.0))
        positions = state.get("positions", {})

        # KPIs
        ret_color = COLORS["green"] if metrics["return_total"] >= 0 else COLORS["red"]
        kpi_equity = create_kpi_card("Equity", f"${fmt_num(last_eq)}", f"Cash: ${fmt_num(cash)}")
        kpi_return = create_kpi_card("Total Return", fmt_pct(metrics["return_total"]), f"Vol: {fmt_pct(metrics['vol'])}", ret_color)
        kpi_drawdown = create_kpi_card("Max Drawdown", fmt_pct(metrics["max_drawdown"]), None, COLORS["red"] if metrics["max_drawdown"] < -0.1 else None)
        kpi_sharpe = create_kpi_card("Sharpe Ratio", fmt_num(metrics["sharpe"]), None, COLORS["green"] if metrics["sharpe"] > 1 else None)

        # Bot status panel
        rebalance = status.get("rebalance_period", "N/A")
        dca_rate = status.get("dca_rate", 0.0)
        use_momentum = status.get("use_momentum", False)
        momentum_k = status.get("momentum_top_k", None)

        bot_status = html.Div([
            html.Div(f"{mode}", style={"fontSize": "11px", "color": COLORS["text_muted"], "marginBottom": "8px"}),
            html.Div(f"{progress:.0f}%", style={"fontSize": "18px", "fontWeight": "600", "marginBottom": "4px"}),
            html.Div(
                style={"height": "4px", "backgroundColor": COLORS["bg"], "borderRadius": "2px", "overflow": "hidden", "marginBottom": "8px"},
                children=[html.Div(style={"width": f"{progress}%", "height": "100%", "backgroundColor": COLORS["accent"]})]
            ),
            html.Div(style={"fontSize": "10px", "color": COLORS["text_muted"], "lineHeight": "1.6"}, children=[
                html.Div(f"Rebalance: {rebalance}"),
                html.Div(f"DCA: {dca_rate*100:.0f}%  •  Mom: {'Top ' + str(momentum_k) if use_momentum else 'Off'}"),
            ]),
        ])

        # Last prices
        last_price = {}
        if not prices_all.empty:
            tmp = prices_all.sort_values("timestamp").groupby("symbol").tail(1)
            for _, r in tmp.iterrows():
                last_price[str(r["symbol"])] = float(r["close"])

        # Allocation pie
        alloc_vals = {}
        for sym, p in positions.items():
            qty = float(p.get("qty", 0.0))
            price = float(last_price.get(sym, 0.0))
            if qty > 0 and price > 0:
                alloc_vals[sym] = qty * price

        if alloc_vals:
            # Sort and take top 10, group rest as "Other"
            sorted_alloc = sorted(alloc_vals.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_alloc) > 10:
                top_10 = dict(sorted_alloc[:10])
                other = sum(v for _, v in sorted_alloc[10:])
                top_10["Other"] = other
                alloc_vals = top_10

            fig_alloc = go.Figure(data=[go.Pie(
                labels=list(alloc_vals.keys()),
                values=list(alloc_vals.values()),
                hole=0.6,
                textinfo="percent",
                textfont={"size": 9, "color": COLORS["text"]},
                marker={"colors": px.colors.qualitative.Set3, "line": {"color": COLORS["card"], "width": 2}},
                showlegend=True,
            )])
            fig_alloc.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": COLORS["text_muted"], "size": 10},
                legend={"font": {"size": 9}, "orientation": "v", "x": 1, "y": 0.5},
                margin={"l": 8, "r": 100, "t": 8, "b": 8},
            )
        else:
            fig_alloc = go.Figure()
            fig_alloc.update_layout(**create_plot_layout())

        # Equity curve
        fig_eq = go.Figure()
        if len(portfolio) > 1:
            fig_eq.add_trace(go.Scatter(
                y=portfolio["account_value"].astype(float),
                mode="lines",
                line={"color": COLORS["accent"], "width": 2},
                fill="tozeroy",
                fillcolor="rgba(59, 130, 246, 0.1)",
            ))
        fig_eq.update_layout(**create_plot_layout())

        # Trades table (simplified columns)
        trades_display = trades.tail(100).copy()
        if not trades_display.empty:
            trades_display = trades_display[["timestamp", "symbol", "side", "qty", "price"]].copy()
            trades_display["timestamp"] = trades_display["timestamp"].astype(str).str[:16]
            trades_display["qty"] = trades_display["qty"].apply(lambda x: f"{float(x):.4f}" if pd.notna(x) else "")
            trades_display["price"] = trades_display["price"].apply(lambda x: f"{float(x):.2f}" if pd.notna(x) else "")

        trades_data = trades_display.to_dict("records") if not trades_display.empty else []
        trades_cols = [{"name": c.upper(), "id": c} for c in ["timestamp", "symbol", "side", "qty", "price"]]

        # Positions table (simplified)
        pos_rows = []
        for sym, p in positions.items():
            qty = float(p.get("qty", 0.0))
            price = float(last_price.get(sym, 0.0))
            value = qty * price if price > 0 else 0.0
            if qty > 0:
                pos_rows.append({
                    "symbol": sym,
                    "qty": f"{qty:.4f}",
                    "price": f"${price:.2f}",
                    "value": f"${value:.2f}",
                })
        pos_rows = sorted(pos_rows, key=lambda x: float(x["value"].replace("$", "").replace(",", "")), reverse=True)
        pos_cols = [{"name": c.upper(), "id": c} for c in ["symbol", "qty", "price", "value"]]

        # Asset dropdown
        trade_syms = trades["symbol"].dropna().astype(str).tolist() if "symbol" in trades.columns else []
        assets = sorted(set(trade_syms + list(positions.keys()) + list(last_price.keys())))
        options = [{"label": a, "value": a} for a in assets]
        if selected_asset not in assets:
            selected_asset = assets[0] if assets else None

        # Price chart
        price_df = load_prices_for_symbol(selected_asset)
        fig_price = go.Figure()
        if len(price_df) > 2:
            fig_price.add_trace(go.Scatter(
                x=price_df["datetime"],
                y=price_df["close"].astype(float),
                mode="lines",
                line={"color": COLORS["accent"], "width": 1.5},
            ))
        fig_price.update_layout(**create_plot_layout())
        fig_price.update_layout(title={"text": selected_asset or "", "font": {"size": 12}, "x": 0.5})

        return (
            status_badge,
            kpi_equity,
            kpi_return,
            kpi_drawdown,
            kpi_sharpe,
            bot_status,
            fig_alloc,
            fig_eq,
            trades_data,
            trades_cols,
            pos_rows,
            pos_cols,
            options,
            selected_asset,
            fig_price,
        )

    except Exception as e:
        empty = go.Figure().update_layout(**create_plot_layout())
        return (
            html.Div("Error", style={"color": COLORS["red"]}),
            create_kpi_card("Equity", "—"),
            create_kpi_card("Return", "—"),
            create_kpi_card("Drawdown", "—"),
            create_kpi_card("Sharpe", "—"),
            html.Div(f"Error: {e}", style={"fontSize": "11px", "color": COLORS["red"]}),
            empty, empty, [], [], [], [], [], None, empty,
        )


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)