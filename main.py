import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta
import plotly.figure_factory as ff


def calculate_rsi(prices, periods=14):
    delta = prices.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=periods).mean()
    avg_loss = loss.rolling(window=periods).mean()

    relative_strength = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + relative_strength))

    return rsi.iloc[-1]


def calculate_beta(stock_returns, market_returns):
    """Calculate beta by comparing stock returns to market returns"""
    common_index = stock_returns.index.intersection(market_returns.index)

    stock_returns_aligned = stock_returns.loc[common_index]
    market_returns_aligned = market_returns.loc[common_index]

    try:
        covariance = np.cov(stock_returns_aligned, market_returns_aligned)[0, 1]
        market_variance = np.var(market_returns_aligned)
        beta = covariance / market_variance if market_variance != 0 else 0

        return beta
    except Exception as e:
        st.warning(f"Beta calculation error: {e}")
        return 0


st.set_page_config(
    page_title="Stock Market Dashboard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

st.title("📈 Comprehensive Stock Market Dashboard")

st.sidebar.header("Dashboard Controls")

tickers = st.sidebar.text_input(
    "Enter Stock Tickers (comma-separated)", "AAPL, GOOGL, MSFT"
)
ticker_list = [ticker.strip() for ticker in tickers.split(",")]

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
with col2:
    end_date = st.date_input("End Date", datetime.now())


@st.cache_data
def fetch_stock_data(tickers, start, end):
    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end)
            if not df.empty:
                df["Ticker"] = ticker
                data[ticker] = df
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {e}")
    return data


stock_data = fetch_stock_data(ticker_list, start_date, end_date)

if stock_data:
    tab1, tab2, tab3 = st.tabs(
        ["Stock Prices", "Performance Comparison", "Detailed Analysis"]
    )

    with tab1:
        st.header("Stock Price Visualization")

        chart_type = st.selectbox(
            "Select Chart Type", ["Candlestick", "Line Chart", "Area Chart"]
        )

        for ticker, df in stock_data.items():
            st.subheader(f"{ticker} Stock Price")

            if chart_type == "Candlestick":
                fig = go.Figure(
                    data=[
                        go.Candlestick(
                            x=df.index,
                            open=df["Open"],
                            high=df["High"],
                            low=df["Low"],
                            close=df["Close"],
                        )
                    ]
                )
            elif chart_type == "Line Chart":
                fig = px.line(
                    df, x=df.index, y=["Open", "Close"], title=f"{ticker} Stock Prices"
                )
            else:
                fig = px.area(
                    df,
                    x=df.index,
                    y=["Open", "Close"],
                    title=f"{ticker} Stock Price Trends",
                )

            fig.update_layout(height=500, width=1000)
            st.plotly_chart(fig)

    with tab2:
        st.header("🚀 Stock Performance Comparison")

        performance_data = []
        for ticker, df in stock_data.items():
            total_return = ((df["Close"][-1] - df["Close"][0]) / df["Close"][0]) * 100
            volatility = df["Close"].pct_change().std() * np.sqrt(252)
            sharpe_ratio = total_return / volatility
            max_drawdown = (
                (df["Close"].cummax() - df["Close"]) / df["Close"].cummax() * 100
            )

            performance_data.append(
                {
                    "Ticker": ticker,
                    "Total Return (%)": round(total_return, 2),
                    "Volatility (%)": round(volatility, 2),
                    "Sharpe Ratio": round(sharpe_ratio, 2),
                    "Max Drawdown (%)": round(max_drawdown.max(), 2),
                    "Current Price": round(df["Close"][-1], 2),
                }
            )

        perf_df = pd.DataFrame(performance_data)

        col1, col2 = st.columns(2)

        with col1:
            categories = ["Total Return (%)", "Sharpe Ratio", "Max Drawdown (%)"]
            fig_radar = go.Figure()

            for index, row in perf_df.iterrows():
                fig_radar.add_trace(
                    go.Scatterpolar(
                        r=[
                            row["Total Return (%)"],
                            row["Sharpe Ratio"],
                            abs(row["Max Drawdown (%)"]),
                        ],
                        theta=categories,
                        fill="toself",
                        name=row["Ticker"],
                    )
                )

            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[-50, 50])),
                title="Multi-Dimensional Performance Radar",
            )
            st.plotly_chart(fig_radar)

        with col2:
            performance_matrix = perf_df.set_index("Ticker")[
                ["Total Return (%)", "Volatility (%)", "Sharpe Ratio"]
            ]
            fig_heatmap = px.imshow(
                performance_matrix,
                text_auto=True,
                aspect="auto",
                title="Performance Heatmap",
                color_continuous_scale="RdYlGn",
            )
            st.plotly_chart(fig_heatmap)

        st.dataframe(
            perf_df.style.background_gradient(
                cmap="RdYlGn", subset=["Total Return (%)", "Sharpe Ratio"]
            ).highlight_max(color="lightgreen", axis=0)
        )

    with tab3:
        st.header("🔍 Comprehensive Stock Analysis")

        col1, col2 = st.columns(2)

        try:
            market_data = yf.download("^GSPC", start=start_date, end=end_date)
            market_returns = market_data["Adj Close"].pct_change().dropna()
        except Exception as e:
            st.error(f"Could not fetch market benchmark data: {e}")
            market_returns = pd.Series()

        with col1:
            for ticker, df in stock_data.items():
                daily_returns = df["Close"].pct_change().dropna()
                hist_data = [daily_returns]
                group_labels = [ticker]

                fig_dist = ff.create_distplot(
                    hist_data, group_labels, show_hist=True, show_curve=True
                )
                fig_dist.update_layout(title=f"{ticker} Daily Returns Distribution")
                st.plotly_chart(fig_dist)

        with col2:
            for ticker, df in stock_data.items():
                rolling_vol = df["Close"].pct_change().rolling(
                    window=30
                ).std() * np.sqrt(252)

                fig_rolling = go.Figure()
                fig_rolling.add_trace(
                    go.Scatter(
                        x=rolling_vol.index,
                        y=rolling_vol.values,
                        mode="lines",
                        name=f"{ticker} 30-Day Rolling Volatility",
                    )
                )
                fig_rolling.update_layout(title=f"{ticker} Rolling Volatility")
                st.plotly_chart(fig_rolling)

        export_ticker = st.selectbox("Select Ticker to Export", ticker_list)

        if st.button("Export Stock Data to CSV"):
            export_df = stock_data[export_ticker]
            csv = export_df.to_csv(index=True)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"{export_ticker}_stock_data.csv",
                mime="text/csv",
            )

        st.subheader("🏆 Advanced Stock Metrics")
        for ticker, df in stock_data.items():
            st.write(f"**{ticker} Advanced Metrics**")

            stock_returns = df["Close"].pct_change().dropna()

            beta = (
                calculate_beta(stock_returns, market_returns)
                if not market_returns.empty
                else 0
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Current Price", f"${df['Close'][-1]:.2f}")
            with col2:
                st.metric("Beta (30-day)", f"{beta:.2f}")
            with col3:
                st.metric(
                    "Relative Strength Index", f"{calculate_rsi(df['Close']):.2f}"
                )
            with col4:
                daily_change = df["Close"][-1] - df["Close"][-2]
                daily_change_pct = (daily_change / df["Close"][-2]) * 100
                st.metric(
                    "Daily Change",
                    f"{daily_change:.2f} ({daily_change_pct:.2f}%)",
                    "green" if daily_change > 0 else "red",
                )

else:
    st.warning("Please enter valid stock ticker symbols")

st.sidebar.markdown("---")
st.sidebar.markdown("### Dashboard Instructions")
st.sidebar.info(
    """
1. Enter stock tickers separated by commas
2. Select date range
3. Explore different tabs and visualizations
4. Export data as needed
"""
)
