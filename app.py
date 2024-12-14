import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import os # Import the os module here
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

# Hàm chuẩn bị dữ liệu cho ARIMA
def prepare_arima_data(df):
    # Chỉ sử dụng cột 'Close' để dự đoán giá cổ phiếu
    return df['Close']

# Hàm xây dựng mô hình ARIMA và dự báo giá cổ phiếu
def predict_arima(df, forecast_days=7):
    # Chuẩn bị dữ liệu
    data = prepare_arima_data(df)

    # Tạo mô hình ARIMA với (p, d, q) = (5, 1, 0) (p, d, q có thể điều chỉnh theo dữ liệu thực tế)
    model = ARIMA(data, order=(5, 1, 0))

    # Huấn luyện mô hình
    model_fit = model.fit()

    # Dự báo giá cổ phiếu cho các ngày tiếp theo
    forecast = model_fit.forecast(steps=forecast_days)

    # Trả về kết quả dự báo
    return forecast

# Hàm duy nhất cho ARIMA: Dự báo và tính toán lỗi
def arima_forecast(df, forecast_days=7):
    # Chuẩn bị dữ liệu
    data = df['Adj Close']  # Sử dụng cột 'Adj Close' cho dự báo

    # Xây dựng mô hình ARIMA với tham số (p,d,q) mặc định là (5,1,0)
    model = ARIMA(data, order=(5, 1, 0))  
    model_fit = model.fit()

    # Dự báo cho các ngày tiếp theo
    forecast = model_fit.forecast(steps=forecast_days)
    
    # Dự báo trên dữ liệu huấn luyện (dự báo giá trị của các ngày đã qua)
    train_predict = model_fit.predict(start=0, end=len(data)-1)

    # Tính toán các tham số lỗi (MSE, MAE)
    mae = mean_absolute_error(data, train_predict)
    mse = mean_squared_error(data, train_predict)

    # Trả về các kết quả: dự báo, tham số lỗi, và mô hình đã huấn luyện
    return forecast, train_predict, mae, mse, model_fit

def safe_float(x):
    """Safely convert Pandas Series or single values to float"""
    try:
        if isinstance(x, pd.Series):  # Kiểm tra nếu đầu vào là Pandas Series
            return float(x.iloc[0])  # Lấy giá trị đầu tiên và chuyển sang float
        return float(x)  # Chuyển trực tiếp sang float nếu là giá trị đơn
    except (ValueError, TypeError) as e:
        print(f"Warning: Cannot convert {x} to float. Error: {e}")
        return None  # Trả về None nếu lỗi

def get_stock_data(symbol, start_date, end_date=datetime.now()):
    """
    Lấy dữ liệu lịch sử cổ phiếu từ Yahoo Finance.
    """
    try:
        df = yf.download(symbol, start=start_date, end=end_date)
        return df
    except Exception as e:
        st.error(f"Lỗi khi tải dữ liệu: {e}")
        return None

def clean_data_with_header(df, symbol):
    """
    Xử lý dữ liệu và thêm hàng tiêu đề chứa mã cổ phiếu.
    """
    # Bước 1: Loại bỏ hàng tiêu đề thừa
    df = df.iloc[1:].reset_index(drop=True)

    # Bước 2: Đặt tên cho cột đầu tiên là 'Date'
    df.rename(columns={df.columns[0]: "Date"}, inplace=True)

    # Bước 3: Chuyển cột 'Date' sang định dạng datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Bước 4: Xóa các hàng thiếu dữ liệu
    df.dropna(inplace=True)

    # Bước 5: Thêm hàng tiêu đề chứa mã cổ phiếu
    new_header = pd.DataFrame([[symbol] + [""] * (df.shape[1] - 1)], columns=df.columns)
    df = pd.concat([new_header, df], ignore_index=True)

    # Bước 6: Reset lại chỉ mục
    df.reset_index(drop=True, inplace=True)

    return df


def calculate_technical_indicators(df):
    """Calculate technical indicators"""
    # RSI Đánh giá tình trạng quá mua/quá bán.
    delta = df['Close'].diff() # Tính chênh lệch giữa giá đóng cửa ngày hiện tại và trước
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()# Lấy trung bình tăng
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()# Lấy trung bình giảm
    rs = gain / loss # Tính tỷ lệ tăng/giảm
    df['RSI'] = 100 - (100 / (1 + rs)) # Công thức RSI

    # MACD Xác định xu hướng giá.
    exp1 = df['Close'].ewm(span=12, adjust=False).mean() # EMA 12 ngày
    exp2 = df['Close'].ewm(span=26, adjust=False).mean() # EMA 26 ngày
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()# Đường tín hiệu

    # Bollinger Bands Xác định biến động giá.
    rolling_mean = df['Close'].rolling(window=20).mean()# Trung bình 20 ngày
    rolling_std = df['Close'].rolling(window=20).std() # Độ lệch chuẩn 20 ngày

    df['BB_middle'] = rolling_mean
    df['BB_upper'] = rolling_mean + (2 * rolling_std)  # Sửa lại
    df['BB_lower'] = rolling_mean - (2 * rolling_std)  # Sửa lại

    # Volume MA Trung bình khối lượng giao dịch
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()

    return df

def predict_prices(df, days):
    """Predict future prices using enhanced algorithm"""
    if df is None or len(df) < 20: # Kiểm tra dữ liệu đủ dài
        return None

    closes = df['Close'].values.flatten() # Lấy giá đóng cửa thành mảng
    ma20 = pd.Series(closes).rolling(20).mean() # Trung bình động 20 ngày
    std = safe_float(closes[-20:].std())  # Độ lệch chuẩn

    last_price = safe_float(closes[-1]) # Giá cuối cùng
    trend = safe_float(ma20.iloc[-1] - ma20.iloc[-20]) / 20 if len(ma20) >= 20 else 0 # Xu hướng

    # Enhanced prediction with technical indicators
    rsi = df['RSI'].iloc[-1] if 'RSI' in df else 50
    macd = df['MACD'].iloc[-1] if 'MACD' in df else 0

    # Adjust trend based on technical indicators
    if rsi > 70:
        trend *= 0.8  # Reduce upward trend if overbought
    elif rsi < 30:
        trend *= 1.2  # Increase upward trend if oversold

    if macd > 0:
        trend *= 1.1  # Increase trend if MACD is positive
    else:
        trend *= 0.9  # Decrease trend if MACD is negative

    predictions = []
    current_price = last_price

    for _ in range(days):
        # Add more sophisticated random variation (Độ biến động)
        volatility = std * 0.1
        technical_factor = (rsi - 50) / 500  # Small adjustment based on RSI
        random_change = np.random.normal(0, volatility)

        current_price += trend + random_change + technical_factor
        predictions.append(max(0, current_price))  # Ensure price doesn't go negative

    return predictions

def calculate_metrics(df, predictions, forecast_days):
    """Calculate enhanced metrics"""
    last_price = safe_float(df['Close'].iloc[-1])
    pred_price = safe_float(predictions[0])
    avg_price = float(sum(predictions) / len(predictions))
    change = ((pred_price - last_price) / last_price) * 100

    # Calculate additional metrics
    historical_volatility = safe_float(df['Close'].pct_change().std() * np.sqrt(252) * 100)
    max_prediction = max(predictions)
    min_prediction = min(predictions)
    pred_volatility = np.std(predictions) / np.mean(predictions) * 100

    # Add technical metrics
    rsi = safe_float(df['RSI'].iloc[-1]) if 'RSI' in df else None
    macd = safe_float(df['MACD'].iloc[-1]) if 'MACD' in df else None
    signal = safe_float(df['Signal'].iloc[-1]) if 'Signal' in df else None

    # Calculate trend strength Tính độ mạnh của xu hướng (Trend Strength) Nếu MA20 cao hơn MA50, xu hướng ngắn hạn tăng mạnh. Ngược lại, nếu MA20 thấp hơn MA50, xu hướng giảm.
    ma20 = df['Close'].rolling(window=20).mean()
    ma50 = df['Close'].rolling(window=50).mean()
    # Chuyển đổi sang float
    trend_strength = safe_float(((ma20.iloc[-1] / ma50.iloc[-1]) - 1) * 100)

    return {
        'last_price': last_price,
        'pred_price': pred_price,
        'avg_price': avg_price,
        'change': change,
        'historical_volatility': historical_volatility,
        'max_prediction': max_prediction,
        'min_prediction': min_prediction,
        'pred_volatility': pred_volatility,
        'rsi': rsi,
        'macd': macd,
        'signal': signal,
        'trend_strength': trend_strength
    }
##Tạo biểu đồ hiển thị dữ liệu cổ phiếu với các chỉ báo kỹ thuật.
def create_enhanced_chart(df, predictions, future_dates, symbol, settings):
    """Create enhanced interactive chart with better layout"""
    indicators = settings['indicators']

    # Tính toán số lượng subplots cần thiết
    n_rows = 1  # Main price chart
    if 'MACD' in indicators:
        n_rows += 1
    if 'RSI' in indicators:
        n_rows += 1

    # Tính toán chiều cao cho từng subplot
    row_heights = []
    if n_rows == 1:
        row_heights = [1]
    elif n_rows == 2:
        row_heights = [0.7, 0.3]
    else:
        row_heights = [0.6, 0.2, 0.2]

    # Tạo subplot với khoảng cách phù hợp
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=row_heights
    )

    # Thêm candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            showlegend=True
        ),
        row=1, col=1
    )

    # Thêm volume với trục y phụ
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            opacity=0.3,
            yaxis='y2'
        ),
        row=1, col=1
    )

    # Thêm đường dự đoán
    fig.add_trace(
        go.Scatter(
            x=future_dates,
            y=predictions,
            name='Prediction',
            line=dict(color='purple', width=2, dash='dash'),
            mode='lines'
        ),
        row=1, col=1
    )

    current_row = 2
# Thêm MA20 nếu được chọn
    if 'MA20' in settings['indicators']:
        ma20 = df['Close'].rolling(window=20).mean()
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=ma20,
                name='MA20',
                line=dict(
                    color='orange',
                    width=1.5
                ),
                hovertemplate=
                "Date: %{x}<br>" +
                "MA20: $%{y:.2f}<br>" +
                "<extra></extra>"
            ),
            secondary_y=False
        )

    # Thêm MA50 nếu được chọn
    if 'MA50' in settings['indicators']:
        ma50 = df['Close'].rolling(window=50).mean()
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=ma50,
                name='MA50',
                line=dict(
                    color='blue',
                    width=1.5
                ),
                hovertemplate=
                "Date: %{x}<br>" +
                "MA50: $%{y:.2f}<br>" +
                "<extra></extra>"
            ),
            secondary_y=False
        )
    # Thêm MACD nếu được chọn
    if 'MACD' in indicators:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACD'],
                name='MACD',
                line=dict(color='blue')
            ),
            row=current_row, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Signal'],
                name='Signal',
                line=dict(color='orange')
            ),
            row=current_row, col=1
        )

        # Thêm MACD histogram
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['MACD'] - df['Signal'],
                name='MACD Histogram',
                marker_color='gray',
                opacity=0.3
            ),
            row=current_row, col=1
        )

        current_row += 1

    # Thêm RSI nếu được chọn
    if 'RSI' in indicators:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['RSI'],
                name='RSI',
                line=dict(color='red')
            ),
            row=current_row, col=1
        )

        # Thêm đường tham chiếu RSI
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1)

    # Cập nhật layout
    fig.update_layout(
        title=f'Technical Analysis - {symbol}',
        template='plotly_white',
        hovermode='x unified',  # Hiển thị tooltip thống nhất theo trục x
        hoverdistance=100,      # Khoảng cách hiển thị tooltip
        spikedistance=1000,     # Khoảng cách hiển thị spike line
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            rangeslider=dict(visible=False),
            type="date",
            showgrid=True,
            showspikes=True,    # Hiển thị spike line khi hover
            spikethickness=1,
            spikecolor="gray",
            spikemode="across"
        ),
        yaxis=dict(
            title="Price ($)",
            side="left",
            showgrid=True,
            tickformat="$.2f"
        ),
        yaxis2=dict(
            title="Volume",
            side="right",
            overlaying="y",
            showgrid=False,
            tickformat=","
        ),
        margin=dict(l=50, r=50, t=50, b=50)
    )



    # Cập nhật trục y cho từng subplot
    for i in range(1, n_rows + 1):
        fig.update_yaxes(title_text="Price" if i == 1 else "", row=i, col=1)
        if i == 1:
            fig.update_yaxes(title_text="Volume", secondary_y=True, row=i, col=1)
        elif i == 2 and 'MACD' in indicators:
            fig.update_yaxes(title_text="MACD", row=i, col=1)
        elif i == n_rows and 'RSI' in indicators:
            fig.update_yaxes(title_text="RSI", row=i, col=1)

    return fig
def create_macd_chart(df, symbol):
    """Create separate MACD chart"""
    fig = go.Figure()

    # MACD Line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['MACD'],
        name='MACD',
        line=dict(color='blue')
    ))

    # Signal Line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Signal'],
        name='Signal',
        line=dict(color='orange')
    ))

    # MACD Histogram
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['MACD'] - df['Signal'],
        name='MACD Histogram',
        marker_color='gray'
    ))

    fig.update_layout(
        title=f'MACD - {symbol}',
        height=300,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )

    return fig

def create_rsi_chart(df, symbol):
    """Create separate RSI chart"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['RSI'],
        name='RSI',
        line=dict(color='purple')
    ))

    # Add overbought/oversold lines
    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.add_hline(y=30, line_dash="dash", line_color="green")

    fig.update_layout(
        title=f'RSI - {symbol}',
        height=300,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        yaxis=dict(range=[0, 100])
    )

    return fig
def display_enhanced_metrics(metrics):
    """Display enhanced metrics with tooltips"""
    st.subheader("Số liệu chi tiết")


    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container():
            st.metric("Giá hiện tại",
                     f"${metrics['last_price']:.2f}")

        with st.container():
            st.metric("Biến động lịch sử",
                     f"{metrics['historical_volatility']:.1f}%")

    with col2:
        with st.container():
            st.metric("Dự đoán ngày tiếp theo",
                     f"${metrics['pred_price']:.2f}",
                     f"{metrics['change']:+.2f}%")

        with st.container():
            if metrics['rsi'] is not None:
                st.metric("RSI",
                         f"{metrics['rsi']:.1f}")

    with col3:
        with st.container():
            if metrics['macd'] is not None:
                st.metric("MACD",
                         f"{metrics['macd']:.2f}")

        with st.container():
            st.metric("Trend Strength",
                     f"{metrics['trend_strength']:+.2f}%")

    # Add custom CSS for tooltips
    st.markdown("""
    <style>
    .tooltip {
        font-size: 0.8em;
        color: gray;
        margin-top: -15px;
        margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

def add_settings_main():
    """Add settings sidebar"""
    with st.expander ("Chọn phương pháp và khoảng tinh cậy"): 
        col1, col2 = st.columns(2) # Create two columns for better layout

    with col1:
        indicators = st.multiselect(
            "Phương pháp kỹ thuật",
            ["MACD", "RSI"],
            help="Chọn chỉ báo kỹ thuật để hiển thị"
        )
    with col2:
        st.subheader("Khoản tin cậy")

        prediction_confidence = st.slider(
            "Độ tin cậy của dự đoán",
            min_value=0.1,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Điều chỉnh khoảng tin cậy dự đoán"
        )

        return {
            "indicators": indicators,
            "prediction_confidence": prediction_confidence
        }

def display_prediction_table(future_dates, predictions, metrics):
    """Display prediction table with enhanced formatting"""
    st.subheader("Bảng dự đoán tương lai")

    df_pred = pd.DataFrame({
        'Date': future_dates,  # Already named "Date"
        'Predicted Price': [f"${p:.2f}" for p in predictions],
        'Change (%)': [
            f"{((p - metrics['last_price']) / metrics['last_price'] * 100):+.2f}%"
            for p in predictions
        ],
        'Confidence Interval': [
            f"${p-p*0.05:.2f} - ${p+p*0.05:.2f}"
            for p in predictions
        ]
    })

    # Add styling
    def highlight_changes(val):
        if '%' in str(val):
            num = float(val.strip('%').replace('+', ''))
            if num > 0:
                return 'color: green'
            elif num < 0:
                return 'color: red'
        return ''

    styled_df = df_pred.style.applymap(highlight_changes)
    st.dataframe(styled_df, height=400)

def calculate_statistics(df):
    """Tính toán các tham số thống kê cho DataFrame."""
    # Loại bỏ cột 'Date' khỏi thống kê
    df_numeric = df.select_dtypes(include=np.number).drop(columns=['Date'], errors='ignore')

    statistics = df_numeric.describe().to_dict()  # Tính toán các tham số cơ bản
    for col in df_numeric.columns:
        # Tính toán các tham số bổ sung
        statistics[col]['Mode'] = df_numeric[col].mode()[0]
        statistics[col]['Sample Variance'] = df_numeric[col].var()
        statistics[col]['Kurtosis'] = df_numeric[col].kurt()
        statistics[col]['Skewness'] = df_numeric[col].skew()
        statistics[col]['Range'] = df_numeric[col].max() - df_numeric[col].min()
        statistics[col]['Sum'] = df_numeric[col].sum()
        statistics[col]['Count'] = df_numeric[col].count()
        # Confidence Level (95.0%)
        confidence_interval = stats.t.interval( # Sử dụng stats thay vì st
            0.95, len(df_numeric[col]) - 1, loc=np.mean(df_numeric[col]), scale=stats.sem(df_numeric[col])
        )
        statistics[col]['Confidence Level(95.0%)'] = f"{confidence_interval[0]:.2f} - {confidence_interval[1]:.2f}"
    stats_df = pd.DataFrame(statistics) # Tạo DataFrame từ statistics

    return stats_df # Trả về DataFrame


def create_chart(df, start_date, end_date):
    """Tạo biểu đồ đường cho tổng của Adj Close, Close, Open và biểu đồ cột cho Volume (Volume là y second)."""

    # Chuyển đổi start_date và end_date thành kiểu datetime64[ns]
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Chuyển đổi cột 'Date' trong df thành kiểu datetime64[ns] nếu cần
    df['Date'] = df['Date'].dt.tz_localize(None)

    # Lọc DataFrame theo start_date và end_date
    mask = (df['Date'] >= start_date.to_numpy()) & (df['Date'] <= end_date.to_numpy())
    filtered_df = df.loc[mask]

    # Tính toán tổng hàng ngày cho mỗi biến
    filtered_df['Daily Adj Close Sum'] = filtered_df['Adj Close']
    filtered_df['Daily Close Sum'] = filtered_df['Close']
    filtered_df['Daily Open Sum'] = filtered_df['Open']
    filtered_df['Daily Volume Sum'] = filtered_df['Volume'] # Thêm cột tổng Volume theo ngày

    # Tạo biểu đồ đường với trục y phụ cho Volume
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Thêm các đường cho Adj Close, Close, Open
    fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['Daily Adj Close Sum'], mode='lines', name='Daily Adj Close Sum'), secondary_y=False)
    fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['Daily Close Sum'], mode='lines', name='Daily Close Sum'), secondary_y=False)
    fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['Daily Open Sum'], mode='lines', name='Daily Open Sum'), secondary_y=False)

    # Thêm cột cho Volume sum theo ngày (trục y phụ)
    fig.add_trace(go.Bar(x=filtered_df['Date'], y=filtered_df['Daily Volume Sum'], name='Volume'), secondary_y=True) # Sử dụng Daily Volume Sum

    # Cập nhật layout biểu đồ đường
    fig.update_layout(
        title_text="Biểu đồ đường tổng của Adj Close, Close, Open và Volume (Volume là y second, tính tổng theo ngày)",
        xaxis_title="Ngày",
        yaxis_title="Giá trị",
        yaxis2_title="Volume",
        xaxis_range=[start_date, end_date],
        height=600,
        yaxis2=dict(side='right') # Hiển thị trục y phụ ở bên phải
    )

    # Trả về chỉ fig và filtered_df
    return fig, filtered_df


def analyze_stock(symbol, start_date, end_date):
    st.write(f"Analyzing {symbol} from {start_date} to {end_date.strftime('%Y-%m-%d')}")
    df = get_stock_data(symbol, start_date, end_date)
    if df is not None and not df.empty:
        df = df.reset_index()  # Resets index, adding 'Date' column

        st.subheader("Dữ liệu nguồn")
        st.dataframe(df)

        # Hiển thị biểu đồ trong st.expander
        with st.expander("Mô tả dữ liệu"):

            # Tính toán và hiển thị các tham số thống kê
            st.subheader("Phân tích các tham số thống kê")
            statistics = calculate_statistics(df)  # Nhận DataFrame từ calculate_statistics
            st.dataframe(statistics)  # Hiển thị DataFrame

            # Tạo bản sao của df và đặt lại index
            df_with_date = df.reset_index()

            # Gọi create_chart và nhận fig
            fig, filtered_df = create_chart(df_with_date, start_date, end_date) # Sử dụng df_with_date
            st.plotly_chart(fig, use_container_width=True)



       # Tạo expander cho bảng tương quan và Pairplot
        with st.expander("Tương quan giữa các biến"):
            # Nhóm dữ liệu theo ngày và tính toán tổng cho mỗi cột
            daily_data = df.groupby('Date').sum()

            # Loại bỏ cột 'Date' khỏi daily_data vì nó đã trở thành index
            daily_data = daily_data.drop(columns=['Date'], errors='ignore')

            # Tính toán ma trận tương quan
            correlation_matrix = daily_data.corr()  # Sử dụng daily_data

            # Hiển thị bảng tương quan
            st.subheader("Bảng tương quan")
            st.dataframe(correlation_matrix)

            # Tạo và hiển thị Pairplot với seaborn và matplotlib
            st.subheader("Pairplot")

            # Chọn các cột cần thiết, sử dụng tên cột gốc từ DataFrame
            cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

            # Vẽ Pairplot với seaborn và matplotlib
            fig = sns.pairplot(daily_data[cols],
                               plot_kws={'color': 'green'},  # Màu cho biểu đồ phân tán
                               diag_kws={'color': 'green'})  # Màu cho histogram

            # Hiển thị biểu đồ trong Streamlit
            st.pyplot(fig)
    else:
        st.error(f"No data found for {symbol}")




def create_adj_close_ma_chart_with_prediction(df, ma_window=20, forecast_days=7, ma_period=None):
    """Tạo biểu đồ đường cho Adj Close, MA của Adj Close, và dự đoán giá trị tương lai với MA."""

    # Chuyển đổi cột 'Date' trong df thành kiểu datetime nếu cần
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    # ... (xử lý trường hợp không có cột 'Date' nếu cần)

    # Tính toán MA của Adj Close
    df['Adj Close MA'] = df['Adj Close'].rolling(window=ma_window).mean()

    # Dự đoán giá trị tương lai với MA
    last_adj_close_values = df['Adj Close'].tail(ma_window).values

    predictions = []
    for i in range(forecast_days):
        # Tính toán MA cho dự đoán bằng cách sử dụng ma_window giá trị cuối cùng và giá trị dự đoán trước đó
        # Nếu chưa có giá trị dự đoán trước đó (i == 0), sử dụng MA cuối cùng từ dữ liệu lịch sử
        if i == 0:
            prediction = np.mean(last_adj_close_values)
        else:
            prediction = np.mean(np.concatenate([last_adj_close_values[i:], predictions]))

        predictions.append(prediction)

    # Đảm bảo df.index[-1] là kiểu dữ liệu datetime
    last_date = pd.to_datetime(df.index[-1])  # Chuyển đổi thành datetime


    # Tạo DataFrame cho dự đoán
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
    df_pred = pd.DataFrame({'Adj Close MA Prediction': predictions}, index=future_dates)

    # Nối DataFrame dự đoán vào DataFrame gốc
    df_full = pd.concat([df, df_pred])

    # Tạo biểu đồ đường
    fig = go.Figure()


     # Adj Close line (Green)
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Adj Close'],
        mode='lines',
        name='Adj Close',
        line=dict(color='green')  # Set color to green
    ))

    # MA line (Red)
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Adj Close MA'],
        mode='lines',
        name=f'MA {ma_window}',
        line=dict(color='orange')  # Set color to red
    ))

    # Prediction line (Blue)
    fig.add_trace(go.Scatter(
        x=df_pred.index,
        y=df_pred['Adj Close MA Prediction'],
        mode='lines',
        name='Prediction',
        line=dict(color='blue', dash='dash')
    ))

    # Cập nhật layout biểu đồ - SỬA ĐOẠN NÀY
    fig.update_layout(
        title_text="Biểu đồ Adj Close và MA",
        xaxis_title="Ngày",
        yaxis_title="Giá trị",
        xaxis_range=[df.index.min(), df.index.max()],  # Sử dụng df.index để lấy phạm vi ngày tháng
        height=600,
    )

    # Dữ liệu thực tế trong khoảng thời gian dự đoán
    actual_values = df['Adj Close'].tail(forecast_days).values

    # Dự đoán trong khoảng thời gian đó
    predicted_values = predictions  # predictions đã được tính toán trong hàm

    mae = mean_absolute_error(actual_values, predicted_values)
    rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
    mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100

    # Trả về chỉ số lỗi cùng với biểu đồ và DataFrame dự đoán
    return fig, df_pred, mae, rmse, mape


def create_adj_close_es_chart_with_prediction(df, smoothing_level=0.1, forecast_days=7):
    """
    Tạo biểu đồ đường cho Adj Close và dự đoán giá trị tương lai với Exponential Smoothing.
    """

    # Chuyển đổi cột 'Date' trong df thành kiểu datetime nếu cần
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    # ... (xử lý trường hợp không có cột 'Date' nếu cần)

    # Kiểm tra xem df có index là DatetimeIndex không
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Tạo và huấn luyện model_hist
    model_hist = SimpleExpSmoothing(df['Adj Close']).fit(smoothing_level=smoothing_level)

    # Tính toán đường ES cho dữ liệu lịch sử
    df['Adj Close ES'] = df['Adj Close'].ewm(alpha=smoothing_level, adjust=False).mean()

    # Dự đoán giá trị tương lai với ES (sửa lỗi ở đây)
    # Lấy giá trị cuối cùng của 'Adj Close ES' làm giá trị ban đầu cho dự đoán
    last_es_value = df['Adj Close ES'].iloc[-1]  # Lấy giá trị ES cuối cùng
    predictions = []
    for i in range(forecast_days):
        # Dự đoán bằng cách sử dụng giá trị ES cuối cùng (hoặc dự đoán trước đó)
        # và áp dụng smoothing_level
        if i == 0:
            prediction = last_es_value  # Giá trị ban đầu cho dự đoán là giá trị ES cuối cùng
        else:
            prediction = predictions[-1]  # Giá trị dự đoán tiếp theo bằng giá trị dự đoán trước đó
                                        # (vì Simple ES giả định không có xu hướng hoặc tính thời vụ)

        predictions.append(prediction)

    # Tạo DataFrame cho dự đoán
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
    df_pred = pd.DataFrame({'Adj Close ES Prediction': predictions}, index=future_dates)


    # Nối DataFrame dự đoán vào DataFrame gốc
    df_full = pd.concat([df, df_pred])

    # Tạo biểu đồ đường
    fig = go.Figure()

    # Adj Close line (Green)
    fig.add_trace(go.Scatter(x=df_full.index, y=df_full['Adj Close'], mode='lines', name='Adj Close', line=dict(color='green')))

    # ES line for historical data (Orange)
    fig.add_trace(go.Scatter(x=df.index, y=df['Adj Close ES'], mode='lines', name='ES (Historical)', line=dict(color='orange')))

    # ES Prediction line (Blue)
    fig.add_trace(go.Scatter(x=df_pred.index, y=df_pred['Adj Close ES Prediction'], mode='lines', name='ES Prediction', line=dict(color='blue', dash='dash')))

    # Cập nhật layout biểu đồ - SỬA ĐOẠN NÀY
    fig.update_layout(
        title_text="Biểu đồ Adj Close và ES",
        xaxis_title="Ngày",
        yaxis_title="Giá trị",
        xaxis_range=[df.index.min(), df.index.max()],  # Sử dụng df.index để lấy phạm vi ngày tháng
        height=600,
    )

    # Dữ liệu thực tế trong khoảng thời gian dự đoán
    actual_values = df['Adj Close'].tail(forecast_days).values
    
    # Dự đoán trong khoảng thời gian đó
    predicted_values = predictions  # predictions đã được tính toán trong hàm

    mae = mean_absolute_error(actual_values, predicted_values)
    rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
    mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100
    
    # Trả về chỉ số lỗi cùng với biểu đồ và DataFrame dự đoán
    return fig, df_pred, mae, rmse, mape
    

def create_adj_close_holt_chart_with_prediction(df, smoothing_level, beta, forecast_days):
    """
    Creates a line chart for Adj Close and predicts future values using the Holt method.

    Args:
        df (pd.DataFrame): The input DataFrame containing stock data with 'Adj Close' column.
        alpha (float, optional): The smoothing parameter for the level (alpha). Defaults to 0.1.
        beta (float, optional): The smoothing parameter for the trend (beta). Defaults to 0.2.
        forecast_days (int, optional): The number of days to forecast. Defaults to 7.

    Returns:
        plotly.graph_objects.Figure: The generated Plotly chart.
        pd.DataFrame: The prediction values in a DataFrame.
    """

    # Chuyển đổi cột 'Date' thành kiểu datetime nếu cần
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

    # Đảm bảo index là DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Tạo và huấn luyện mô hình Holt
    model_holt = Holt(df['Adj Close'], initialization_method="estimated").fit(
        smoothing_level=smoothing_level, smoothing_trend=beta  # Sử dụng smoothing_level và beta
    )

    # Gán giá trị dự đoán vào cột 'Adj Close Holt'
    df['Adj Close Holt'] = model_holt.fittedvalues

    # Tính toán dự đoán trong mẫu để tính toán lỗi
    du_doan_trong_mau = model_holt.fittedvalues

    # Tính toán các chỉ số lỗi
    mae = mean_absolute_error(df['Adj Close'], du_doan_trong_mau)
    rmse = np.sqrt(mean_squared_error(df['Adj Close'], du_doan_trong_mau))
    mape = np.mean(np.abs((df['Adj Close'] - du_doan_trong_mau) / df['Adj Close'])) * 100

    # Khởi tạo giá trị mức (level) và xu hướng (trend) cuối cùng từ mô hình
    level = model_holt.level[-1]
    trend = model_holt.trend[-1]

    # Dự đoán giá trị tương lai bằng công thức Holt
    predictions = []
    for i in range(forecast_days):
        # Giá trị dự đoán = mức hiện tại + (xu hướng hiện tại * (i + 1))
        prediction = level + (trend * (i + 1))
        predictions.append(prediction)

    # Tạo DataFrame cho dự đoán
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
    df_pred = pd.DataFrame({'Adj Close Holt Prediction': predictions}, index=future_dates)


    # Create line chart
    fig = go.Figure()

    # Adj Close line (Green)
    fig.add_trace(go.Scatter(x=df.index, y=df['Adj Close'], mode='lines', name='Adj Close', line=dict(color='green')))

    # Holt line for historical data (Orange)
    fig.add_trace(go.Scatter(x=df.index, y=df['Adj Close Holt'], mode='lines', name='Holt (Historical)', line=dict(color='orange')))

    # Holt Prediction line (Blue)
    fig.add_trace(go.Scatter(x=df_pred.index, y=df_pred['Adj Close Holt Prediction'], mode='lines', name='Holt Prediction', line=dict(color='blue', dash='dash')))

    # Update layout
    fig.update_layout(
        title_text="Biểu đồ Adj Close và Holt",
        xaxis_title="Ngày",
        yaxis_title="Giá trị",
        xaxis_range=[df.index.min(), df.index.max()],
        height=600,
    )


    # Hiển thị tham số và chỉ số lỗi
    st.write(f"Alpha: {smoothing_level:.2f}, Beta: {beta:.2f}")
    st.write(f"**Chỉ số lỗi (Holt):**")
    st.write(f"  - MAE: {mae:.2f}")
    st.write(f"  - RMSE: {rmse:.2f}")
    st.write(f"  - MAPE: {mape:.2f}%")
    
    return fig, df_pred

def apply_es_monthly(df, smoothing_level, forecast_days):
    """Applies Exponential Smoothing method with monthly aggregation and returns predictions."""

    # Aggregate 'Adj Close' by month
    df['Month'] = df.index.to_period('M')
    monthly_df = df.groupby('Month')['Adj Close'].sum().reset_index()
    monthly_df['Month'] = monthly_df['Month'].dt.to_timestamp()
    monthly_df.set_index('Month', inplace=True)
    monthly_df = monthly_df.dropna(subset=['Adj Close'])
    # Train Exponential Smoothing model on monthly data
    model_es = SimpleExpSmoothing(monthly_df['Adj Close'], initialization_method="estimated").fit(
        smoothing_level=smoothing_level
    )

    # Add historical predictions to DataFrame
    monthly_df['Adj Close ES'] = model_es.fittedvalues

    # Get the last level value
    level = model_es.level[-1]

    # Generate predictions for the next forecast_days
    last_es_value = monthly_df['Adj Close ES'].iloc[-1]  # Lấy giá trị ES cuối cùng

    predictions = []
    for i in range(forecast_days):
        # Dự đoán bằng cách sử dụng giá trị ES cuối cùng (hoặc dự đoán trước đó)
        # và áp dụng smoothing_level
        if i == 0:
            prediction = last_es_value  # Giá trị ban đầu cho dự đoán là giá trị ES cuối cùng
        else:
            prediction = predictions[-1]  # Giá trị dự đoán tiếp theo bằng giá trị dự đoán trước đó
                                        # (vì Simple ES giả định không có xu hướng hoặc tính thời vụ)

        predictions.append(prediction)

    # Create DataFrame for predictions
    future_dates = pd.date_range(start=monthly_df.index[-1] + pd.DateOffset(months=1), periods=forecast_days, freq='M')
    df_pred = pd.DataFrame({'Adj Close ES Prediction': predictions}, index=future_dates)

    # Dự đoán trong khoảng thời gian đó
    predicted_values = predictions  # predictions đã được tính toán trong hàm
    # Get the actual values for the forecast horizon
    actual_values = monthly_df['Adj Close'][-forecast_days:].values

    # Calculate errors using actual and predicted values
    mae = mean_absolute_error(actual_values, predicted_values)  # Replace predicted_values with predictions
    rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))  # Replace predicted_values with predictions
    mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100  # Replace predicted_values with predictions


    # Generate predictions for the next forecast_days
    predictions = [level] * forecast_days  # ES predictions are constant

    # Create DataFrame for predictions
    future_dates = pd.date_range(start=monthly_df.index[-1] + pd.DateOffset(months=1), periods=forecast_days, freq='M')
    df_pred = pd.DataFrame({'Adj Close ES Prediction': predictions}, index=future_dates)

    # Create the plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly_df.index, y=monthly_df['Adj Close'], mode='lines', name='Adj Close (Historical)', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=monthly_df.index, y=monthly_df['Adj Close ES'], mode='lines', name='ES (Historical)', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df_pred.index, y=df_pred['Adj Close ES Prediction'], mode='lines', name='ES Prediction', line=dict(color='blue', dash='dash')))
    

    # Update layout
    fig.update_layout(
        title_text="Adj Close and Exponential Smoothing Chart (Monthly Aggregation)",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_range=[monthly_df.index.min(), df_pred.index.max()],  # Extend x-axis range to include predictions
        height=600,
    )

    return fig, df_pred, mae, rmse, mape


def create_adj_close_holt_winters_chart_with_prediction(df, smoothing_level, smoothing_trend, smoothing_seasonal, seasonality_periods, forecast_days):
    """
    Creates a line chart for Adj Close and predicts future values using the Holt-Winters method.

    Args:
        df (pd.DataFrame): The input DataFrame containing stock data with 'Adj Close' column.
        smoothing_level (float): The smoothing parameter for the level (alpha).
        smoothing_trend (float): The smoothing parameter for the trend (beta).
        smoothing_seasonal (float): The smoothing parameter for the seasonality (gamma).
        seasonality_periods (int): The number of periods in a season (e.g., 12 for monthly data with yearly seasonality).
        forecast_days (int): The number of days to forecast.

    Returns:
        plotly.graph_objects.Figure: The generated Plotly chart.
        pd.DataFrame: The prediction values in a DataFrame.
    """
    # Ensure 'Date' column is datetime and set as index
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Train the Holt-Winters model
    model_hw = ExponentialSmoothing(
        df['Adj Close'], 
        trend="add", 
        seasonal="add", 
        seasonal_periods=seasonality_periods,
        initialization_method="estimated"
    ).fit(
        smoothing_level=smoothing_level, 
        smoothing_trend=smoothing_trend, 
        smoothing_seasonal=smoothing_seasonal
    )

    # Add historical predictions to the DataFrame
    df['Adj Close Holt-Winters'] = model_hw.fittedvalues

    # Calculate errors
    mae = mean_absolute_error(df['Adj Close'], df['Adj Close Holt-Winters'])
    rmse = np.sqrt(mean_squared_error(df['Adj Close'], df['Adj Close Holt-Winters']))
    mape = np.mean(np.abs((df['Adj Close'] - df['Adj Close Holt-Winters']) / df['Adj Close'])) * 100

    # Prepare for future predictions
    predictions = []  # Use a list to store predictions

    # Get last values of level and trend
    level = model_hw.level[-1]
    trend = model_hw.trend[-1]

    # Holt-Winters seasonal values are not directly accessible, so we must calculate them
    seasonal_values = model_hw.fittedvalues - (level + trend)

    # Generate predictions for the next forecast_days
    for i in range(forecast_days):
        seasonal_index = (i + len(df)) % seasonality_periods  # Wrap around seasonality
        prediction = level + trend * (i + 1) + seasonal_values[seasonal_index]
        predictions.append(prediction)

    # Create a DataFrame for predictions
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
    df_pred = pd.DataFrame({'Adj Close Holt-Winters Prediction': predictions}, index=future_dates)

    # Create the plot
    fig = go.Figure()

    # Adj Close line (Green)
    fig.add_trace(go.Scatter(x=df.index, y=df['Adj Close'], mode='lines', name='Adj Close', line=dict(color='green')))

    # Holt-Winters line for historical data (Orange)
    fig.add_trace(go.Scatter(x=df.index, y=df['Adj Close Holt-Winters'], mode='lines', name='Holt-Winters (Historical)', line=dict(color='orange')))

    # Holt-Winters Prediction line (Blue)
    fig.add_trace(go.Scatter(x=df_pred.index, y=df_pred['Adj Close Holt-Winters Prediction'], mode='lines', name='Holt-Winters Prediction', line=dict(color='blue', dash='dash')))

    # Update layout
    fig.update_layout(
        title_text="Adj Close and Holt-Winters Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_range=[df.index.min(), df.index.max()],
        height=600,
    )

    # Display parameters and error metrics
    st.write(f"Alpha: {smoothing_level:.2f}, Beta: {smoothing_trend:.2f}, Gamma: {smoothing_seasonal:.2f}, Seasonality Periods: {seasonality_periods}")
    st.write(f"**Chỉ số lỗi (Holt Winter):**")
    st.write(f"  - MAE: {mae:.2f}")
    st.write(f"  - RMSE: {rmse:.2f}")
    st.write(f"  - MAPE: {mape:.2f}%")

    return fig, df_pred

def apply_holt_monthly(df, smoothing_level, smoothing_trend, forecast_days):
    """Applies Holt method with monthly aggregation and returns predictions."""

    # Ensure 'Date' column is datetime and set as index
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Aggregate 'Adj Close' by month
    df['Month'] = df.index.to_period('M')
    monthly_df = df.groupby('Month')['Adj Close'].sum().reset_index()
    monthly_df['Month'] = monthly_df['Month'].dt.to_timestamp()
    monthly_df.set_index('Month', inplace=True)

    # Train Holt model on monthly data
    model_holt = Holt(monthly_df['Adj Close'], initialization_method="estimated").fit(
        smoothing_level=smoothing_level, smoothing_trend=smoothing_trend
    )

    # Add historical predictions to DataFrame
    monthly_df['Adj Close Holt'] = model_holt.fittedvalues

    # Get the last level and trend values
    level = model_holt.level[-1]
    trend = model_holt.trend[-1]

    # Generate predictions for the next forecast_days
    predictions = []
    for i in range(forecast_days):
        prediction = level + trend * (i + 1)  # Holt prediction formula
        predictions.append(prediction)

    # Create DataFrame for predictions
    future_dates = pd.date_range(start=monthly_df.index[-1] + pd.DateOffset(months=1), periods=forecast_days, freq='M')
    df_pred = pd.DataFrame({'Adj Close Holt Prediction': predictions}, index=future_dates)

    # Calculate in-sample errors
    mae = mean_absolute_error(monthly_df['Adj Close'], model_holt.fittedvalues)
    rmse = np.sqrt(mean_squared_error(monthly_df['Adj Close'], model_holt.fittedvalues))
    mape = np.mean(np.abs((monthly_df['Adj Close'] - model_holt.fittedvalues) / monthly_df['Adj Close'])) * 100

    # Create the plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly_df.index, y=monthly_df['Adj Close'], mode='lines', name='Adj Close (Historical)', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=monthly_df.index, y=monthly_df['Adj Close Holt'], mode='lines', name='Holt (Historical)', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df_pred.index, y=df_pred['Adj Close Holt Prediction'], mode='lines', name='Holt Prediction', line=dict(color='blue', dash='dash')))
    # Update layout
    fig.update_layout(
        title_text="Adj Close and Holt-Winters Chart (Monthly Aggregation)",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_range=[monthly_df.index.min(), df_pred.index.max()],  # Extend x-axis range to include predictions
        height=600,
    )

    # Display parameters and error metrics
    st.write(f"Alpha: {smoothing_level:.2f}, Beta: {smoothing_trend:.2f}%")
    st.write(f"**Chỉ số lỗi (Holt):**")
    st.write(f"  - MAE: {mae:.2f}")
    st.write(f"  - RMSE: {rmse:.2f}")
    st.write(f"  - MAPE: {mape:.2f}%")

    return fig, df_pred


def apply_holt_winters_monthly(df, smoothing_level, smoothing_trend, smoothing_seasonal, forecast_days):
    """Applies Holt-Winters method with monthly aggregation and returns predictions."""

    # Ensure 'Date' column is datetime and set as index
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Aggregate 'Adj Close' by month
    df['Month'] = df.index.to_period('M')
    monthly_df = df.groupby('Month')['Adj Close'].sum().reset_index()
    monthly_df['Month'] = monthly_df['Month'].dt.to_timestamp()
    monthly_df.set_index('Month', inplace=True)

    # Train Holt-Winters model on monthly data
    seasonality_periods = 12  # Set to 12 for yearly seasonality with monthly data
    model_hw = ExponentialSmoothing(
        monthly_df['Adj Close'],
        trend="add",
        seasonal="add",
        seasonal_periods=seasonality_periods,
        initialization_method="estimated"
    ).fit(
        smoothing_level=smoothing_level,
        smoothing_trend=smoothing_trend,
        smoothing_seasonal=smoothing_seasonal
    )

    # Generate future dates for predictions
    future_dates = pd.date_range(start=monthly_df.index[-1] + pd.DateOffset(months=1), periods=forecast_days, freq='MS')  # 'MS' for month start frequency

    # Make predictions
    predictions = model_hw.forecast(forecast_days)

    # Get last values of level and trend
    level = model_hw.level[-1]
    trend = model_hw.trend[-1]

    # Holt-Winters seasonal values are not directly accessible, so we must calculate them
    seasonal_values = model_hw.fittedvalues - (level + trend)

    # Initialize predictions as a list (This is correct)
    predictions = []  

    # Add this line to create the 'Adj Close Holt-Winters' column
    monthly_df['Adj Close Holt-Winters'] = model_hw.fittedvalues  


    # Generate predictions for the next forecast_days
    for i in range(forecast_days):
        seasonal_index = (i + len(df)) % seasonality_periods  # Wrap around seasonality
        prediction = level + trend * (i + 1) + seasonal_values[seasonal_index]
        predictions.append(prediction)

    # Create DataFrame for predictions
    df_pred = pd.DataFrame({'Adj Close Holt-Winters Prediction': predictions}, index=future_dates)
    
    # Calculate in-sample errors
    mae = mean_absolute_error(monthly_df['Adj Close'], model_hw.fittedvalues)
    rmse = np.sqrt(mean_squared_error(monthly_df['Adj Close'], model_hw.fittedvalues))
    mape = np.mean(np.abs((monthly_df['Adj Close'] - model_hw.fittedvalues) / monthly_df['Adj Close'])) * 100
    
    # Create the plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=monthly_df.index, y=monthly_df['Adj Close'], mode='lines', name='Adj Close (Historical)', line=dict(color='green')))

    # Add Holt-Winters historical data line (Orange)
    # Assuming you have the fitted values in a column named 'Adj Close Holt-Winters' in your monthly_df
    fig.add_trace(go.Scatter(x=monthly_df.index, y=monthly_df['Adj Close Holt-Winters'], mode='lines', name='Holt-Winters (Historical)', line=dict(color='orange')))  
    
    # Add this trace for future predictions (Blue dashed line)
    fig.add_trace(go.Scatter(x=df_pred.index, y=df_pred['Adj Close Holt-Winters Prediction'], mode='lines', name='Holt-Winters Prediction', line=dict(color='blue', dash='dash'))) 
    

    # Update layout
    fig.update_layout(
        title_text="Adj Close and Holt-Winters Chart (Monthly Aggregation)",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_range=[monthly_df.index.min(), df_pred.index.max()],  # Extend x-axis range to include predictions
        height=600,
    )

    # Display parameters and error metrics
    st.write(f"Alpha: {smoothing_level:.2f}, Beta: {smoothing_trend:.2f}, Gamma: {smoothing_seasonal:.2f}")
    st.write(f"**Chỉ số lỗi (Holt Winter):**")
    st.write(f"  - MAE: {mae:.2f}")
    st.write(f"  - RMSE: {rmse:.2f}")
    st.write(f"  - MAPE: {mape:.2f}%")


    return fig, df_pred # R



def main():
    st.set_page_config(
        page_title="Stock Prediction",
        page_icon="📈",
        layout="wide"
    )

    ma_period = None  # Khởi tạo ma_period bằng None
    # Định nghĩa forecast_days ở đây
    forecast_days = 7  # Hoặc bất kỳ giá trị nào bạn muốn

    # Custom CSS for designing the sticky tab
    st.markdown("""
        <style>
        .main {
            padding-top: 1rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            position: fixed;
            top: 0;
            right: 0;
            background: #f0f0f0;
            z-index: 100;
            border-bottom: 1px solid #ddd;
        }
        .stTab {
            padding: 1rem;
        }
        .input-section {
            margin-bottom: 2rem;
        }
        .analyze-button {
            margin-top: 1rem;
            padding: 0.5rem 1rem;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        .analyze-button:hover {
            background-color: #45a049;
        }
        </style>
    """, unsafe_allow_html=True)

    # Create two main tabs in the sidebar
    with st.sidebar:
        st.title("📈Ứng Dụng Phân Tích Thống Kê trong Dự Báo Giá Chứng Khoán - Nhóm 11")  # Sidebar title
        selected_tab = st.radio("Khám phá", ["Trang chủ", "Dự đoán", "Dự đoán nâng cao"])  # Thêm tab "Prediction"


    # Home Tab
    if selected_tab == "Trang chủ":
        # Dùng HTML để căn giữa tiêu đề chính
        st.markdown(
        """
        <h1 style="text-align:center;">📊 Phân tích tổng quan</h1>
        """,
        unsafe_allow_html=True
        )

        st.header("🏠 Trang chủ")
        st.subheader('Thông số đầu vào')

        col1, col2, col3 = st.columns(3)
        with col1:
            symbol = st.text_input('Mã chứng khoán', 'NKE')
        with col2:
            start_date = st.date_input('Ngày bắt đầu', datetime.now() - timedelta(days=1826))
        with col3:
            end_date = st.date_input('Ngày kết thúc', datetime.now())

            # Chuyển đổi start_date và end_date thành chuỗi trước khi lưu trữ
            st.session_state.start_date = start_date.strftime('%Y-%m-%d')
            st.session_state.end_date = end_date.strftime('%Y-%m-%d')

            # Lưu trữ symbol, start_date, end_date vào session_state
            st.session_state.symbol = symbol
            st.session_state.start_date = start_date
            st.session_state.end_date = end_date

            # Khởi tạo start_date và end_date nếu chưa tồn tại
        if 'start_date' not in st.session_state:
            st.session_state.start_date = datetime.now() - timedelta(days=365)
        if 'end_date' not in st.session_state:
            st.session_state.end_date = datetime.now()

           # Thêm xử lý sự kiện cho nút "Analyze"
        if st.button('Phân tích'):
            analyze_stock(symbol, start_date, end_date)

    # Prediction Tab
    elif selected_tab == "Dự đoán":
        # Dùng HTML để căn giữa tiêu đề chính
        st.markdown(
        """
        <h1 style="text-align:center;">📊 Phân tích dự đoán</h1>
        """,
        unsafe_allow_html=True
        )
        st.header("🔮 Dự đoán")

        # Nhập thông tin start_date và end_date
        st.subheader("Nhập thông tin dự báo:")

        # Danh sách tên các file CSV đã tải sẵn
        csv_files = ["VFC.csv", "TSLA.csv", "NOK.csv", "NKE.csv", "ADDYY.csv"]

        # Lấy danh sách các file CSV trong thư mục dataset
        dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
        csv_files = [f for f in os.listdir(dataset_dir) if f.endswith('.csv')]

        # Tạo lựa chọn file CSV
        selected_file = st.selectbox("Chọn mã chứng khoán", csv_files)

        # Kiểm tra xem người dùng đã chọn file hay chưa
        if selected_file:
            # Đường dẫn tới file CSV đã chọn
            file_path = os.path.join(dataset_dir, selected_file)

            # Đọc dữ liệu từ file CSV đã chọn
            df = pd.read_csv(file_path)

        # Chọn mô hình dự báo
        st.subheader("Chọn mô hình dự báo:")
        model_choice = st.selectbox("Mô hình:",
                                    ["Simple Moving Average",
                                    "Exponential Smoothing By Day",
                                    "Exponential Smoothing By Month",
                                    "Holt By Day", "Holt By Month",
                                    "Holt Winter By Day", "Holt Winter By Month", 
                                    "ARIMA"
                                    ])

        # Chọn thời gian dự đoán (chỉ cho Simple Moving Average)
        if model_choice == "Simple Moving Average":
            st.subheader("Chọn thời gian dự đoán:")

            # Thêm ô nhập ma_window
            ma_window = st.number_input("Nhập kỳ hạn MA:", min_value=1, value=20)

            forecast_period = st.selectbox("Thời gian:",
                                                ["1 ngày", "1 tuần (5 ngày)",
                                                "1 tháng (22 ngày)", "Khác"])

            # Nếu chọn "Khác", cho phép nhập số ngày dự đoán
            if forecast_period == "Khác":
                custom_days = st.number_input("Nhập số ngày dự đoán:", min_value=1, value=1)
                forecast_days = custom_days  # Gán custom_days cho forecast_days nếu chọn "Khác"
            else:
                # Gán forecast_days dựa trên forecast_period đã chọn
                forecast_days = {
                    "1 ngày": 1,
                    "1 tuần (5 ngày)": 5,
                    "1 tháng (22 ngày)": 22,
                }[forecast_period]


        elif model_choice == "Exponential Smoothing By Day":
            st.subheader("Chọn thời gian dự đoán:")
            forecast_period = st.selectbox("Thời gian:",
                                                ["1 ngày", "1 tuần (5 ngày)",
                                                "1 tháng (22 ngày)", "Khác"])
            
             # Trong phần Pred, thêm thanh điều chỉnh:
            smoothing_level = st.slider("Alpha (Smoothing Level)", 0.01, 1.0, 0.1, 0.01)

            # Nếu chọn "Khác", cho phép nhập số ngày dự đoán
            if forecast_period == "Khác":
                custom_days = st.number_input("Nhập số ngày dự đoán:", min_value=1, value=1)
                ma_period = custom_days  # Gán custom_days cho ma_period nếu chọn "Khác"
            else:
                # Gán ma_period dựa trên forecast_period đã chọn
                ma_period = {
                    "1 ngày": 1,
                    "1 tuần (5 ngày)": 5,
                    "1 tháng (22 ngày)": 22,
                }[forecast_period]

        elif model_choice == "Exponential Smoothing By Month":
            st.subheader("Chọn thời gian dự đoán:")

            seasonality_periods = st.number_input("Giai đoạn mùa vụ", min_value=1, value=12, step=1)

            forecast_period = st.selectbox("Thời gian:",
                                                ["1 tháng", "6 tháng",
                                                "12 tháng", "Khác"])
            
             # Trong phần Pred, thêm thanh điều chỉnh:
            alpha_es = st.slider("Alpha (Smoothing Level)", 0.01, 1.0, 0.1, 0.01)

            # Nếu chọn "Khác", cho phép nhập số ngày dự đoán
            if forecast_period == "Khác":
                custom_days = st.number_input("Nhập số ngày dự đoán:", min_value=1, value=1)
                ma_period = custom_days  # Gán custom_days cho ma_period nếu chọn "Khác"
            else:
                # Gán ma_period dựa trên forecast_period đã chọn
                ma_period = {
                    "1 tháng": 1,
                    "6 tháng": 6,
                    "12 tháng": 12,
                }[forecast_period]


        elif model_choice == "Holt By Day":
            st.subheader("Chọn thời gian dự đoán:")
            forecast_period = st.selectbox("Thời gian:",
                                                ["1 ngày", "1 tuần (5 ngày)",
                                                "1 tháng (22 ngày)", "Khác"])

            st.subheader("Chọn hệ số alpha và beta:")
            alpha = st.slider("Alpha (Smoothing Level):", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
            beta = st.slider("Beta (Smoothing Trend):", min_value=0.01, max_value=1.0, value=0.2, step=0.01)


            # Nếu chọn "Khác", cho phép nhập số ngày dự đoán
            if forecast_period == "Khác":
                custom_days = st.number_input("Nhập số ngày dự đoán:", min_value=1, value=1)
                ma_period = custom_days  # Gán custom_days cho ma_period nếu chọn "Khác"
            else:
                # Gán ma_period dựa trên forecast_period đã chọn
                ma_period = {
                    "1 ngày": 1,
                    "1 tuần (5 ngày)": 5,
                    "1 tháng (22 ngày)": 22,
                }[forecast_period]

        elif model_choice == "Holt By Month":
            st.subheader("Chọn thời gian dự đoán:")

            seasonality_periods = st.number_input("Giai đoạn mùa vụ", min_value=1, value=12, step=1)

            forecast_period = st.selectbox("Thời gian:",
                                                ["1 tháng", "6 tháng",
                                                "12 tháng", "Khác"])

            # Add sliders for Holt-Winters parameters
            st.subheader("Holt-Winters Parameters")
            alpha_holt = st.slider("Smoothing Level (Alpha)", 0.01, 1.0, 0.2, 0.01)
            beta_holt = st.slider("Smoothing Trend (Beta)", 0.01, 1.0, 0.1, 0.01)


            # Nếu chọn "Khác", cho phép nhập số ngày dự đoán
            if forecast_period == "Khác":
                custom_days = st.number_input("Nhập số ngày dự đoán:", min_value=1, value=1)
                ma_period = custom_days  # Gán custom_days cho ma_period nếu chọn "Khác"
            else:
                # Gán ma_period dựa trên forecast_period đã chọn
                ma_period = {
                    "1 tháng": 1,
                    "6 tháng": 6,
                    "12 tháng": 12,
                }[forecast_period]

        elif model_choice == "Holt Winter By Day":
            st.subheader("Chọn thời gian dự đoán:")

            seasonality_periods = st.number_input("Giai đoạn mùa vụ", min_value=1, value=252, step=1)

            forecast_period = st.selectbox("Thời gian:",
                                                ["1 ngày", "1 tuần (5 ngày)",
                                                "1 tháng (22 ngày)", "Khác"])

            st.subheader("Chọn hệ số alpha và beta:")
            alpha = st.slider("Alpha (Smoothing Level):", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
            beta = st.slider("Beta (Smoothing Trend):", min_value=0.01, max_value=1.0, value=0.2, step=0.01)
            gamma = st.slider("Gamma (Smoothing Seasonality):", min_value=0.01, max_value=1.0, value=0.2, step=0.01)


            # Nếu chọn "Khác", cho phép nhập số ngày dự đoán
            if forecast_period == "Khác":
                custom_days = st.number_input("Nhập số ngày dự đoán:", min_value=1, value=1)
                ma_period = custom_days  # Gán custom_days cho ma_period nếu chọn "Khác"
            else:
                # Gán ma_period dựa trên forecast_period đã chọn
                ma_period = {
                    "1 ngày": 1,
                    "1 tuần (5 ngày)": 5,
                    "1 tháng (22 ngày)": 22,
                }[forecast_period]

        elif model_choice == "Holt Winter By Month":
            st.subheader("Chọn thời gian dự đoán:")

            seasonality_periods = st.number_input("Giai đoạn mùa vụ", min_value=1, value=12, step=1)

            forecast_period = st.selectbox("Thời gian:",
                                                ["1 tháng", "6 tháng",
                                                "12 tháng", "Khác"])

            # Add sliders for Holt-Winters parameters
            st.subheader("Holt-Winters Parameters")
            alpha_hwm = st.slider("Smoothing Level (Alpha)", 0.01, 1.0, 0.2, 0.01)
            beta_hwm = st.slider("Smoothing Trend (Beta)", 0.01, 1.0, 0.1, 0.01)
            gamma_hwm = st.slider("Smoothing Seasonal (Gamma)", 0.01, 1.0, 0.1, 0.01)


            # Nếu chọn "Khác", cho phép nhập số ngày dự đoán
            if forecast_period == "Khác":
                custom_days = st.number_input("Nhập số ngày dự đoán:", min_value=1, value=1)
                ma_period = custom_days  # Gán custom_days cho ma_period nếu chọn "Khác"
            else:
                # Gán ma_period dựa trên forecast_period đã chọn
                ma_period = {
                    "1 tháng": 1,
                    "6 tháng": 6,
                    "12 tháng": 12,
                }[forecast_period]

        # Nút Dự báo
        if st.button('Dự đoán'):
            if selected_file:
                # Lấy đường dẫn đến thư mục dataset
                dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
                file_path = os.path.join(dataset_dir, selected_file)

                # Đọc dữ liệu từ file CSV đã chọn
                df = pd.read_csv(file_path)
                df['Date'] = pd.to_datetime(df['Date'])  # Chuyển đổi cột 'Date' thành kiểu datetime
                df.set_index('Date', inplace=True)


            # Xử lý dự đoán dựa trên model_choice
            if model_choice == "Simple Moving Average":
                # Vẽ biểu đồ SMA (Adj Close và MA) với dự đoán
                # custom_days is defined within your 'Predict' section - ensure it is defined
                fig_ma_adj_close, df_pred, mae_ma, rmse_ma, mape_ma = create_adj_close_ma_chart_with_prediction(
                    df, ma_window=ma_window, forecast_days=forecast_days, ma_period=ma_period
                )  
                
                    # Hiển thị chỉ số lỗi cho MA
                st.write(f"**Chỉ số lỗi (MA):**")
                st.write(f"  - MAE: {mae_ma:.2f}")
                st.write(f"  - RMSE: {rmse_ma:.2f}")
                st.write(f"  - MAPE: {mape_ma:.2f}%")

                # This line was causing issues
                st.plotly_chart(fig_ma_adj_close, use_container_width=True)

                # Hiển thị bảng dự đoán
                st.subheader("Bảng dự đoán:")
                st.dataframe(df_pred)

            elif model_choice == "Exponential Smoothing By Day":
           
                # Vẽ biểu đồ với dự đoán bằng Exponential Smoothing
                fig_es_adj_close, df_pred, mae, rmse, mape = create_adj_close_es_chart_with_prediction(df, smoothing_level=smoothing_level, forecast_days=ma_period)

                # Hiển thị chỉ số lỗi
                st.write(f"Alpha: {smoothing_level:.2f}%")
                st.write(f"**Chỉ số lỗi (ES):**")
                st.write(f"  - MAE: {mae:.2f}")
                st.write(f"  - RMSE: {rmse:.2f}")
                st.write(f"  - MAPE: {mape:.2f}%")

                st.plotly_chart(fig_es_adj_close, use_container_width=True)

                # Hiển thị bảng dự đoán
                st.subheader("Bảng dự đoán:")
                st.dataframe(df_pred)

            elif model_choice == "Exponential Smoothing By Month":

                # Call the ES monthly function
                fig_es_monthly, df_pred_es_monthly, mae, rmse, mape = apply_es_monthly(df, alpha_es, ma_period)

                # Hiển thị chỉ số lỗi
                st.write(f"Alpha: {alpha_es:.2f}%")
                st.write(f"**Chỉ số lỗi (ES):**")
                st.write(f"  - MAE: {mae:.2f}")
                st.write(f"  - RMSE: {rmse:.2f}")
                st.write(f"  - MAPE: {mape:.2f}%")

                # Display the chart and prediction table
                st.plotly_chart(fig_es_monthly, use_container_width=True)
                st.subheader("Bảng dự đoán ES (Monthly):")
                st.dataframe(df_pred_es_monthly)  # Display the prediction DataFrame

            elif model_choice == "Holt By Day":

                # Call the cached function
                fig_holt, df_pred_holt = create_adj_close_holt_chart_with_prediction(
                    df, smoothing_level=alpha, beta=beta, forecast_days=ma_period
                )

                # Hiển thị biểu đồ và bảng dự đoán (chỉ một lần)
                st.plotly_chart(fig_holt, use_container_width=True)  # Giữ lại lệnh này
                st.subheader("Bảng dự đoán Holt:")
                st.dataframe(df_pred_holt)

            elif model_choice == "Holt By Month":
            # Call the Holt-Winters monthly function
                fig_holt_monthly, df_pred_holt_monthly = apply_holt_monthly(
                    df,
                    smoothing_level=alpha_holt,
                    smoothing_trend=beta_holt,
                    forecast_days=ma_period
                )

                # Display the chart and prediction table
                st.plotly_chart(fig_holt_monthly, use_container_width=True)
                st.subheader("Bảng dự đoán Holt (Monthly):")
                st.dataframe(df_pred_holt_monthly)  

            elif model_choice == "Holt Winter By Day":

                # Call the Holt-Winters function
                fig_hw, df_pred_hw = create_adj_close_holt_winters_chart_with_prediction(
                    df, 
                    smoothing_level=alpha, 
                    smoothing_trend=beta, 
                    smoothing_seasonal=gamma, 
                    seasonality_periods=seasonality_periods, 
                    forecast_days=ma_period
                )

                # Hiển thị biểu đồ và bảng dự đoán (chỉ một lần)
                st.plotly_chart(fig_hw, use_container_width=True)
                st.subheader("Bảng dự đoán Holt-Winters:")
                st.dataframe(df_pred_hw)

            elif model_choice == "Holt Winter By Month":
            # Call the Holt-Winters monthly function
                fig_hwm, df_pred_hwm = apply_holt_winters_monthly(
                    df,
                    smoothing_level=alpha_hwm,
                    smoothing_trend=beta_hwm,
                    smoothing_seasonal=gamma_hwm,
                    forecast_days=ma_period
                )
                
                # Display the chart and prediction table
                st.plotly_chart(fig_hwm, use_container_width=True)
                st.subheader("Bảng dự đoán Holt-Winters (Monthly):")
                st.dataframe(df_pred_hwm)

            # Phần mã trong tab Dự đoán
            elif model_choice == "ARIMA":
                with st.spinner('Đang tải dữ liệu...'):
                    if df is not None and not df.empty:
                        # Số ngày dự đoán nhập từ người dùng
                        forecast_days = st.number_input('Số ngày dự đoán:', min_value=1, max_value=365, value=7, step=1)

                        # Gọi hàm dự báo ARIMA
                        forecast_values, train_predict, mae, mse, model_fit = arima_forecast(df, forecast_days)

                        # Tạo bảng hiển thị dự báo
                        future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
                        forecast_df = pd.DataFrame({
                            'Ngày': future_dates,
                            'Dự báo Giá': forecast_values
                        })

                        # Hiển thị biểu đồ với Plotly
                        st.subheader('Biểu đồ Dự báo')

                        # Tạo đồ thị giá cổ phiếu thực tế và dự báo
                        fig = go.Figure()

                        # Biểu đồ giá cổ phiếu thực tế (Adj Close) màu xanh lá
                        fig.add_trace(go.Scatter(
                            x=df.index, y=df['Adj Close'], mode='lines', name='Giá cổ phiếu (Adj Close)', line=dict(color='green')
                        ))

                        # Biểu đồ dự báo (dashed line)
                        fig.add_trace(go.Scatter(
                            x=future_dates, y=forecast_values, mode='lines', name='Dự báo ARIMA', line=dict(dash='dash', color='blue')
                        ))

                        # Biểu đồ đường huấn luyện (màu đỏ, nét liền)
                        fig.add_trace(go.Scatter(
                            x=df.index, y=train_predict, mode='lines', name='Đường huấn luyện', line=dict(color='red')
                        ))

                        # Cập nhật các thuộc tính của đồ thị
                        fig.update_layout(
                            title="Dự báo Giá Cổ phiếu với ARIMA",
                            xaxis_title="Ngày",
                            yaxis_title="Giá Cổ phiếu",
                            showlegend=True
                        )

                        # Hiển thị biểu đồ Plotly
                        st.plotly_chart(fig, use_container_width=True)

                        # Hiển thị bảng dự báo
                        st.write(f"Dự báo giá cổ phiếu trong {forecast_days} ngày tới:")
                        st.write(forecast_df)

                        # Hiển thị các tham số lỗi
                        st.write(f"Tham số lỗi trên dữ liệu huấn luyện:")
                        st.write(f"MAE: {mae:.4f}")
                        st.write(f"MSE: {mse:.4f}")
                    else:
                        st.error(f"Không tìm thấy mã chứng khoáng {symbol}")

            else:
                st.warning("Vui lòng chọn một file CSV để tiếp tục.")





    # Advanced Stock Price Prediction Tab
    elif selected_tab == "Dự đoán nâng cao":
         # Dùng HTML để căn giữa tiêu đề chính
        st.markdown(
        """
        <h1 style="text-align:center;">📊 Phân tích dự đoán nâng cao</h1>
        """,
        unsafe_allow_html=True
        )

            # Get the settings from the main screen:
        settings = add_settings_main()

        # Divide the layout into 3 columns for input
        col1, col2, col3 = st.columns(3)

        with col1:
            symbol = st.text_input('Mã chứng khoán', 'NKE')

        with col2:
            start_date = st.date_input('Ngày bắt đầu', datetime.now() - timedelta(days=365))

        with col3:
            forecast_days = st.number_input("Thời gian dự đoán", min_value=1, max_value=365, value=7, step=1)  # Replace slider with number input
    

        # Generate Forecast button
        if st.button('Dự đoán', use_container_width=True):
            with st.spinner('Loading data...'):
                df = get_stock_data(symbol, start_date)

                if df is not None and not df.empty:
                    # Calculate technical indicators
                    df = calculate_technical_indicators(df)

                    # Generate predictions
                    predictions = predict_prices(df, forecast_days)

                    if predictions:
                        future_dates = pd.date_range(
                            start=df.index[-1] + pd.Timedelta(days=1),
                            periods=forecast_days
                        )

                        metrics = calculate_metrics(df, predictions, forecast_days)

                        # Display important metrics
                        display_enhanced_metrics(metrics)

                        # Create container for the chart
                        chart_container = st.container()

                        with chart_container:
                            # Main chart
                            st.subheader('Biểu đồ và Bảng dự đoán')
                            fig = create_enhanced_chart(df, predictions, future_dates, symbol, settings)
                            st.plotly_chart(fig, use_container_width=True)

                            # Prediction Table
                            display_prediction_table(future_dates, predictions, metrics)
                    else:
                        st.error("Failed to generate predictions")
                else:
                    st.error(f"Không tìm thấy mã chứng khoáng {symbol}")

if __name__ == "__main__":
    main()
