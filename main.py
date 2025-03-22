import os
import time
import datetime
import requests
import ta
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from googletrans import Translator
from textblob import TextBlob
import dash
from dash import Dash, dcc, html, Input, Output, callback
from dash import dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.subplots as sp
import nltk

nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

# Cấu hình Chrome ở chế độ headless
chrome_options = Options()
chrome_options.add_argument("--headless")  # Chạy không có giao diện
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# Khởi tạo ứng dụng Dash
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Biến toàn cục để lưu trữ mã cổ phiếu và nội dung trang Tổng quan
stock_code = None
overview_content = None
df_stock = None  # Biến để lưu trữ dữ liệu cổ phiếu
df_news = None  # Biến để lưu trữ dữ liệu tin tức

# Layout của ứng dụng
app.layout = html.Div(
    className="bg-blue-100 font-sans",
    children=[
        dbc.Container(
            className="p-4",
            children=[
                # Header
                dbc.Row(
                    className="mb-4 align-items-center",
                    children=[
                        dbc.Col(
                            className="d-flex align-items-center",
                            children=[
                                html.Img(
                                    src="https://i.postimg.cc/XNQHGkx1/snapedit-1742179026556-removebg-preview.png",
                                    height="50",
                                    width="50",
                                    className="mr-2",
                                    alt="Stock logo"
                                ),
                                dbc.Input(
                                    id='stock-code-input',
                                    placeholder='Nhập mã cổ phiếu (VD: VNM, HPG, SSI...)',
                                    type="text",
                                    className="border rounded p-2",
                                    style={"width": "200px"}
                                ),
                                dbc.Button(
                                    "PHÂN TÍCH",
                                    color="light",
                                    className="ml-2",
                                    id='analyze-button',
                                    style={"margin-left": "10px", "width": "150px"}
                                ),
                            ],
                        ),
                        dbc.Col(
                            className="d-flex justify-content-end",
                            children=[
                                dbc.Button("TỔNG QUAN", color="light", className="mr-2", id='overview-button',
                                           style={"width": "150px"}),
                                dbc.Button("BIỂU ĐỒ", color="light", className="mr-2", id='chart-button',
                                           style={"width": "150px"}),
                                dbc.Button("TIN TỨC", color="light", id='news-button', style={"width": "150px"}),
                            ],
                        ),
                    ],
                ),
                # Main Content
                html.Div(id='page-content'),
            ],
        ),
    ],
)


# Nội dung cho trang Tổng quan
def get_overview_content():
    return html.Div(
        children=[
            html.H2("TỔNG QUAN", className="text-lg font-bold mb-2"),
            html.P("Nội dung tổng quan sẽ được hiển thị ở đây."),
        ]
    )


# Nội dung cho trang Biểu đồ
def get_chart_content(df_stock):
    # Tính toán các chỉ số cho biểu đồ
    df_stock["Middle_Band"] = df_stock["Giá Đóng cửa (nghìn VNĐ)"].rolling(window=20).mean()
    df_stock["Std_Dev"] = df_stock["Giá Đóng cửa (nghìn VNĐ)"].rolling(window=20).std()
    df_stock["Upper_Band"] = df_stock["Middle_Band"] + (df_stock["Std_Dev"] * 2)
    df_stock["Lower_Band"] = df_stock["Middle_Band"] - (df_stock["Std_Dev"] * 2)

    df_filtered = df_stock.dropna(
        subset=["Giá Đóng cửa (nghìn VNĐ)", "EMA_12", "EMA_26", "Upper_Band", "Lower_Band", "RSI"])
    latest_date = df_filtered["Ngày"].max()
    df_filtered = df_filtered[df_filtered["Ngày"] <= latest_date]

    # Tạo biểu đồ
    fig = sp.make_subplots(rows=3, cols=1, shared_xaxes=True,
                           vertical_spacing=0.1,
                           row_heights=[0.4, 0.3, 0.3],
                           subplot_titles=("Stock Price & Bollinger Bands", "MACD & Signal Line", "RSI Indicator"))

    fig.add_trace(go.Scatter(x=df_filtered["Ngày"], y=df_filtered["Giá Đóng cửa (nghìn VNĐ)"],
                             mode='lines', name='Price', line=dict(color='blue', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_filtered["Ngày"], y=df_filtered["Upper_Band"],
                             mode='lines', name='Bollinger Upper', line=dict(color='gray', width=1, dash="dash")),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=df_filtered["Ngày"], y=df_filtered["Lower_Band"],
                             mode='lines', name='Bollinger Lower', line=dict(color='gray', width=1, dash="dash")),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=df_filtered['Ngày'], y=df_filtered['MACD'], mode='lines',
                             line=dict(color='orange', width=2.5), name="MACD"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_filtered['Ngày'], y=df_filtered['Signal_Line'], mode='lines',
                             line=dict(color='red', width=2, dash='dot'), name="Signal Line"), row=2, col=1)

    fig.add_trace(go.Scatter(x=df_filtered["Ngày"], y=df_filtered["RSI"],
                             mode='lines', name='RSI', line=dict(color='purple', width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_filtered["Ngày"], y=[70] * len(df_filtered),
                             mode="lines", name="RSI 70 (Overbought)", line=dict(color="red", width=1, dash="dash")),
                  row=3, col=1)
    fig.add_trace(go.Scatter(x=df_filtered["Ngày"], y=[30] * len(df_filtered),
                             mode="lines", name="RSI 30 (Oversold)", line=dict(color="green", width=1, dash="dash")),
                  row=3, col=1)

    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="#f8f9fa",
        font=dict(size=14),
        title=f"{stock_code.upper()} Chart",
        xaxis_title="Date",
        showlegend=True,
        height=800
    )

    return dcc.Graph(figure=fig)


# Nội dung cho trang Tin tức
def get_news_content(df_news):
    news_output = []
    for _, row in df_news.iterrows():
        news_output.append(html.Div(
            className="news-item mb-4 p-3 border rounded d-flex align-items-start",
            style={'display': 'flex', 'alignItems': 'center', 'border': '1px solid #ccc',
                   'boxShadow': '0 0 10px rgba(0, 0, 0, 0.1)'},
            children=[
                html.Img(src=row['Hình ảnh'], style={'width': '150px', 'height': '150px', 'marginRight': '15px'},
                         title=row['Tiêu đề bài báo'],
                         id={'type': 'news-image', 'index': _}),
                html.Div(
                    children=[
                        html.H4(row['Tiêu đề bài báo'], style={'fontWeight': 'bold', 'color': '#111111'}),
                        html.P(f"Ngày: {row['Ngày']}", className="text-muted"),
                        html.P(row['Nội dung chính']),
                        html.A("Đọc thêm", href=row['Link bài báo'], target="_blank",
                               className="btn btn-dark", style={'color': 'white'})  # Thay đổi màu nút ở đây
                    ],
                    style={'flex': '1', 'backgroundColor': '#f9f9f9', 'padding': '10px'}
                )
            ]
        ))

    return html.Div(children=news_output)


@app.callback(
    [Output('page-content', 'children'),
     Output('analyze-button', 'color'),
     Output('overview-button', 'color'),
     Output('chart-button', 'color'),
     Output('news-button', 'color')],
    [Input('overview-button', 'n_clicks'),
     Input('chart-button', 'n_clicks'),
     Input('news-button', 'n_clicks'),
     Input('analyze-button', 'n_clicks'),
     Input('stock-code-input', 'value')]
)
def update_content(overview_clicks, chart_clicks, news_clicks, analyze_clicks, input_stock_code):
    global stock_code, overview_content, df_stock, df_news  # Sử dụng biến toàn cục

    ctx = dash.callback_context

    if not ctx.triggered:
        return get_overview_content(), "dark", "dark", "light", "light"  # Mặc định hiển thị trang Tổng quan

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'overview-button':
        return overview_content if overview_content else get_overview_content(), "dark", "dark", "light", "light"

    elif button_id == 'chart-button':
        if df_stock is not None:  # Kiểm tra xem có dữ liệu cổ phiếu đã lưu hay không
            return get_chart_content(df_stock), "dark", "light", "dark", "light"
        elif input_stock_code:  # Nếu không có dữ liệu, kiểm tra mã cổ phiếu đã nhập
            df_stock = fetch_stock_data(input_stock_code)
            if df_stock is None:
                return "KHÔNG CÓ KẾT QUẢ PHÙ HỢP. Vui lòng kiểm tra lại mã cổ phiếu.", "dark", "light", "dark", "light"
            df_stock = calculate_indicators(df_stock)  # Tính toán các chỉ số trước khi gọi get_chart_content
            return get_chart_content(df_stock), "dark", "light", "dark", "light"
        else:
            return html.Div([
                html.H2("BIỂU ĐỒ", className="text-lg font-bold mb-2"),
                html.P("Nội dung biểu đồ sẽ được hiển thị ở đây."),
            ]), "dark", "light", "dark", "light"

    elif button_id == 'news-button':
        if input_stock_code:
            df_news = fetch_news_data(input_stock_code)  # Lấy dữ liệu tin tức
            if df_news is None or df_news.empty:
                return "KHÔNG CÓ DỮ LIỆU TIN TỨC TRONG 7 NGÀY GẦN NHẤT.", "dark", "light", "light", "dark"
            return get_news_content(df_news), "dark", "light", "light", "dark"
        else:
            return html.Div([
                html.H2("TIN TỨC", className="text-lg font-bold mb-2"),
                html.P("Nội dung tin tức 7 ngày gần nhất sẽ được hiển thị ở đây."),
            ]), "dark", "light", "light", "dark"

    elif button_id == 'analyze-button' and input_stock_code:
        stock_code = input_stock_code  # Lưu mã cổ phiếu đã nhập
        df_stock = fetch_stock_data(stock_code)
        if df_stock is None:
            return "KHÔNG CÓ KẾT QUẢ PHÙ HỢP. Vui lòng kiểm tra lại mã cổ phiếu.", "dark", "light", "light", "light"

        df_stock = calculate_indicators(df_stock)  # Tính toán các chỉ số
        df_news = fetch_news_data(stock_code)
        sentiment_score, total_articles = analyze_sentiment(df_news)

        latest_rsi = df_stock["RSI"].dropna().iloc[-1] if not df_stock["RSI"].isna().all() else None
        latest_macd = df_stock["MACD"].dropna().iloc[-1] if not df_stock["MACD"].isna().all() else None
        latest_signal_line = df_stock["Signal_Line"].dropna().iloc[-1] if not df_stock[
            "Signal_Line"].isna().all() else None
        latest_stoch_k = df_stock["%K"].dropna().iloc[-1] if not df_stock["%K"].isna().all() else None
        latest_stoch_d = df_stock["%D"].dropna().iloc[-1] if not df_stock["%D"].isna().all() else None
        latest_close_price = df_stock["Giá Đóng cửa (nghìn VNĐ)"].dropna().iloc[-1] if not df_stock[
            "Giá Đóng cửa (nghìn VNĐ)"].isna().all() else None

        recommendation = get_recommendation(latest_rsi, sentiment_score, latest_macd, latest_signal_line,
                                            latest_stoch_k, latest_stoch_d)

        report = html.Div([
            html.H2("BÁO CÁO TỔNG QUAN", className="text-black font-bold mb-2"),
            html.Ul(
                className="list-disc pl-5",
                children=[
                    html.Li(
                        f"Giá đóng cửa gần nhất: {latest_close_price:.2f} VNĐ" if latest_close_price is not None else "Giá đóng cửa: Không có dữ liệu",
                        style={"color": "black"}),
                    html.Li(f"RSI gần nhất: {latest_rsi:.2f}" if latest_rsi is not None else "RSI: Không có dữ liệu",
                            style={"color": "black"}),
                    html.Li(
                        f"MACD: {latest_macd:.4f}, Signal Line: {latest_signal_line:.4f}" if latest_macd is not None else "MACD: Không có dữ liệu",
                        style={"color": "black"}),
                    html.Li(
                        f"Stochastic %K: {latest_stoch_k:.2f}, %D: {latest_stoch_d:.2f}" if latest_stoch_k is not None else "Stochastic: Không có dữ liệu",
                        style={"color": "black"}),
                    html.Li(f"Tổng số bài báo: {total_articles}", style={"color": "black"}),
                    html.Li(f"Sentiment Score: {sentiment_score}", style={"color": "black"}),
                ],
            ),
        ])

        transaction_table = dash_table.DataTable(
            data=df_stock.tail(30).to_dict('records'),
            columns=[{"name": i, "id": i} for i in df_stock.columns],
            export_format='xlsx',
            style_table={'overflowX': 'auto'},
        )

        news_table = dash_table.DataTable(
            data=df_news.to_dict('records'),
            columns=[{"name": i, "id": i} for i in df_news.columns],
            export_format='xlsx',
            style_table={'overflowX': 'auto'},
        )

        recommendation_display = create_recommendation_display(recommendation)

        # Lưu nội dung Tổng quan
        overview_content = html.Div([
            dbc.Row(
                className="mb-4",
                children=[
                    dbc.Col(
                        className="bg-white p-4 rounded shadow",
                        style={"border": "2px solid black", "borderRadius": "10px", "marginRight": "10px"},
                        # Thêm marginRight
                        children=[
                            html.H2("BÁO CÁO TỔNG QUAN", className="text-black font-bold mb-2"),
                            html.Ul(
                                className="list-disc pl-5",
                                children=[
                                    html.Li(
                                        f"Giá đóng cửa gần nhất: {latest_close_price:.2f} VNĐ" if latest_close_price is not None else "Giá đóng cửa: Không có dữ liệu",
                                        style={"color": "black"}),
                                    html.Li(
                                        f"RSI gần nhất: {latest_rsi:.2f}" if latest_rsi is not None else "RSI: Không có dữ liệu",
                                        style={"color": "black"}),
                                    html.Li(
                                        f"MACD: {latest_macd:.4f}, Signal Line: {latest_signal_line:.4f}" if latest_macd is not None else "MACD: Không có dữ liệu",
                                        style={"color": "black"}),
                                    html.Li(
                                        f"Stochastic %K: {latest_stoch_k:.2f}, %D: {latest_stoch_d:.2f}" if latest_stoch_k is not None else "Stochastic: Không có dữ liệu",
                                        style={"color": "black"}),
                                    html.Li(f"Tổng số bài báo: {total_articles}", style={"color": "black"}),
                                    html.Li(f"Sentiment Score: {sentiment_score}", style={"color": "black"}),
                                ],
                            ),
                        ],
                    ),
                    dbc.Col(
                        className="bg-white p-4 rounded shadow",
                        style={"border": "2px solid black", "borderRadius": "10px"},  # Không cần marginRight ở đây
                        children=[
                            html.H2("KHUYẾN NGHỊ", className="font-bold text-lg", style={"textAlign": "left"}),
                            html.P(f"{recommendation.upper()}", className="font-bold text-xl",
                                   style={"textAlign": "left", "marginTop": "5px"}),
                            recommendation_display
                        ],
                    ),
                ],
            ),
            dbc.Row(
                className="mb-4",
                children=[
                    dbc.Col(
                        className="bg-light p-4 rounded",
                        children=[
                            html.H2("DỮ LIỆU LỊCH SỬ", className="text-lg font-bold mb-2"),
                            transaction_table
                        ],
                    ),
                    dbc.Col(
                        className="bg-white p-4 rounded shadow",
                        children=[
                            html.H2("TIN TỨC", className="text-lg font-bold mb-2"),
                            news_table
                        ],
                    ),
                ],
            ),
        ])
        return overview_content, "dark", "dark", "light", "light"

    return "", "dark", "dark", "light", "light"


# Hàm lấy dữ liệu giao dịch cổ phiếu
def fetch_stock_data(stock_code, days_limit=30):
    url = f"https://s.cafef.vn/Lich-su-giao-dich-{stock_code}-1.chn"

    # Tự động tải và sử dụng ChromeDriver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    driver.get(url)

    all_data = []
    days_collected = 0
    header = [
        "Ngày", "Giá Đóng cửa (nghìn VNĐ)", "Giá Điều chỉnh (nghìn VNĐ)", "Thay đổi",
        "GD khớp lệnh (Khối lượng)", "GD khớp lệnh (Giá trị (tỷ VNĐ))",
        "GD thỏa thuận (Khối lượng)", "GD thỏa thuận (Giá trị (tỷ VNĐ))",
        "Giá Mở cửa (nghìn VNĐ)", "Giá Cao nhất (nghìn VNĐ)", "Giá Thấp nhất (nghìn VNĐ)"
    ]

    while days_collected < days_limit:
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "wrapper-table-information-owner"))
            )
            page_html = driver.page_source
            soup = BeautifulSoup(page_html, "html.parser")

            # Kiểm tra xem có thông báo "KHÔNG CÓ KẾT QUẢ PHÙ HỢP" hay không
            no_results_message = soup.find("td", colspan="13", string="KHÔNG CÓ KẾT QUẢ PHÙ HỢP")
            if no_results_message:
                return None

            data_div = soup.find("div", class_="wrapper-table-information-owner")
            body = data_div.find("tbody")
            if body:
                rows = body.find_all("tr")
                for row in rows:
                    row_data = [td.get_text(strip=True) for td in row.find_all("td")]
                    if row_data:
                        all_data.append(row_data)
                        days_collected += 1
                        if days_collected >= days_limit:
                            break

            if days_collected >= days_limit:
                break
            next_button = driver.find_element(By.ID, "paging-right")
            driver.execute_script("arguments[0].click();", next_button)
            time.sleep(2)
        except Exception as e:
            print(f"Error: {e}")
            break

    driver.quit()
    if all_data:
        df_stock = pd.DataFrame(all_data, columns=header)
        df_stock['Ngày'] = pd.to_datetime(df_stock['Ngày'], format='%d/%m/%Y')
        df_stock.sort_values(by='Ngày', ascending=False, inplace=True)
        return df_stock
    else:
        return None  # Trả về None nếu không có dữ liệu


# Hàm tính toán RSI, MACD, STOCH
def calculate_indicators(df_stock, period_rsi=14, stoch_period=14):
    df_stock = df_stock.sort_values(by="Ngày").copy()
    df_stock["Giá Đóng cửa (nghìn VNĐ)"] = pd.to_numeric(df_stock["Giá Đóng cửa (nghìn VNĐ)"], errors="coerce")

    df_stock["RSI"] = ta.momentum.RSIIndicator(df_stock["Giá Đóng cửa (nghìn VNĐ)"], window=period_rsi).rsi()
    df_stock["EMA_12"] = df_stock["Giá Đóng cửa (nghìn VNĐ)"].ewm(span=12, adjust=False).mean()
    df_stock["EMA_26"] = df_stock["Giá Đóng cửa (nghìn VNĐ)"].ewm(span=26, adjust=False).mean()
    df_stock["MACD"] = df_stock["EMA_12"] - df_stock["EMA_26"]
    df_stock["Signal_Line"] = df_stock["MACD"].ewm(span=9, adjust=False).mean()
    df_stock["L"] = df_stock["Giá Đóng cửa (nghìn VNĐ)"].rolling(window=stoch_period).min()
    df_stock["H"] = df_stock["Giá Đóng cửa (nghìn VNĐ)"].rolling(window=stoch_period).max()
    df_stock["%K"] = ((df_stock["Giá Đóng cửa (nghìn VNĐ)"] - df_stock["L"]) / (df_stock["H"] - df_stock["L"])) * 100
    df_stock["%D"] = df_stock["%K"].rolling(window=3).mean()

    return df_stock


# Hàm lấy dữ liệu tin tức
def fetch_news_data(stock_code):
    url = f'https://cafef.vn/{stock_code}.html'
    all_results = []
    seven_days_ago = datetime.datetime.now() - datetime.timedelta(days=7)

    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    items = soup.find_all("div", class_="tlitem")
    for item in items:
        try:
            title_tag = item.find("h3").find("a") if item.find("h3") else None
            title = title_tag.get_text(strip=True) if title_tag else "Không có tiêu đề"
            time_tag = item.find(class_="time")
            date_str = time_tag.get_text(strip=True) if time_tag else None
            formatted_date = None
            if date_str:
                try:
                    date_obj = datetime.datetime.strptime(date_str, "%d/%m/%Y %H:%M")
                    formatted_date = date_obj.strftime("%Y-%m-%d")
                except ValueError:
                    continue
            content_tag = item.find(class_="sapo")
            content = content_tag.get_text(strip=True) if content_tag else "Không có nội dung"
            article_link = None
            avatar_tag = item.find("a", class_="avatar img-resize")
            if avatar_tag:
                raw_href = avatar_tag.get("href")
                article_link = f"https://cafef.vn{raw_href}" if raw_href and raw_href.startswith("/") else raw_href
            image_url = None
            image_tag = avatar_tag.find("img") if avatar_tag else None
            if image_tag:
                raw_src = image_tag.get("data-src") or image_tag.get("src")
                if raw_src:
                    if raw_src.startswith("//"):
                        image_url = f"https:{raw_src}"
                    elif raw_src.startswith("/"):
                        image_url = f"https://cafef.vn{raw_src}"
                    else:
                        image_url = raw_src
            if formatted_date and datetime.datetime.strptime(formatted_date, "%Y-%m-%d") >= seven_days_ago:
                all_results.append({
                    'Tiêu đề bài báo': title,
                    'Ngày': formatted_date,
                    'Nội dung chính': content,
                    'Hình ảnh': image_url,
                    'Link bài báo': article_link
                })
        except Exception as e:
            print(f"Lỗi khi xử lý bài viết: {e}")
            continue

    return pd.DataFrame(all_results)


# Hàm phân tích cảm xúc
def analyze_sentiment(news_df):
    translator = Translator()
    sentiment_score = 0
    total_articles = len(news_df)

    if total_articles == 0:
        return 50, 0

    for _, article in news_df.iterrows():
        try:
            if article['Nội dung chính'] is None or article['Nội dung chính'].strip() == "":
                continue

            translated_text = translator.translate(article['Nội dung chính'], src='vi', dest='en').text
            word_tokens = word_tokenize(translated_text)
            filtered_text = ' '.join(word for word in word_tokens if word.lower() not in stop_words)

            sentiment = TextBlob(filtered_text).sentiment.polarity
            sentiment_score += sentiment
            time.sleep(1)
        except Exception as e:
            print(f"Lỗi khi dịch bài viết: {e}")
            continue

    average_sentiment = sentiment_score / total_articles if total_articles > 0 else 0
    scaled_sentiment = int((average_sentiment + 1) * 50)
    return scaled_sentiment, total_articles


# Hàm đưa ra khuyến nghị
def get_recommendation(rsi, sentiment_score, macd, signal_line, stoch_k, stoch_d):
    if any(pd.isna(x) for x in [rsi, sentiment_score, macd, signal_line, stoch_k, stoch_d]):
        return "Neutral"
    buy_conditions = [
        sentiment_score > 60,
        rsi < 30,
        macd > signal_line,
        stoch_k < 30,
        stoch_k > stoch_d
    ]
    sell_conditions = [
        sentiment_score < 35,
        rsi > 70,
        macd < signal_line,
        stoch_k > 80,
        stoch_k < stoch_d
    ]
    buy_score = sum(buy_conditions)
    sell_score = sum(sell_conditions)
    if buy_score >= 2 and sell_score < 2:
        return "Buy"
    elif sell_score >= 2 and buy_score < 2:
        return "Sell"
    return "Neutral"


def create_recommendation_display(recommendation):
    if recommendation == "Buy":
        colors = ["red", "yellow", "green"]
        indicator_position = 2
        indicator_color = "black"
    elif recommendation == "Sell":
        colors = ["red", "yellow", "green"]
        indicator_position = 0
        indicator_color = "white"
    else:
        colors = ["red", "yellow", "green"]
        indicator_position = 1
        indicator_color = "black"

    return html.Div(
        style={"position": "relative", "width": "100%", "height": "50px"},
        children=[
            html.Div(style={"backgroundColor": colors[0], "height": "100%", "width": "33.33%", "float": "left"}),
            html.Div(style={"backgroundColor": colors[1], "height": "100%", "width": "33.33%", "float": "left"}),
            html.Div(style={"backgroundColor": colors[2], "height": "100%", "width": "33.33%", "float": "left"}),
            html.Div(
                style={
                    "position": "absolute",
                    "top": "0",
                    "left": f"{indicator_position * 33.33}%",
                    "width": "0",
                    "height": "0",
                    "borderLeft": "10px solid transparent",
                    "borderRight": "10px solid transparent",
                    "borderBottom": f"10px solid {indicator_color}",
                }
            )
        ]
    )


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run_server(host="0.0.0.0", port=5000, debug=True)