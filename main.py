import streamlit as st
from docx import Document
from io import BytesIO
from thefuzz import fuzz
import re
from collections import Counter
import pandas as pd
import nltk
import base64
import requests
import os
import datetime
from urllib.parse import urlparse
from nltk.tokenize import wordpunct_tokenize
from nltk.chat.util import Chat, reflections  # Kept for completeness, but no longer used for Eliza
import piexif
from PIL import Image
from bs4 import BeautifulSoup  # For HTML parsing in the HTML Processor tab

st.set_page_config(page_title="📄 Sibi's Research Bin", layout="wide")

# -----------------------------------------------------------------------------
#                           Precompile Regex Patterns
# -----------------------------------------------------------------------------
HEADER_REGEX = re.compile(r'^(Article|Content)\s*#\s*\d+', re.IGNORECASE)
URL_REGEX = re.compile(r'https?://\S+')
LINE_SPLIT_REGEX = re.compile(r'\s{2,}|\t+')

# -----------------------------------------------------------------------------
#                           NLTK Download
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def download_nltk_data():
    """
    Downloads necessary NLTK data packages once and caches them.
    """
    nltk.download('punkt', force=True)
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('stopwords')

download_nltk_data()

# -----------------------------------------------------------------------------
#                           CSV Loading
# -----------------------------------------------------------------------------
@st.cache_data
def load_client_database_csv():
    """
    Loads the client instructions CSV file from /home/databased.csv.
    Returns a pandas DataFrame or None if not found.
    """
    try:
        df = pd.read_csv('/home/databased.csv')
        return df
    except FileNotFoundError:
        return None

@st.cache_data
def load_keywords_csv():
    """
    Loads the keywords CSV file from /home/kw.csv.
    Returns a pandas DataFrame or None if not found.
    """
    try:
        df = pd.read_csv('/home/kw.csv')
        return df
    except FileNotFoundError:
        return None

# -----------------------------------------------------------------------------
#                           Keyword & Document Processing Functions
# -----------------------------------------------------------------------------
def parse_keywords_links(keywords_text):
    """
    Parses input text to extract keywords and their associated URLs or [No Hyperlink].
    Groups keywords under articles based on lines without URLs.
    """
    articles = []
    lines = keywords_text.splitlines()
    current_article_keywords = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # A line with no URL and not ending in [No Hyperlink] is a new article header
        if not URL_REGEX.search(line) and not line.endswith('[No Hyperlink]'):
            if current_article_keywords:
                articles.append(current_article_keywords)
                current_article_keywords = []
        else:
            parts = LINE_SPLIT_REGEX.split(line)
            if len(parts) >= 2:
                keyword = parts[0].strip()
                url_part = parts[1].strip()
                url = None if url_part.lower() == '[no hyperlink]' else url_part
                if keyword:
                    current_article_keywords.append((keyword, url))
    if current_article_keywords:
        articles.append(current_article_keywords)
    return articles

def parse_document_articles(doc):
    """
    Splits a DOCX document into articles using headers like 'Article #1'.
    Returns a list of articles, each a list of paragraph objects.
    """
    articles = []
    article_content = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if HEADER_REGEX.match(text):
            if article_content:
                articles.append(article_content)
                article_content = []
        else:
            if text:
                article_content.append(para)
    if article_content:
        articles.append(article_content)
    return articles

def get_paragraph_hyperlinks(paragraph):
    """
    Extracts all hyperlinks from a paragraph.
    Returns a list of (text, url) tuples.
    """
    hyperlinks = []
    for hyperlink in paragraph._element.findall('.//w:hyperlink', namespaces=paragraph._element.nsmap):
        r_id = hyperlink.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
        if r_id:
            url = paragraph.part.rels[r_id].target_ref
            text_elems = hyperlink.findall('.//w:t', namespaces=paragraph._element.nsmap)
            text = ''.join([t.text for t in text_elems if t.text])
            if text:
                hyperlinks.append((text, url))
    return hyperlinks

def extract_urls_from_paragraph(paragraph):
    """
    Extracts all URLs from a paragraph by analyzing its hyperlinks.
    """
    urls = []
    hyperlinks = get_paragraph_hyperlinks(paragraph)
    for _, url in hyperlinks:
        urls.append(url)
    return urls

def extract_article_content(article_paragraphs):
    """
    Extracts full text and all hyperlinks from a list of paragraphs representing an article.
    """
    article_text = ''
    article_hyperlinks = []
    for para in article_paragraphs:
        article_text += para.text + '\n'
        hyperlinks = get_paragraph_hyperlinks(para)
        article_hyperlinks.extend(hyperlinks)
    return article_text, article_hyperlinks

def preprocess_text(text):
    """
    Preprocess text by lowercasing, tokenizing, and removing certain tokens for cleaner matching.
    """
    tokens = wordpunct_tokenize(text.lower())
    tokens = [token for token in tokens if token not in ['in', ',']]
    return ' '.join(tokens)

def check_keyword_hyperlink(keyword, url, article_hyperlinks, article_text):
    """
    Checks keyword presence and correct hyperlinking in the article.
    Returns: 'correct', 'incorrect_url', or 'not_found'.
    """
    keyword_processed = preprocess_text(keyword)
    article_text_processed = preprocess_text(article_text)
    # Check if the keyword is present
    if keyword_processed.lower() in article_text_processed.lower():
        keyword_present = True
    else:
        keyword_present = False

    if not keyword_present:
        return 'not_found'

    # Attempt to match the hyperlink text to the processed keyword text
    matching_links = []
    for text, link in article_hyperlinks:
        text_processed = preprocess_text(text)
        if text_processed.lower() == keyword_processed.lower():
            matching_links.append(link)

    if not matching_links:
        return 'incorrect_url'

    # If a URL is specified, check if it matches
    for link in matching_links:
        if url and link.lower() == url.lower():
            return 'correct'

    return 'incorrect_url'

def analyze_dialect(text):
    """
    Analyzes dialect by counting occurrences of words associated with different dialects.
    Returns 'American', 'British', 'Canadian', 'Australian', or 'Unknown'.
    """
    dialect_words = {
        'American': {
            'color','center','organize','analyze','behavior','license','meter',
            'fiber','honor','favor','defense','offense','traveling','traveled','realize',
            'jewelry','program','catalog','dialog','liter','modeling','plow',
            'apartment','soccer','truck','cookie','movie','vacation'
        },
        'British': {
            'colour','centre','organise','analyse','behaviour','licence','metre',
            'fibre','honour','favour','defence','offence','travelling','travelled','realise',
            'jewellery','programme','catalogue','dialogue','litre','modelling','plough',
            'flat','football','lorry','biscuit','film','holiday'
        },
        'Canadian': {
            'colour','centre','organize','analyse','behaviour','licence','metre',
            'fibre','honour','favor','defence','offense','traveling','traveled','realize',
            'jewellery','program','catalog','dialogue','litre','modelling','plow',
            'apartment','soccer','truck','cookie','movie','vacation',
            'eh','tuque','toque','zed'
        },
        'Australian': {
            'colour','centre','organise','analyse','behaviour','licence','metre',
            'fibre','honour','favour','defence','offence','travelling','travelled','realise',
            'jewellery','programme','catalogue','dialogue','litre','modelling','plough',
            'flat','football','lorry','biscuit','film','holiday',
            'thongs','brolly','ute','arvo'
        },
    }
    word_list = re.findall(r'\b\w+\b', text.lower())
    word_counts = Counter(word_list)
    dialect_scores = {dialect:0 for dialect in dialect_words}
    for dialect, words in dialect_words.items():
        for word in words:
            dialect_scores[dialect] += word_counts.get(word,0)
    max_dialect = max(dialect_scores, key=dialect_scores.get)
    if dialect_scores[max_dialect] == 0:
        return "Unknown"
    return max_dialect

def generate_html_report(summary_data, num_images, all_hyperlinks):
    """
    Generates an HTML report showing analysis results:
    - Detected Dialect
    - Resource Box
    - Keyword status table
    - Additional Hyperlinks
    - Image count
    """
    report = ''
    for article in summary_data:
        report += f"<h3>{article['Article']}</h3>"
        report += f"<p><strong>Detected Dialect:</strong> {article['Detected Dialect']}</p>"
        report += f"<p><strong>Resource Box:</strong> {article['Resource Box']}</p>"
        keywords_df = pd.DataFrame(article['Keywords'])

        def color_status(row):
            color = row['Color']
            return [f'background-color: {color}; color: white']*len(row)

        styled_keywords = keywords_df.style.apply(color_status, axis=1)
        report += styled_keywords.to_html(escape=False)
        report += "<br>"

    report += f"<h4>Total images in the document: {num_images}</h4>"

    report += '<h4>Other Hyperlinks in the Document:</h4>'
    unique_hyperlinks = set(all_hyperlinks)
    if unique_hyperlinks:
        report += '<ul>'
        for text, url in unique_hyperlinks:
            report += f'<li>"{text}" linked to <a href="{url}" target="_blank">{url}</a></li>'
        report += '</ul>'
    else:
        report += "<p>No additional hyperlinks found.</p>"
    return report

# -----------------------------------------------------------------------------
#                           URL Checker Functions
# -----------------------------------------------------------------------------
def parse_input_urls(input_text):
    """
    Parses input text containing URLs separated by commas or newlines.
    """
    input_text = input_text.replace('\n', ',')
    urls = [url.strip() for url in input_text.split(',') if url.strip()]
    return urls

def normalize_url(url):
    """
    Ensures the URL starts with http:// or https://
    """
    if not re.match(r'http[s]?://', url):
        return 'http://' + url
    return url

def check_url_online(url):
    """
    Checks if the given URL is reachable online.
    Returns a tuple (status, error_message).
    """
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return ("Online", "")
        else:
            return (f"Error {response.status_code}: {response.reason}", "")
    except requests.exceptions.RequestException as e:
        return ("Offline", str(e))

# -----------------------------------------------------------------------------
#                           EXIF Editor Functions
# -----------------------------------------------------------------------------
def write_exif_tags(image_data, keywords="", description="", lat=None, lon=None):
    """
    Writes EXIF data (keywords, description, optional GPS coordinates) into a JPEG image.
    Returns a BytesIO object with the updated image.
    """
    img = Image.open(image_data)
    # Convert to JPEG if not already
    if img.format != 'JPEG':
        rgb_img = img.convert('RGB')
        buf = BytesIO()
        rgb_img.save(buf, format='JPEG')
        buf.seek(0)
        img = Image.open(buf)

    # Load or initialize EXIF
    exif_data = img.info.get('exif')
    if exif_data is None:
        exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
    else:
        exif_dict = piexif.load(exif_data)

    # Description tag
    description_tag = 270  # ImageDescription in 0th IFD
    exif_dict["0th"][description_tag] = description.encode('utf-8', 'replace')

    # XPKeywords tag
    xpkeywords_tag = 0x9C9E
    xpkeywords_str = keywords.encode('utf-16-le')
    exif_dict["0th"][xpkeywords_tag] = xpkeywords_str

    # If lat/lon are provided, write GPS data
    if lat is not None and lon is not None:
        lat_ref = 'N' if lat >= 0 else 'S'
        lon_ref = 'E' if lon >= 0 else 'W'
        lat = abs(lat)
        lon = abs(lon)

        def deg_to_dms_rational(deg):
            d = int(deg)
            m = int((deg - d) * 60)
            s = (deg - d - m/60) * 3600
            return [(d, 1), (m, 1), (int(s * 100), 100)]

        exif_dict['GPS'][piexif.GPSIFD.GPSLatitudeRef] = lat_ref.encode('ascii')
        exif_dict['GPS'][piexif.GPSIFD.GPSLongitudeRef] = lon_ref.encode('ascii')
        exif_dict['GPS'][piexif.GPSIFD.GPSLatitude] = deg_to_dms_rational(lat)
        exif_dict['GPS'][piexif.GPSIFD.GPSLongitude] = deg_to_dms_rational(lon)

    exif_bytes = piexif.dump(exif_dict)
    out_buf = BytesIO()
    img.save(out_buf, format='JPEG', exif=exif_bytes)
    out_buf.seek(0)
    return out_buf

# -----------------------------------------------------------------------------
#                           Main App
# -----------------------------------------------------------------------------
def main():
    st.title("📄 Sibi's Research Bin")

    # Create Tabs:
    # 1. Keyword Checker
    # 2. Exif Editor
    # 3. URL Checker
    # 4. Date Calculator  <--- REPLACED the Eliza Chatbot
    # 5. HTML Processor
    tabs = st.tabs(["Keyword Checker", "Exif Editor", "URL Checker", "Date Calculator", "HTML Processor"])

    # -----------------------------------
    # 1) Keyword Checker Tab
    # -----------------------------------
    with tabs[0]:
        st.markdown("""
        ### 📄 Keyword Checker for Word Documents

        **This tool allows you to:**
        - Check if each keyword is present in the uploaded Word document.
        - Verify if each keyword is correctly hyperlinked to the given URL.
        - Detect the dialect (American, British, Canadian, Australian) used in the article.

        **How to Use:**
        1. Upload a .docx file.
        2. Enter keywords and URLs (or `[No Hyperlink]`).
        3. Click "Run Analysis".
        4. View the detailed report below.
        """)
        uploaded_file = st.file_uploader("Upload .docx", type=["docx"])
        keywords_text = st.text_area(
            "Enter Keywords and URLs",
            height=300,
            placeholder="keyword    URL or [No Hyperlink]\nFor example:\nproperty in limassol    https://example.com/..."
        )
        run_button, clear_button = st.columns(2)
        with run_button:
            run = st.button("Run Analysis", key="run_analysis")
        with clear_button:
            clear = st.button("Clear Inputs", key="clear_inputs")

        output = st.container()

        if run:
            with output:
                st.markdown("### 📊 Analysis Report")
                if uploaded_file is None:
                    st.error("Please upload a Word document (.docx).")
                elif not keywords_text.strip():
                    st.error("Please enter the keywords and URLs.")
                else:
                    try:
                        doc = Document(BytesIO(uploaded_file.read()))
                    except Exception as e:
                        st.error(f"Error reading the document: {e}")
                        st.stop()

                    articles_keywords = parse_keywords_links(keywords_text)
                    if not articles_keywords:
                        st.error("No valid keywords and URLs found in the input.")
                        st.stop()

                    doc_articles = parse_document_articles(doc)
                    if not doc_articles:
                        st.error("No articles found in the document based on the expected headers.")
                        st.stop()

                    if len(articles_keywords) != len(doc_articles):
                        st.warning(
                            "The number of keyword groups does not match the number of articles."
                        )

                    report_data = []
                    all_hyperlinks = []
                    num_images = len(doc.inline_shapes)

                    for idx, article_keywords in enumerate(articles_keywords):
                        if idx < len(doc_articles):
                            article_paragraphs = doc_articles[idx]
                            article_text, article_hyperlinks = extract_article_content(article_paragraphs)
                            all_hyperlinks.extend(article_hyperlinks)
                            has_resource_box, resource_box_url = False, "N/A"
                            # Check for a 'resource box:' line near the end
                            for para in reversed(article_paragraphs):
                                if 'resource box:' in para.text.lower():
                                    has_resource_box = True
                                    urls = extract_urls_from_paragraph(para)
                                    if urls:
                                        resource_box_url = urls[-1]
                                    break

                            dialect = analyze_dialect(article_text)
                            keyword_results = []
                            for keyword, url in article_keywords:
                                status = check_keyword_hyperlink(
                                    keyword,
                                    url if url else "[No Hyperlink]",
                                    article_hyperlinks,
                                    article_text
                                )
                                if status == 'correct':
                                    symbol = '✔️ Correct'
                                    color = 'green'
                                    details = (
                                        f'Keyword "<strong>{keyword}</strong>" is correctly linked.'
                                        if url
                                        else f'Keyword "<strong>{keyword}</strong>" present with no hyperlink as specified.'
                                    )
                                elif status == 'incorrect_url':
                                    symbol = '⚠️ Incorrect URL'
                                    color = 'orange'
                                    details = (
                                        f'Keyword "<strong>{keyword}</strong>" is present but linked incorrectly.'
                                    )
                                else:
                                    symbol = '❌ Not Found'
                                    color = 'red'
                                    details = f'Keyword "<strong>{keyword}</strong>" not found in the article.'
                                keyword_results.append({
                                    'Status': symbol,
                                    'Details': details,
                                    'Color': color
                                })

                            report_data.append({
                                'Article': f'Article #{idx+1}',
                                'Detected Dialect': dialect,
                                'Resource Box': resource_box_url if has_resource_box else 'Not found or missing URL.',
                                'Keywords': keyword_results
                            })
                        else:
                            # If there are more keyword groups than articles
                            report_data.append({
                                'Article': f'Article #{idx+1}',
                                'Detected Dialect': 'Not found.',
                                'Resource Box': 'N/A',
                                'Keywords': [{
                                    'Status': '❌ Not Found',
                                    'Details': 'No article content.',
                                    'Color': 'red'
                                }]
                            })

                    report_html = generate_html_report(report_data, num_images, all_hyperlinks)
                    st.markdown(report_html, unsafe_allow_html=True)

        if clear:
            st.rerun()

    # -----------------------------------
    # 2) EXIF Editor Tab
    # -----------------------------------
    with tabs[1]:
        st.markdown("""
        ### 🖼️ EXIF Editor

        **This tool allows you to:**
        - Upload one or multiple images (JPEG or PNG).
        - View them in a simple gallery.
        - Edit EXIF data: keywords, description, and optional GPS coordinates (lat/long).
        - Download the updated image with the newly written EXIF tags.

        **How to Use:**
        1. Upload one or more images.
        2. Select an image from the gallery on the right.
        3. Enter EXIF details (keywords, description, lat/long).
        4. Click "Write EXIF Tags" to embed the data.
        5. Download your updated image.
        """)

        if "exif_images" not in st.session_state:
            st.session_state.exif_images = []
        if "selected_image_index" not in st.session_state:
            st.session_state.selected_image_index = None

        uploaded_exif_files = st.file_uploader(
            "Upload Image(s)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )
        if uploaded_exif_files:
            for f in uploaded_exif_files:
                img_bytes = f.read()
                st.session_state.exif_images.append(img_bytes)

        btn_clear_gallery, btn_delete_selected = st.columns(2)
        with btn_clear_gallery:
            if st.button("Clear Gallery"):
                st.session_state.exif_images = []
                st.session_state.selected_image_index = None

        with btn_delete_selected:
            if st.button("Delete Selected Image"):
                if (
                    st.session_state.selected_image_index is not None
                    and 0 <= st.session_state.selected_image_index < len(st.session_state.exif_images)
                ):
                    del st.session_state.exif_images[st.session_state.selected_image_index]
                    st.session_state.selected_image_index = None

        left_col, right_col = st.columns([2, 1])

        with right_col:
            st.markdown("#### Gallery")
            if st.session_state.exif_images:
                for i, img_data in enumerate(st.session_state.exif_images):
                    try:
                        img = Image.open(BytesIO(img_data))
                        st.image(img, width=120)
                        if st.button(f"Select Image {i+1}", key=f"select_image_{i}"):
                            st.session_state.selected_image_index = i
                    except Exception:
                        st.error("Error loading image from memory.")
            else:
                st.info("No images in the gallery yet. Please upload images above.")

        with left_col:
            st.markdown("#### EXIF Data Editor")
            idx = st.session_state.selected_image_index
            if idx is not None and 0 <= idx < len(st.session_state.exif_images):
                selected_image_data = st.session_state.exif_images[idx]

                keywords_tags = st.text_input("Keywords / Tags (comma-separated)")
                description_text = st.text_input("Description / Alt Text")
                lat_input = st.text_input("Latitude (e.g. 34.70713)")
                lon_input = st.text_input("Longitude (e.g. 33.02261)")

                try:
                    lat_val = float(lat_input.strip()) if lat_input.strip() else None
                    lon_val = float(lon_input.strip()) if lon_input.strip() else None
                except ValueError:
                    st.warning("Invalid latitude or longitude. Please ensure numeric values.")
                    lat_val, lon_val = None, None

                if lat_val is not None and lon_val is not None:
                    map_data = pd.DataFrame({'lat': [lat_val], 'lon': [lon_val]})
                    st.map(map_data)

                if st.button("Write EXIF Tags"):
                    updated_img = write_exif_tags(
                        BytesIO(selected_image_data),
                        keywords=keywords_tags,
                        description=description_text,
                        lat=lat_val,
                        lon=lon_val
                    )
                    st.session_state.exif_images[idx] = updated_img.read()
                    st.success("EXIF tags written successfully!")

                if idx < len(st.session_state.exif_images):
                    out_img_data = st.session_state.exif_images[idx]
                    b64_img = base64.b64encode(out_img_data).decode()
                    download_link = f'<a href="data:file/jpg;base64,{b64_img}" download="updated_image.jpg">Download Updated Image</a>'
                    st.markdown(download_link, unsafe_allow_html=True)
            else:
                st.info("Select an image from the gallery on the right to edit its EXIF data.")

    # -----------------------------------
    # 3) URL Checker Tab
    # -----------------------------------
    with tabs[2]:
        st.markdown("""
        ### 🔗 URL Checker

        **Check if a list of URLs is online and retrieve information from a CSV database.**

        **How to Use:**
        1. Ensure `/home/databased.csv` exists and contains a 'url' column.
        2. Enter URLs separated by commas or newlines.
        3. Click "Check URLs" to see which are online and any matching CSV info.
        """)
        urls_input = st.text_area("Enter URLs", height=200)
        check_urls_btn = st.button("Check URLs")

        if check_urls_btn:
            if not urls_input.strip():
                st.error("Please enter URLs.")
            else:
                csv_df = load_client_database_csv()
                if csv_df is None:
                    st.error("The client instructions CSV file 'databased.csv' was not found.")
                else:
                    if 'url' not in csv_df.columns:
                        st.error("The CSV file must contain a column named 'url'.")
                    else:
                        input_urls = parse_input_urls(urls_input)
                        if not input_urls:
                            st.error("No valid URLs found.")
                        else:
                            normalized_urls = [normalize_url(url) for url in input_urls]
                            results = []
                            for original_url, url in zip(input_urls, normalized_urls):
                                matching_row = csv_df[csv_df['url'].str.lower() == url.lower()]
                                status, error = check_url_online(url)
                                result = {'URL': original_url, 'Status': status}
                                if not matching_row.empty:
                                    row_data = matching_row.to_dict(orient='records')[0]
                                    for key, value in row_data.items():
                                        if key.lower() != 'url':
                                            result[key] = value
                                else:
                                    result['CSV Data'] = 'N/A'
                                if error:
                                    result['Error'] = error
                                results.append(result)
                            results_df = pd.DataFrame(results)
                            # Re-order columns for display
                            cols = ['URL','Status'] + [c for c in results_df.columns if c not in ['URL','Status']]
                            results_df = results_df[cols]
                            st.dataframe(results_df, use_container_width=True)

    # -----------------------------------
    # 4) Date Calculator Tab (Replacing Eliza Chatbot)
    # -----------------------------------
    with tabs[3]:
        st.markdown("""
        ### 📅 Date Calculator
        This tool lets you:
        - Find the number of days between two dates.
        - Add or subtract a certain number of days from a given date.
        """)
        calc_mode = st.radio("Select Mode", ("Date Difference", "Date Arithmetic"))

        if calc_mode == "Date Difference":
            st.markdown("**Find the number of days between two dates**")
            start_date = st.date_input("Select the start date")
            end_date = st.date_input("Select the end date")

            if start_date and end_date:
                diff = (end_date - start_date).days
                st.write(f"Number of days between {start_date.strftime('%m/%d/%y')} and {end_date.strftime('%m/%d/%y')} is: {abs(diff)} day(s).")

        else:  # Date Arithmetic
            st.markdown("**Add or Subtract Days from a Given Date**")
            base_date = st.date_input("Select the base date")
            days_offset = st.number_input("Number of days to add (positive) or subtract (negative)", value=0, step=1)
            direction = "after" if days_offset >= 0 else "before"

            new_date = base_date + datetime.timedelta(days=days_offset)
            st.write(f"{abs(days_offset)} day(s) {direction} {base_date.strftime('%m/%d/%y')} is {new_date.strftime('%m/%d/%y')}")

    # -----------------------------------
    # 5) HTML Processor Tab
    # -----------------------------------
    with tabs[4]:
        st.markdown("""
        ### 🏷️ HTML Processor

        **This tool allows you to:**
        - Upload an HTML file.
        - Extract text from specific tags (`<h1>...<h6>`, `<p>`, `<a>`, `<ul>`, `<li>`).
        - Download the extracted text as a `.txt` file.

        **How to Use:**
        1. Upload an `.html` file.
        2. The text from the specified tags will be extracted in order.
        3. Click "Download Processed Text" to save as a `.txt` file.
        """)
        uploaded_html_file = st.file_uploader("Upload HTML", type=["html"])
        if uploaded_html_file is not None:
            try:
                html_content = uploaded_html_file.read().decode('utf-8')
                soup = BeautifulSoup(html_content, 'html.parser')
            except Exception as e:
                st.error(f"Error reading the HTML file: {e}")
                st.stop()

            # Extract text from tags
            extracted_lines = []
            for tag in soup.find_all(['h1','h2','h3','h4','h5','h6','p','a','ul','li']):
                extracted_lines.append(f"<{tag.name}> {tag.get_text(strip=True)}")

            # Join extracted lines into a single string
            processed_text = "\n".join(extracted_lines)

            # Display a preview of the processed text
            st.markdown("#### Extracted Text Preview:")
            st.text(processed_text)

            # Download button for the processed text
            st.download_button(
                label="Download Processed Text",
                data=processed_text,
                file_name="web_text_output.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()