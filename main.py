import streamlit as st
from docx import Document
from io import BytesIO
import hashlib
import re
import subprocess
from collections import Counter
import pandas as pd
import nltk
import base64
import requests
import os
import datetime
from urllib.parse import urlparse
from nltk.tokenize import wordpunct_tokenize
from nltk.chat.util import Chat, reflections  # Not used, but kept for completeness
from PIL import Image, UnidentifiedImageError
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
    for _, url in get_paragraph_hyperlinks(paragraph):
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
        article_hyperlinks.extend(get_paragraph_hyperlinks(para))
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
    if keyword_processed.lower() in article_text_processed.lower():
        keyword_present = True
    else:
        keyword_present = False

    if not keyword_present:
        return 'not_found'

    matching_links = []
    for text, link in article_hyperlinks:
        text_processed = preprocess_text(text)
        if text_processed.lower() == keyword_processed.lower():
            matching_links.append(link)

    if not matching_links:
        return 'incorrect_url'

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
    import re
    word_list = re.findall(r'\b\w+\b', text.lower())
    from collections import Counter
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
            return [f'background-color: {color}; color: white'] * len(row)

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
    return [url.strip() for url in input_text.split(',') if url.strip()]

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
#                    EXIF Editor using exiftool (subprocess)
# -----------------------------------------------------------------------------
def can_exiftool_edit(file_ext):
    """
    Basic check for file types commonly editable by exiftool.
    You can expand this list if desired.
    """
    # exiftool can handle more than these, but here’s a small example set
    editable_exts = {'.jpg', '.jpeg', '.png', '.pdf', '.tif', '.tiff'}
    return file_ext.lower() in editable_exts

def write_exif_tags_exiftool(
    file_bytes,
    orig_filename,
    keywords="",
    description="",
    lat=None,
    lon=None,
    new_filename=""
):
    """
    Writes IPTC/XMP data via exiftool, if supported. Otherwise returns file unchanged.
    Returns (updated_file_bytes, final_filename).
    """
    _, extension = os.path.splitext(orig_filename)
    if not can_exiftool_edit(extension):
        # If exiftool can't handle this file type, skip editing and rename if desired
        final_name = new_filename.strip() if new_filename.strip() else orig_filename
        return file_bytes, final_name

    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as in_temp:
        in_temp.write(file_bytes)
        temp_input_path = in_temp.name

    cmd = ["exiftool", "-overwrite_original"]

    if keywords.strip():
        kw_list = [kw.strip() for kw in keywords.split(",") if kw.strip()]
        cmd.append("-Keywords=")  # Clear existing
        for kw in kw_list:
            cmd.append(f"-Keywords+={kw}")

    if description.strip():
        cmd.append(f"-Description={description}")
        cmd.append(f"-Caption-Abstract={description}")

    if lat is not None and lon is not None:
        cmd.append(f"-GPSLatitude={lat}")
        cmd.append(f"-GPSLongitude={lon}")
        lat_ref = "N" if lat >= 0 else "S"
        lon_ref = "E" if lon >= 0 else "W"
        cmd.append(f"-GPSLatitudeRef={lat_ref}")
        cmd.append(f"-GPSLongitudeRef={lon_ref}")

    cmd.append(temp_input_path)
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)

    with open(temp_input_path, "rb") as updated_file:
        updated_file_data = updated_file.read()

    os.remove(temp_input_path)

    # Decide final name
    if new_filename.strip():
        final_name = new_filename.strip()
    else:
        final_name = f"updated{extension}"  # Just "updated" + same extension

    return updated_file_data, final_name

# -----------------------------------------------------------------------------
#                           Main App
# -----------------------------------------------------------------------------
def main():
    st.title("📄 Sibi's Research Bin")

    # Create Tabs:
    tabs = st.tabs([
        "Keyword Checker",
        "Exif Editor",
        "URL Checker",
        "Date Calculator",
        "HTML Processor"
    ])

    # -----------------------------------
    # 1) Keyword Checker Tab
    # -----------------------------------
    with tabs[0]:
        st.markdown("""
        ### 📄 Keyword Checker for Word Documents

        **This tool allows you to:**
        - Check if each keyword is present in the uploaded Word document.
        - Verify if each keyword is correctly hyperlinked.
        - Detect the dialect (American, British, Canadian, Australian).
        """)
        uploaded_file = st.file_uploader("Upload .docx", type=["docx"])
        keywords_text = st.text_area("Enter Keywords and URLs", height=300)
        run_btn = st.button("Run Analysis")
        if run_btn:
            if not uploaded_file:
                st.error("Please upload a .docx file.")
            elif not keywords_text.strip():
                st.error("Please enter some keywords/URLs.")
            else:
                try:
                    doc = Document(BytesIO(uploaded_file.read()))
                except Exception as e:
                    st.error(f"Error reading the document: {e}")
                    st.stop()

                articles_keywords = parse_keywords_links(keywords_text)
                if not articles_keywords:
                    st.error("No valid keywords/URLs found.")
                    st.stop()

                doc_articles = parse_document_articles(doc)
                if not doc_articles:
                    st.error("No articles found.")
                    st.stop()

                if len(articles_keywords) != len(doc_articles):
                    st.warning("Number of keyword groups doesn’t match number of articles.")

                all_hyps = []
                num_images = len(doc.inline_shapes)
                final_data = []
                for idx, group in enumerate(articles_keywords):
                    if idx < len(doc_articles):
                        paras = doc_articles[idx]
                        art_text, art_hyps = extract_article_content(paras)
                        all_hyps.extend(art_hyps)

                        # check resource box
                        has_box, res_url = False, "N/A"
                        for p in reversed(paras):
                            if "resource box:" in p.text.lower():
                                has_box = True
                                potential_urls = extract_urls_from_paragraph(p)
                                if potential_urls:
                                    res_url = potential_urls[-1]
                                break

                        dial = analyze_dialect(art_text)
                        results_kw = []
                        for (kw, link) in group:
                            status = check_keyword_hyperlink(
                                kw,
                                link if link else "[No Hyperlink]",
                                art_hyps,
                                art_text
                            )
                            if status == 'correct':
                                symbol = '✔️ Correct'
                                color = 'green'
                                detail = (f'Keyword "<strong>{kw}</strong>" correctly linked.'
                                          if link else f'Keyword "<strong>{kw}</strong>" present with no hyperlink.')
                            elif status == 'incorrect_url':
                                symbol = '⚠️ Incorrect URL'
                                color = 'orange'
                                detail = f'Keyword "<strong>{kw}</strong>" present but linked incorrectly.'
                            else:
                                symbol = '❌ Not Found'
                                color = 'red'
                                detail = f'Keyword "<strong>{kw}</strong>" not found.'
                            results_kw.append({
                                'Status': symbol,
                                'Details': detail,
                                'Color': color
                            })

                        final_data.append({
                            'Article': f'Article #{idx+1}',
                            'Detected Dialect': dial,
                            'Resource Box': res_url if has_box else 'Missing or None',
                            'Keywords': results_kw
                        })
                    else:
                        final_data.append({
                            'Article': f'Article #{idx+1}',
                            'Detected Dialect': 'Not found',
                            'Resource Box': 'N/A',
                            'Keywords': [{
                                'Status': '❌ Not Found',
                                'Details': 'No article content.',
                                'Color': 'red'
                            }]
                        })
                rep_html = generate_html_report(final_data, num_images, all_hyps)
                st.markdown(rep_html, unsafe_allow_html=True)

    # -----------------------------------
    # 2) Exif Editor Tab
    # -----------------------------------
    with tabs[1]:
        st.markdown("""
        ### 🖼️ EXIF Editor (exiftool)

        Upload **any file** (not limited to JPEG/PNG). Only certain formats can actually be edited.
        """)

        if "exif_images" not in st.session_state:
            st.session_state.exif_images = {}
        if "selected_file_hash" not in st.session_state:
            st.session_state.selected_file_hash = None

        def file_md5(data):
            return hashlib.md5(data).hexdigest()

        # Accept any file type
        uploaded_files = st.file_uploader("Upload Files", type=None, accept_multiple_files=True)
        if uploaded_files:
            for upf in uploaded_files:
                content = upf.read()
                h = file_md5(content)
                if h not in st.session_state.exif_images:
                    st.session_state.exif_images[h] = {
                        "filename": upf.name,
                        "bytes": content
                    }
                else:
                    st.warning(f"Duplicate file skipped: {upf.name}")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Clear Gallery"):
                st.session_state.exif_images = {}
                st.session_state.selected_file_hash = None
        with c2:
            if st.button("Delete Selected"):
                if st.session_state.selected_file_hash in st.session_state.exif_images:
                    del st.session_state.exif_images[st.session_state.selected_file_hash]
                    st.session_state.selected_file_hash = None

        left_side, right_side = st.columns([2,1])
        with right_side:
            st.markdown("#### File Gallery")
            for hsh, info in st.session_state.exif_images.items():
                fn = info["filename"]
                data = info["bytes"]
                # Try to display as image
                try:
                    img = Image.open(BytesIO(data))
                    st.image(img, caption=fn, width=130)
                except UnidentifiedImageError:
                    # Non-image or unsupported image
                    st.write(f"{fn} (not displayed as image)")

                if st.button(f"Select {fn}", key=f"sel_{hsh}"):
                    st.session_state.selected_file_hash = hsh

        with left_side:
            st.markdown("#### Edit Metadata")
            sel_hash = st.session_state.selected_file_hash
            if sel_hash and sel_hash in st.session_state.exif_images:
                rec = st.session_state.exif_images[sel_hash]
                raw_bytes = rec["bytes"]
                orig_name = rec["filename"]
                st.write(f"Currently selected: **{orig_name}**")

                kw_in = st.text_input("Keywords (comma-separated)")
                desc_in = st.text_input("Description / Alt Text")
                lat_in = st.text_input("Latitude")
                lon_in = st.text_input("Longitude")
                newf_in = st.text_input("New Filename (optional)")

                try:
                    lat_val = float(lat_in) if lat_in.strip() else None
                    lon_val = float(lon_in) if lon_in.strip() else None
                except ValueError:
                    lat_val, lon_val = None, None
                    st.warning("Invalid lat/lon values.")
             # **Map Display Code Starts Here**
                if lat_val is not None and lon_val is not None:
                    st.markdown("#### Location Map")
                    map_data = pd.DataFrame({
                        'latitude': [lat_val],
                        'longitude': [lon_val]
                    })
                st.map(map_data)
                elif lat_in or lon_in:
                    st.warning("Please provide both valid latitude and longitude to display the map.")
            # **Map Display Code Ends Here**


                if st.button("Write EXIF Tags"):
                    updated, final_name = write_exif_tags_exiftool(
                        file_bytes=raw_bytes,
                        orig_filename=orig_name,
                        keywords=kw_in,
                        description=desc_in,
                        lat=lat_val,
                        lon=lon_val,
                        new_filename=newf_in
                    )
                    st.session_state.exif_images[sel_hash]["bytes"] = updated
                    st.session_state.exif_images[sel_hash]["filename"] = final_name
                    st.success(f"Successfully updated metadata. Final file name: {final_name}")

                # Download link
                dload_data = st.session_state.exif_images[sel_hash]["bytes"]
                dload_fname = st.session_state.exif_images[sel_hash]["filename"]
                b64file = base64.b64encode(dload_data).decode()
                mime_type = "application/octet-stream"
                dlink = f'<a href="data:{mime_type};base64,{b64file}" download="{dload_fname}">Download Updated File</a>'
                st.markdown(dlink, unsafe_allow_html=True)
            else:
                st.info("Select a file from the gallery on the right.")

    # -----------------------------------
    # 3) URL Checker Tab
    # -----------------------------------
    with tabs[2]:
        st.markdown("### 🔗 URL Checker")
        user_urls = st.text_area("Enter URLs (comma or newline separated)")
        check_btn = st.button("Check URLs")
        if check_btn:
            if not user_urls.strip():
                st.error("Please provide at least one URL.")
            else:
                df_clients = load_client_database_csv()
                if df_clients is None:
                    st.error("CSV not found at /home/databased.csv")
                else:
                    if 'url' not in df_clients.columns:
                        st.error("CSV must have a 'url' column.")
                    else:
                        splitted = parse_input_urls(user_urls)
                        if not splitted:
                            st.error("No valid URLs found.")
                        else:
                            results = []
                            for s in splitted:
                                norm = normalize_url(s)
                                match_row = df_clients[df_clients['url'].str.lower() == norm.lower()]
                                stat, err = check_url_online(norm)
                                row_data = {
                                    "URL": s,
                                    "Status": stat
                                }
                                if not match_row.empty:
                                    rdict = match_row.to_dict(orient='records')[0]
                                    for k, v in rdict.items():
                                        if k.lower() != 'url':
                                            row_data[k] = v
                                else:
                                    row_data["CSV Data"] = "N/A"
                                if err:
                                    row_data["Error"] = err
                                results.append(row_data)
                            out_df = pd.DataFrame(results)
                            col_order = ['URL','Status'] + [c for c in out_df.columns if c not in ['URL','Status']]
                            st.dataframe(out_df[col_order], use_container_width=True)

    # -----------------------------------
    # 4) Date Calculator Tab
    # -----------------------------------
    with tabs[3]:
        st.markdown("""
        ### 📅 Date Calculator
        - Compute day differences
        - Add or subtract days
        """)
        mode_sel = st.radio("Mode", ["Date Difference", "Date Arithmetic"])
        if mode_sel == "Date Difference":
            d1 = st.date_input("Start Date")
            d2 = st.date_input("End Date")
            if d1 and d2:
                delta = abs((d2 - d1).days)
                st.write(f"Days difference: {delta}")
        else:
            base_date = st.date_input("Base Date")
            day_offset = st.number_input("Days (+/-)", value=0, step=1)
            final_date = base_date + datetime.timedelta(days=day_offset)
            st.write(f"Result: {final_date}")

    # -----------------------------------
    # 5) HTML Processor Tab
    # -----------------------------------
    with tabs[4]:
        st.markdown("""
        ### 🏷️ HTML Processor
        Upload an HTML file to extract text from <h1>...<h6>, <p>, <a>, <ul>, <li>.
        """)
        up_html = st.file_uploader("Upload HTML File", type=["html"])
        if up_html:
            try:
                raw_html = up_html.read().decode("utf-8", errors="replace")
                soup = BeautifulSoup(raw_html, 'html.parser')
            except Exception as e:
                st.error(f"Failed reading HTML: {e}")
                st.stop()

            lines_found = []
            for t in soup.find_all(['h1','h2','h3','h4','h5','h6','p','a','ul','li']):
                txt = t.get_text(strip=True)
                lines_found.append(f"<{t.name}> {txt}")

            combined = "\n".join(lines_found)
            st.markdown("#### Preview:")
            st.text(combined)

            st.download_button(
                label="Download Extracted Text",
                data=combined,
                file_name="extracted_text.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()
