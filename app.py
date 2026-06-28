import gradio as gr
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK resources saat startup
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ── Load model & vectorizer ──────────────────────────────────────────────────
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
stemmer = PorterStemmer()

# ── Preprocessing (HARUS sama persis dengan saat training) ───────────────────
def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)        # hapus URL
    text = re.sub(r'\d+', '', text)                    # hapus angka
    text = ''.join([c for c in text if c not in string.punctuation])  # hapus tanda baca

    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [w for w in words if w not in stop_words]  # hapus stopword
    words = [stemmer.stem(w) for w in words]            # stemming

    return ' '.join(words)

# ── Fungsi inferensi ─────────────────────────────────────────────────────────
def predict(title: str, content: str) -> tuple[str, str]:
    if not title.strip() and not content.strip():
        return "⚠️ Mohon isi judul atau isi berita terlebih dahulu.", ""

    combined = title.strip() + " " + content.strip()
    cleaned  = preprocess_text(combined)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]

    if prediction == 1:
        label    = "✅ REAL NEWS"
        detail   = "Model mengklasifikasikan berita ini sebagai **berita nyata** berdasarkan pola teks yang dipelajari dari dataset ISOT."
    else:
        label    = "🚨 FAKE NEWS"
        detail   = "Model mengklasifikasikan berita ini sebagai **berita palsu** berdasarkan pola teks yang dipelajari dari dataset ISOT."

    return label, detail

# ── Contoh input ─────────────────────────────────────────────────────────────
examples = [
    [
        "WASHINGTON (Reuters) - Senate passes bipartisan infrastructure bill",
        "The U.S. Senate on Tuesday passed a bipartisan infrastructure bill after months of negotiations between the two parties."
    ],
    [
        "Scientists Confirm Earth Is Flat, NASA Admits Cover-Up",
        "Anonymous sources claim NASA has been hiding the truth for decades. Share before this gets deleted!"
    ],
    [
        "Federal Reserve Raises Interest Rates",
        "The Federal Reserve announced an interest rate decision following its policy meeting on Wednesday to combat inflation."
    ],
    [
        "You Won't Believe What This Celebrity Did Last Night",
        "Shocking footage leaked online. The government doesn't want you to see this. Forward to everyone you know!"
    ],
]

# ── UI Gradio ─────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="Fake News Detector",
    theme=gr.themes.Default(
        primary_hue="blue",
        neutral_hue="slate",
    ),
    css="""
        .result-box { font-size: 1.4rem; font-weight: 700; text-align: center; padding: 16px; border-radius: 10px; }
        .detail-box { font-size: 0.95rem; }
        footer { display: none !important; }
    """
) as demo:

    # Header
    gr.Markdown(
        """
        # 🔍 Fake News Detector
        Deteksi berita palsu menggunakan model **Support Vector Machine (LinearSVC)** + **TF-IDF**.  
        Dataset pelatihan: [ISOT Fake News Dataset](https://www.kaggle.com/datasets/csmalarkodi/isot-fake-news-dataset) — 44.898 artikel berita.
        """
    )

    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=3):
            title_input = gr.Textbox(
                label="Judul Berita",
                placeholder="Masukkan judul berita di sini...",
                lines=2,
            )
            content_input = gr.Textbox(
                label="Isi Berita",
                placeholder="Masukkan isi/body berita di sini...",
                lines=8,
            )

            with gr.Row():
                clear_btn  = gr.Button("🗑️ Bersihkan", variant="secondary")
                submit_btn = gr.Button("🔎 Deteksi Sekarang", variant="primary", scale=2)

        with gr.Column(scale=2):
            result_label  = gr.Textbox(
                label="Hasil Deteksi",
                interactive=False,
                elem_classes=["result-box"],
            )
            result_detail = gr.Markdown(
                label="Penjelasan",
                elem_classes=["detail-box"],
            )

            gr.Markdown(
                """
                ### ℹ️ Catatan
                - Model dilatih khusus untuk teks **berbahasa Inggris**
                - Hasil deteksi bersifat prediksi — selalu verifikasi ke sumber terpercaya
                - Model: **LinearSVC** | Fitur: **TF-IDF** (max 5.000 fitur)
                """
            )

    gr.Markdown("---")

    # Contoh
    gr.Examples(
        examples=examples,
        inputs=[title_input, content_input],
        label="📋 Contoh Berita (klik untuk mencoba)",
        examples_per_page=4,
    )

    # Event handlers
    submit_btn.click(
        fn=predict,
        inputs=[title_input, content_input],
        outputs=[result_label, result_detail],
    )
    clear_btn.click(
        fn=lambda: ("", "", "", ""),
        inputs=[],
        outputs=[title_input, content_input, result_label, result_detail],
    )

if __name__ == "__main__":
    demo.launch()
