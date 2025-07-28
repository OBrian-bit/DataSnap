
# DataSnap

DataSnap is a Flask-based web application for uploading, analyzing, and cleaning CSV, TXT, and Excel files. It provides a powerful UI with live editing, data quality metrics, and features like anomaly detection, fuzzy matching, undo/redo, and filter recipes.

---

## **Features**
- Upload CSV, TXT, or Excel files and analyze them instantly.
- Automatic name splitting, phone number formatting, and date standardization.
- Data quality metrics including missing values, duplicates, and anomalies.
- Interactive filters (remove duplicates, find & replace, fill missing).
- Undo/redo for transformations.
- Export cleaned data as CSV/XLSX.
- Fuzzy matching and anomaly detection (Isolation Forest).
- Profile reports (via `ydata-profiling`).

---

## **Requirements**
- Python 3.9+
- Flask
- Pandas
- Flask-SocketIO
- TheFuzz (for fuzzy matching)
- scikit-learn (for anomaly detection)
- ydata-profiling (for data profiling)
- chardet (for encoding detection)

---

## **Installation**

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/OBrian-bit/DataSnap.git
   cd DataSnap


2. **Create a Virtual Environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate     # For Linux/Mac
   venv\Scripts\activate        # For Windows
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   If there is no `requirements.txt`, install manually:

   ```bash
   pip install flask pandas flask-socketio thefuzz scikit-learn ydata-profiling chardet
   ```

4. **Create Uploads Folder:**

   ```bash
   mkdir uploads
   ```

---

## **Running the Application**

1. **Start the Flask App:**

   ```bash
   python app.py
   ```

2. **Open in Browser:**
   Go to [http://127.0.0.1:5000](http://127.0.0.1:5000).

---

## **File Structure**

```
.
├── app.py            # Flask backend
├── templates/
│   ├── index.html    # Main UI
│   └── nav.html      # Navigation template
└── uploads/          # Uploaded files
```

---

## **Usage**

1. Select a group and upload your dataset (CSV/TXT/Excel).
2. Analyze the dataset and view the data quality charts.
3. Apply filters or transformations using the interactive UI.
4. Export the cleaned dataset as CSV or Excel.

---

## **License**

This project is open-source and available under the MIT License.

```

---

Would you like me to **create a `requirements.txt` file** for you based on your `app.py`?
```

---

Would you like me to **generate a `requirements.txt` file** automatically for your `app.py` so you can commit it alongside this README?
```
