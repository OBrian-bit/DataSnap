
# 📊 Modern File Analyzer

A sleek, browser-based file analysis app built with **Flask** and **Pandas**. Upload a CSV or Excel file and get instant insights—including cleaned name and phone number fields, missing value counts, duplicate detection, and a live data preview.

---

## 🚀 Features

- 📂 Upload CSV or Excel files
- 🔍 Automatically detects and formats:
  - Name columns (`First`, `Middle`, `Last`)
  - Phone numbers (standard U.S. formats)
- 📉 Displays metadata like:
  - File size
  - Number of rows & columns
  - Missing & duplicate values
  - Column names
- 🖥️ Beautiful, responsive UI using modern HTML & CSS
- ⚡ Fast in-browser preview of your dataset

---

## 🛠️ Technologies Used

- **Python 3**
- **Flask**
- **Pandas**
- **HTML5 + CSS3 (no external frameworks)**
- **Jinja2 Templating**

---

## 🧪 How to Use

1. **Clone this repository:**

   ```bash
   git clone https://github.com/OBrian-bit/DataSnap.git
   cd DataSnap


2. **Install dependencies:**

   ```bash
   pip install flask pandas openpyxl
   ```

3. **Run the app:**

   ```bash
   python app.py
   ```

4. **Open your browser** and go to `http://127.0.0.1:5000`

---

## 📁 File Structure

```
.
├── app.py             # Main Flask backend
├── index.html         # User interface with upload + results
├── uploads/           # Temporary folder for uploaded files (auto-created)
```

---

## 🧼 Data Cleaning Logic

* **Name Formatting:** Detects common name fields and splits them into First, Middle, and Last Name.
* **Phone Numbers:** Cleans and formats 10- or 11-digit phone numbers with optional country code (e.g., `(123) 456-7890`).
* **Statistics:** Computes dataset metadata like missing values, duplicates, and structural info.

---

## 🔒 Security Note

This project uses a demo secret key and processes files locally. For production use:

* Replace the `SECRET_KEY` in `app.py`
* Sanitize file inputs
* Deploy with HTTPS and server hardening

---


