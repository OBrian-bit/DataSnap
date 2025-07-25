# 📊 DataSnap - Modern File Analyzer

A sleek, browser-based file analysis and cleaning app built with **Flask**, **Pandas**, and a dynamic **Tailwind CSS** frontend.

Upload a CSV, TXT, or Excel file and get instant insights. The app automatically detects and formats common data types, provides rich metadata, and allows you to apply interactive cleaning filters with a live preview—all without leaving your browser.

 <!-- It's highly recommended to add a screenshot of the app here! -->

---

## 🚀 Features

-   **Multi-Format Upload**: Supports CSV, TXT, XLSX, and XLS files.
-   **Group Management**: Work with multiple datasets in a single session, organized into groups.
-   **Automated Cleaning**:
    -   Intelligently parses and splits name columns (e.g., `Last, First M.` → `First`, `Middle`, `Last`).
    -   Standardizes phone numbers into a clean `(XXX) XXX-XXXX` format.
    -   Converts date-like columns into a consistent `YYYY-MM-DD` format.
-   **Interactive Filtering**:
    -   Apply cleaning rules like **"Remove Duplicates"** or **"Remove Rows with Junk Values"**.
    -   Build a stack of multiple filters and see the results instantly.
    -   Your filter workflow is saved per-group in your session.
-   **Rich Data Visualization**:
    -   Key statistics dashboard (rows, columns, missing values, duplicates).
    -   Doughnut charts for a quick overview of data quality.
    -   Live data preview table that updates as you apply filters.
-   **Modern UI**:
    -   A beautiful, responsive interface built with **Tailwind CSS**.
    -   Light and Dark mode support.
    -   Interactive sidebars and toast notifications for a smooth user experience.
-   **Export Cleaned Data**: Download your processed data as a CSV or XLSX file.

---

## 🛠️ Technologies Used

-   **Backend**: Python 3, Flask, Pandas
-   **Frontend**: HTML5, Tailwind CSS, Vanilla JavaScript, Chart.js
-   **Core Libraries**: `chardet` (encoding detection), `openpyxl` (Excel), `xlsxwriter` (Excel export)

---

## 🧪 How to Run Locally

Follow these steps to get the application running on your local machine.

#### 1. **Prerequisites**

-   Python 3.7 or newer
-   `pip` (Python package installer)

#### 2. **Clone the Repository**

```bash
git clone https://github.com/OBrian-bit/DataSnap.git
cd DataSnap
```

#### 3. **Set Up a Virtual Environment (Recommended)**

A virtual environment keeps your project's dependencies isolated.

-   **On macOS/Linux:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
-   **On Windows:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

#### 4. **Install Dependencies**

Install all the required Python libraries.

```bash
pip install Flask pandas openpyxl chardet xlsxwriter
```

#### 5. **Run the Application**

Execute the main Python script to start the Flask server.

```bash
python app.py
```

The terminal will show output indicating that the server is running, usually with a message like:
`* Running on http://127.0.0.1:5000`

#### 6. **Open in Your Browser**

Open your web browser and navigate to the local address:

**[http://127.0.0.1:5000](http://127.0.0.1:5000)**

You should now see the DataSnap application interface, ready for you to upload a file!

---

## 📁 File Structure

```
.
├── app.py             # Main Flask backend logic and API routes
├── index.html         # The complete user interface (HTML, CSS, JS)
├── uploads/           # Temporary folder for uploaded files (auto-created)
└── README.md          # You are here!
```

---

## 🧼 Data Cleaning & Processing Logic

-   **Initial Processing**: On upload, the app automatically detects file encoding and applies standard formatting for names, phones, and dates.
-   **Interactive Filtering**: The `/apply-filters` API endpoint receives filter rules from the frontend, re-processes the original dataset with these new rules, and returns updated statistics and a data preview.
-   **State Management**: All uploaded files and their applied filters are stored in the user's server-side session, allowing for a persistent workflow.
-   **Export**: The export functionality re-runs the entire cleaning pipeline (initial formatting + user filters) on the original file to ensure the final output is fully processed.

---

## 🔒 Security Note

This project is configured for local development and demonstration. For a production environment, please consider the following:

-   **Secret Key**: Replace the hardcoded `app.config['SECRET_KEY']` in `app.py` with a secure, environment-loaded variable.
-   **File Sanitization**: Implement stricter checks on uploaded file contents and metadata.
-   **Deployment**: Use a production-grade WSGI server (like Gunicorn or Waitress) instead of Flask's built-in development server.
-   **HTTPS**: Configure your deployment to use HTTPS to encrypt traffic.
```

