# ==============================================================================
# app.py - Main Flask Application for the Modern File Analyzer
# ==============================================================================

import os
import re # Import the regular expression module
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

# --- Flask App Initialization ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'a-truly-secret-key-that-should-be-changed'

# --- Helper Functions ---

def allowed_file(filename):
    """Checks if a filename has one of the allowed extensions."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_name_column(df):
    """Intelligently finds and processes name columns. Handles multiple common formats."""
    potential_name_cols = ['name', 'full name', 'fullname', 'contact name', 'artist name', 'author', 'player']
    potential_middle_cols = ['middle name', 'middlename', 'middle', 'middle initial']
    name_col_found, middle_col_found = None, None
    df_cols = list(df.columns)
    for col in df_cols:
        col_lower = col.lower().replace('_', ' ')
        if col_lower in potential_name_cols: name_col_found = col
        if col_lower in potential_middle_cols: middle_col_found = col
    if name_col_found:
        try:
            df[name_col_found] = df[name_col_found].astype(str).fillna('')
            if df[name_col_found].str.strip().eq('').all(): return df
            first_valid_name = df[name_col_found][df[name_col_found] != ''].iloc[0]
            if ',' in first_valid_name:
                name_parts = df[name_col_found].str.split(',', n=1, expand=True)
                df['Last Name'] = name_parts[0].str.strip()
                first_and_middle_part = name_parts.get(1, '').str.strip()
                first_middle_split = first_and_middle_part.str.split(' ', n=1, expand=True)
                df['First Name'] = first_middle_split[0]
                parsed_middle_name = first_middle_split.get(1, pd.Series(dtype=str)).fillna('')
                df['Middle Name'] = parsed_middle_name
                if middle_col_found:
                    separate_middle = df[middle_col_found].astype(str).fillna('')
                    df['Middle Name'] = df['Middle Name'].mask(df['Middle Name'] == '', separate_middle)
            else:
                name_parts = df[name_col_found].str.split()
                df['First Name'] = name_parts.str[0]
                df['Middle Name'] = name_parts.apply(lambda x: ' '.join(x[1:-1]) if len(x) > 2 else '')
                df['Last Name'] = name_parts.apply(lambda x: x[-1] if len(x) > 1 else '')
            cols_to_drop = [name_col_found]
            if middle_col_found and middle_col_found in df.columns: cols_to_drop.append(middle_col_found)
            original_cols = [c for c in df.columns if c not in cols_to_drop and c not in ['First Name', 'Middle Name', 'Last Name']]
            new_col_order = ['First Name', 'Middle Name', 'Last Name'] + original_cols
            return df[new_col_order]
        except Exception as e:
            print(f"Could not process name column due to error: {e}")
            return df
    return df

# ==============================================================================
# REVISED: Helper function to format a single phone number, now handles numeric types
# ==============================================================================
def _format_phone_number(phone):
    """
    Helper function to format a single phone number string or number.
    It now correctly handles numbers that were read as floats (e.g., 3365145915.0).
    """
    # 1. Handle missing values (like NaN) first, return them as is.
    if pd.isna(phone):
        return phone
    
    # 2. Convert the input to a string to handle floats, integers, and text.
    phone_str = str(phone)
    
    # 3. If the string representation ends with '.0', slice it off.
    #    This specifically fixes the float-to-string conversion issue.
    if phone_str.endswith('.0'):
        phone_str = phone_str[:-2]
    
    # 4. Use regex to strip all remaining non-digit characters.
    digits = re.sub(r'\D', '', phone_str)
    
    # 5. If we have exactly 10 digits, format it.
    if len(digits) == 10:
        return f"({digits[0:3]}) {digits[3:6]}-{digits[6:10]}"
    # 6. If we have 11 digits and it starts with '1', strip '1' and format.
    elif len(digits) == 11 and digits.startswith('1'):
        return f"({digits[1:4]}) {digits[4:7]}-{digits[7:11]}"
    
    # 7. Fallback: If it's not a standard format, return the original value.
    #    This prevents messing up international numbers or numbers with extensions.
    return phone

def format_phone_columns(df):
    """Finds columns that look like phone numbers and formats them."""
    potential_phone_cols = ['phone', 'phone number', 'telephone', 'tel', 'contact', 'contact number', 'mobile']
    for col in df.columns:
        if col.lower().replace('_', ' ') in potential_phone_cols:
            print(f"Found and formatting phone column: '{col}'")
            df[col] = df[col].apply(_format_phone_number)
    return df


# --- Main Application Route ---
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files: return redirect(request.url)
        file = request.files['file']
        if file.filename == '': return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            try:
                if filename.endswith('.csv'):
                    df = pd.read_csv(filepath)
                else:
                    df = pd.read_excel(filepath)
                # --- Run All Data Transformations Sequentially ---
                df = process_name_column(df)
                df = format_phone_columns(df) # Call the phone formatter
                # --- Perform All Data Calculations ---
                file_size = os.path.getsize(filepath)
                num_rows, num_cols = df.shape
                column_names = df.columns.tolist()
                missing_values = int(df.isnull().sum().sum())
                duplicate_rows = int(df.duplicated().sum())
                empty_rows = int(df.isnull().all(axis=1).sum())
                data_preview = df.head().to_html(classes='data-table', index=False, border=0)
                # --- Assemble all results ---
                results = {"filename": filename, "filesize_kb": round(file_size / 1024, 2), "rows": num_rows, "columns": num_cols, "column_names": column_names, "missing_values": missing_values, "duplicate_rows": duplicate_rows, "empty_rows": empty_rows, "preview": data_preview}
                return render_template('index.html', results=results)
            except Exception as e:
                error_message = f"Error processing file: {e}"
                return render_template('index.html', error=error_message)
    return render_template('index.html', results=None)


# --- Application Entry Point ---
if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER): os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)