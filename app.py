import os
import re # Import the regular expression module
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, session, jsonify
from werkzeug.utils import secure_filename

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

# --- Flask App Initialization ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'a-truly-secret-key-that-should-be-changed' # Essential for session management

# --- Helper Functions ---

def slugify(value):
    """
    Converts a string to a 'slug'.
    Now more robust: prevents empty slugs and trims underscores.
    """
    value = str(value).strip().lower()
    value = re.sub(r'[^\w\s-]', '', value) # remove non-alphanumeric
    value = re.sub(r'[-\s]+', '_', value).strip('_') # replace space/hyphen with _, then trim leading/trailing _
    if not value: # Handle cases where slug becomes empty (e.g., from "!!!")
        return "untitled_group"
    return value


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

def _format_phone_number(phone):
    if pd.isna(phone): return phone
    phone_str = str(phone)
    if phone_str.endswith('.0'): phone_str = phone_str[:-2]
    digits = re.sub(r'\D', '', phone_str)
    if len(digits) == 10: return f"({digits[0:3]}) {digits[3:6]}-{digits[6:10]}"
    elif len(digits) == 11 and digits.startswith('1'): return f"({digits[1:4]}) {digits[4:7]}-{digits[7:11]}"
    return phone

def format_phone_columns(df):
    potential_phone_cols = ['phone', 'phone number', 'telephone', 'tel', 'contact', 'contact number', 'mobile']
    for col in df.columns:
        if col.lower().replace('_', ' ') in potential_phone_cols:
            print(f"Found and formatting phone column: '{col}'")
            df[col] = df[col].apply(_format_phone_number)
    return df


# --- Main Application Route ---
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if 'groups' not in session: session['groups'] = {}

    if request.method == 'POST':
        group_id = request.form.get('group_id')
        if not group_id: return render_template('index.html', error="Please select a group.", groups=session.get('groups', {}))
        if 'file' not in request.files: return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '': return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            try:
                df = pd.read_csv(filepath) if filename.endswith('.csv') else pd.read_excel(filepath)
                df = process_name_column(df)
                df = format_phone_columns(df)
                
                results = {
                    "display_name": group_id.replace('_', ' ').title(),
                    "filename": filename, 
                    "filesize_kb": round(os.path.getsize(filepath) / 1024, 2), 
                    "rows": df.shape[0], "columns": df.shape[1], 
                    "column_names": df.columns.tolist(), 
                    "missing_values": int(df.isnull().sum().sum()), 
                    "duplicate_rows": int(df.duplicated().sum()), 
                    "empty_rows": int(df.isnull().all(axis=1).sum()), 
                    "preview": df.head().to_html(classes='data-table', index=False, border=0)
                }
                session['groups'][group_id] = results
                session.modified = True
                return render_template('index.html', groups=session['groups'])
            except Exception as e:
                return render_template('index.html', error=f"Error processing file: {e}", groups=session.get('groups', {}))
        else:
            return render_template('index.html', error=f"Unsupported file type. Please upload a .csv, .xlsx, or .xls file.", groups=session.get('groups', {}))

    return render_template('index.html', groups=session.get('groups', {}))

# --- API Routes ---
@app.route('/rename-group', methods=['POST'])
def rename_group():
    if 'groups' not in session: return jsonify({'status': 'error', 'message': 'No groups in session.'}), 400
    data = request.get_json()
    old_id, new_display_name = data.get('old_id'), data.get('new_name')
    if not all([old_id, new_display_name]): return jsonify({'status': 'error', 'message': 'Missing data.'}), 400
    if old_id not in session['groups']: return jsonify({'status': 'error', 'message': 'Group not found.'}), 404

    new_id = slugify(new_display_name)
    if new_id == old_id:
        session['groups'][old_id]['display_name'] = new_display_name
        session.modified = True
        return jsonify({'status': 'success', 'new_id': old_id, 'new_display_name': new_display_name})
    if new_id in session['groups']: return jsonify({'status': 'error', 'message': 'New name already exists.'}), 409
    
    group_data = session['groups'].pop(old_id)
    group_data['display_name'] = new_display_name
    session['groups'][new_id] = group_data
    session.modified = True
    return jsonify({'status': 'success', 'new_id': new_id, 'new_display_name': new_display_name})

@app.route('/delete-group', methods=['POST'])
def delete_group():
    if 'groups' not in session: return jsonify({'status': 'error', 'message': 'No groups in session.'}), 400
    data = request.get_json()
    group_id = data.get('group_id')
    if not group_id: return jsonify({'status': 'error', 'message': 'Missing group_id.'}), 400
    
    if group_id in session['groups']:
        session['groups'].pop(group_id)
        session.modified = True
        return jsonify({'status': 'success', 'message': 'Group deleted.'})
    else:
        return jsonify({'status': 'error', 'message': 'Group not found.'}), 404

# --- Application Entry Point ---
if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER): os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)