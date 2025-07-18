import os
import re # Import the regular expression module
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, session, jsonify, Response, flash
from werkzeug.utils import secure_filename
import io
import chardet # Requires: pip install chardet

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'txt'} # Added 'txt'

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

def format_date_columns(df):
    """Identifies and formats common date columns to YYYY-MM-DD."""
    potential_date_cols = ['date', 'timestamp', 'created_at', 'updated_at', 'order_date', 'start_date', 'end_date', 'dob', 'date of birth']
    for col in df.columns:
        if col.lower().replace('_', ' ') in potential_date_cols:
            print(f"Found and formatting date column: '{col}'")
            # Convert to datetime, coercing errors to NaT (Not a Time)
            original_dtype = str(df[col].dtype)
            if 'datetime' not in original_dtype:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                # Format to string, NaT will become NaN which we can fill
                df[col] = df[col].dt.strftime('%Y-%m-%d').fillna('')
    return df


def _apply_filters_to_df(df, filters):
    """Applies a list of filter dictionaries to a DataFrame."""
    if not filters:
        return df
    
    df_filtered = df.copy()
    for f in filters:
        try:
            filter_type = f.get('type')
            if filter_type == 'remove_duplicates':
                df_filtered.drop_duplicates(inplace=True)
            
            elif filter_type == 'remove_junk':
                columns = f.get('columns', [])
                junk_values = [v for v in f.get('values', []) if v] # Ensure no empty strings
                if not junk_values: continue

                # Use a temporary DataFrame for the boolean mask to avoid chain assignment warnings
                mask = pd.DataFrame(False, index=df_filtered.index, columns=df_filtered.columns)

                if columns == "ALL":
                    # For each column, check if its values are in junk_values
                    for col in df_filtered.columns:
                        mask[col] = df_filtered[col].astype(str).isin(junk_values)
                    # Keep rows where NO cell contains a junk value
                    df_filtered = df_filtered[~mask.any(axis=1)]

                elif columns:
                    valid_cols = [c for c in columns if c in df_filtered.columns]
                    if not valid_cols: continue
                    # For specified columns, check if values are in junk_values
                    for col in valid_cols:
                         mask[col] = df_filtered[col].astype(str).isin(junk_values)
                    # Keep rows where NO specified cell contains a junk value
                    df_filtered = df_filtered[~mask[valid_cols].any(axis=1)]
                    
        except Exception as e:
            print(f"Error applying filter {f}: {e}")
            # Potentially skip this filter and continue
            continue
    return df_filtered


# --- Main Application Route ---
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if 'groups' not in session: session['groups'] = {}

    if request.method == 'POST':
        group_id = request.form.get('group_id')
        if not group_id:
            flash("Please select a group before uploading.", 'error')
            return redirect(url_for('upload_file'))
        
        if 'file' not in request.files:
            flash("No file part in the request.", 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash("No file selected for uploading.", 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # --- Encoding detection ---
                with open(filepath, 'rb') as f:
                    raw_data = f.read(100000)
                    result = chardet.detect(raw_data)
                    encoding = result['encoding'] if result['confidence'] > 0.7 else 'utf-8'
                
                # --- File Reading ---
                ext = filename.rsplit('.', 1)[1].lower()
                if ext in ['csv', 'txt']:
                    df = pd.read_csv(filepath, encoding=encoding)
                else:
                    df = pd.read_excel(filepath)

                # --- Processing & Logging Transformations ---
                transformations = [f"Detected file encoding as '{encoding}'."]
                original_cols = set(df.columns)
                
                df = process_name_column(df)
                new_cols = set(df.columns)
                if new_cols != original_cols:
                    transformations.append("Processed and split name column(s).")

                df = format_phone_columns(df)
                transformations.append("Formatted phone numbers to (xxx) xxx-xxxx.")
                
                df = format_date_columns(df)
                transformations.append("Standardized date columns to YYYY-MM-DD.")
                
                column_types = {k: str(v) for k, v in df.dtypes.to_dict().items()}
                
                total_rows = df.shape[0]
                duplicate_rows = int(df.duplicated().sum())
                unique_rows = total_rows - duplicate_rows
                total_cells = df.shape[0] * df.shape[1]
                missing_cells = int(df.isnull().sum().sum())
                valid_cells = total_cells - missing_cells
                
                preview_df = df.head()

                results = {
                    "display_name": group_id.replace('_', ' ').title(),
                    "filename": filename, 
                    "filesize_kb": round(os.path.getsize(filepath) / 1024, 2), 
                    "rows": total_rows, "columns": df.shape[1],
                    "encoding": encoding,
                    "missing_values": missing_cells, 
                    "duplicate_rows": duplicate_rows,
                    "empty_rows": int(df.isnull().all(axis=1).sum()), 
                    "preview": {
                        "columns": preview_df.columns.tolist(),
                        "data": preview_df.fillna('').values.tolist()
                    },
                    "column_types": column_types,
                    "transformations": transformations,
                    "quality_stats": {
                        "total_rows": total_rows, "unique_rows": unique_rows,
                        "duplicate_rows": duplicate_rows, "total_cells": total_cells,
                        "valid_cells": valid_cells, "missing_cells": missing_cells
                    },
                    "filters": [] # NEW: Add empty list for filters
                }
                session['groups'][group_id] = results
                session.modified = True
                flash(f"Successfully processed '{filename}' for group '{results['display_name']}'.", 'success')
                return redirect(url_for('upload_file'))
            except Exception as e:
                flash(f"Error processing file: {e}", 'error')
                return redirect(url_for('upload_file'))
        else:
            flash(f"Unsupported file type. Please upload a .csv, .xlsx, .xls, or .txt file.", 'error')
            return redirect(url_for('upload_file'))

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

@app.route('/apply-filters/<group_id>', methods=['POST'])
def apply_filters(group_id):
    if 'groups' not in session or group_id not in session['groups']:
        return jsonify({'status': 'error', 'message': 'Group not found.'}), 404

    try:
        filters = request.get_json().get('filters', [])
        
        # --- Update session with the new filters ---
        session['groups'][group_id]['filters'] = filters
        session.modified = True

        # --- Re-run the entire pipeline on the original file ---
        group_data = session['groups'][group_id]
        filename = group_data['filename']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        encoding = group_data.get('encoding', 'utf-8')

        ext = filename.rsplit('.', 1)[1].lower()
        if ext in ['csv', 'txt']:
            df = pd.read_csv(filepath, encoding=encoding)
        else:
            df = pd.read_excel(filepath)

        # Apply initial transformations first
        df = process_name_column(df)
        df = format_phone_columns(df)
        df = format_date_columns(df)
        
        # Apply the user-defined filters
        df_filtered = _apply_filters_to_df(df, filters)

        # --- Recalculate stats for the filtered data ---
        total_rows = df_filtered.shape[0]
        duplicate_rows = int(df_filtered.duplicated().sum())
        unique_rows = total_rows - duplicate_rows
        total_cells = df_filtered.shape[0] * df_filtered.shape[1]
        missing_cells = int(df_filtered.isnull().sum().sum())
        valid_cells = total_cells - missing_cells
        preview_df = df_filtered.head()

        response_data = {
            "status": "success",
            "rows": total_rows, "columns": df_filtered.shape[1],
            "missing_values": missing_cells, 
            "duplicate_rows": duplicate_rows,
            "preview": {
                "columns": preview_df.columns.tolist(),
                "data": preview_df.fillna('').values.tolist()
            },
            "quality_stats": {
                "total_rows": total_rows, "unique_rows": unique_rows,
                "duplicate_rows": duplicate_rows, "total_cells": total_cells,
                "valid_cells": valid_cells, "missing_cells": missing_cells
            }
        }
        return jsonify(response_data)

    except Exception as e:
        print(f"Error applying filters: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# --- Export Route ---
# Requires: pip install xlsxwriter
@app.route('/export/<group_id>/<file_format>')
def export_file(group_id, file_format):
    if 'groups' not in session or group_id not in session['groups']:
        return "Group not found.", 404

    group_data = session['groups'][group_id]
    filename = group_data['filename']
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    encoding = group_data.get('encoding', 'utf-8')
    filters = group_data.get('filters', []) # Get stored filters

    if not os.path.exists(filepath):
        return "Original file not found on server.", 404

    try:
        # Re-create the cleaned dataframe by re-running the same pipeline
        ext = filename.rsplit('.', 1)[1].lower()
        if ext in ['csv', 'txt']:
            df = pd.read_csv(filepath, encoding=encoding)
        else:
            df = pd.read_excel(filepath)

        df = process_name_column(df)
        df = format_phone_columns(df)
        df = format_date_columns(df)
        
        # NEW: Apply filters before exporting
        df = _apply_filters_to_df(df, filters)

        # Prepare export
        output_filename = f"{secure_filename(group_id)}_cleaned.{file_format}"
        if file_format == 'csv':
            buffer = io.StringIO()
            df.to_csv(buffer, index=False, encoding='utf-8')
            buffer.seek(0)
            return Response(
                buffer.getvalue(),
                mimetype='text/csv',
                headers={"Content-Disposition": f"attachment;filename={output_filename}"}
            )
        elif file_format == 'xlsx':
            buffer = io.BytesIO()
            df.to_excel(buffer, index=False, engine='xlsxwriter')
            buffer.seek(0)
            return Response(
                buffer.getvalue(),
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                headers={"Content-Disposition": f"attachment;filename={output_filename}"}
            )
        elif file_format == 'json':
            buffer = io.StringIO()
            # orient='records' creates a list of dicts, which is a common JSON format.
            df.to_json(buffer, orient='records', indent=4, date_format='iso')
            buffer.seek(0)
            return Response(
                buffer.getvalue(),
                mimetype='application/json',
                headers={"Content-Disposition": f"attachment;filename={output_filename}"}
            )
        else:
            return "Unsupported format", 400

    except Exception as e:
        print(f"Error exporting file: {e}")
        return f"Could not export file due to an error: {e}", 500


# --- Application Entry Point ---
if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER): os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)