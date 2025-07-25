import os
import re
import pandas as pd
import json
from flask import Flask, request, render_template, redirect, url_for, session, jsonify, Response, flash, make_response
from werkzeug.utils import secure_filename
import io
import chardet # Requires: pip install chardet

# --- NEW: Import SocketIO and related functions ---
# Requires: pip install flask-socketio
from flask_socketio import SocketIO, emit

# --- NEW: Import for Fuzzy Matching ---
# Requires: pip install thefuzz python-Levenshtein
from thefuzz import fuzz

# --- NEW: Import for Anomaly Detection ---
# Requires: pip install scikit-learn
from sklearn.ensemble import IsolationForest

# --- NEW: Import for Profile Report ---
# Requires: pip install ydata-profiling
from ydata_profiling import ProfileReport

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'txt'}

# --- Flask App Initialization ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'a-truly-secret-key-that-should-be-changed' # Essential for session and SocketIO

# --- Initialize SocketIO ---
socketio = SocketIO(app)

# --- In-memory cache for active DataFrames ---
# This dictionary will store the current, processed state of the data for each group,
# enabling real-time edits and faster exports.
ACTIVE_DATAFRAMES = {}

# --- NEW: In-memory undo/redo stacks ---
UNDO_STACK: dict[str, list[pd.DataFrame]] = {}
REDO_STACK: dict[str, list[pd.DataFrame]] = {}


# --- Helper Functions ---

def push_undo(group_id):
    """Pushes the current state of a group's DataFrame to the undo stack."""
    if group_id not in ACTIVE_DATAFRAMES:
        return
    UNDO_STACK.setdefault(group_id, []).append(ACTIVE_DATAFRAMES[group_id].copy())
    # Cap history length to prevent excessive memory use
    if len(UNDO_STACK[group_id]) > 20:
        UNDO_STACK[group_id].pop(0)
    # A new action clears the redo stack, as the history has diverged
    REDO_STACK[group_id] = []

def _build_reset_payload(group_id):
    """Helper to build a full data payload for a group."""
    df = ACTIVE_DATAFRAMES[group_id]
    
    total_rows, duplicate_rows = df.shape[0], int(df.duplicated().sum())
    missing_cells = int(df.isnull().sum().sum())

    # Get transformation history and filter recipe from the session
    group_info = session.get('groups', {}).get(group_id, {})
    transformations = group_info.get('transformations', [])
    filters = group_info.get('filters', [])

    # --- MODIFIED FOR BAR CHART ---
    # Get original and empty counts from the session data stored on upload
    original = group_info.get('rows', total_rows)
    empty = group_info.get('empty_rows', 0)

    return {
        "status": "success",
        "group_id": group_id,
        "rows": total_rows,  # Cleaned count
        "columns": df.shape[1],
        "missing_values": missing_cells,
        "duplicate_rows": duplicate_rows,
        "preview": {"columns": df.head().columns.tolist(), "data": df.head().fillna('').values.tolist()},
        "quality_stats": {
            "total_rows": total_rows,
            "unique_rows": total_rows - duplicate_rows,
            "duplicate_rows": duplicate_rows,
            "total_cells": df.size,
            "valid_cells": df.size - missing_cells,
            "missing_cells": missing_cells
        },
        "transformations": transformations,
        "filters": filters,
        "undo_count": len(UNDO_STACK.get(group_id, [])),
        "redo_count": len(REDO_STACK.get(group_id, [])),
        # --- NEW KEYS FOR BAR CHART ---
        "original_rows": original,
        "empty_rows": empty
    }

def slugify(value):
    """
    Converts a string to a 'slug'.
    Now more robust: prevents empty slugs and trims underscores.
    """
    value = str(value).strip().lower()
    value = re.sub(r'[^\w\s-]', '', value)
    value = re.sub(r'[-\s]+', '_', value).strip('_')
    if not value:
        return "untitled_group"
    return value

def allowed_file(filename):
    """Checks if a filename has one of the allowed extensions."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
            df[col] = df[col].apply(_format_phone_number)
    return df

def format_date_columns(df):
    potential_date_cols = ['date', 'timestamp', 'created_at', 'updated_at', 'order_date', 'start_date', 'end_date', 'dob', 'date of birth']
    for col in df.columns:
        if col.lower().replace('_', ' ') in potential_date_cols:
            original_dtype = str(df[col].dtype)
            if 'datetime' not in original_dtype:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                df[col] = df[col].dt.strftime('%Y-%m-%d').fillna('')
    return df

def _apply_filters_to_df(df, filters):
    if not filters: return df
    df_filtered = df.copy()
    for f in filters:
        try:
            filter_type = f.get('type')
            if filter_type == 'remove_duplicates':
                df_filtered.drop_duplicates(inplace=True)
            elif filter_type == 'remove_junk':
                columns = f.get('columns', [])
                junk_values = [v for v in f.get('values', []) if v]
                if not junk_values: continue
                mask = pd.DataFrame(False, index=df_filtered.index, columns=df_filtered.columns)
                if columns == "ALL":
                    for col in df_filtered.columns:
                        mask[col] = df_filtered[col].astype(str).isin(junk_values)
                    df_filtered = df_filtered[~mask.any(axis=1)]
                elif columns:
                    valid_cols = [c for c in columns if c in df_filtered.columns]
                    if not valid_cols: continue
                    for col in valid_cols:
                         mask[col] = df_filtered[col].astype(str).isin(junk_values)
                    df_filtered = df_filtered[~mask[valid_cols].any(axis=1)]
            elif filter_type == 'find_replace': # NEW: Find & Replace Filter
                columns = f.get('columns', [])
                find_val = f.get('find')
                replace_val = f.get('replace', '')
                is_regex = f.get('is_regex', False)

                if not find_val or not columns:
                    continue

                target_cols = []
                if columns == "ALL":
                    # Only apply find/replace to string-like columns to avoid errors
                    target_cols = df_filtered.select_dtypes(include='object').columns.tolist()
                else:
                    target_cols = [c for c in columns if c in df_filtered.columns]

                for col in target_cols:
                    # Ensure column is string type for '.str' accessor to work reliably
                    df_filtered[col] = df_filtered[col].astype(str)
                    df_filtered[col] = df_filtered[col].str.replace(
                        str(find_val),
                        str(replace_val),
                        regex=is_regex
                    )
            elif filter_type == 'fill_missing':
                method = f.get('method')
                columns = f.get('columns', [])
                if not method or not columns:
                    continue

                target_cols = []
                if columns == "ALL":
                    target_cols = df_filtered.columns.tolist()
                elif columns == "ALL_NUMERIC":
                    target_cols = df_filtered.select_dtypes(include='number').columns.tolist()
                else: # It's a list of column names
                    target_cols = [c for c in columns if c in df_filtered.columns]

                for col in target_cols:
                    if df_filtered[col].notna().all(): continue # Skip if column is already full

                    if method == 'mean':
                        if pd.api.types.is_numeric_dtype(df_filtered[col]):
                            df_filtered[col].fillna(df_filtered[col].mean(), inplace=True)
                    elif method == 'median':
                        if pd.api.types.is_numeric_dtype(df_filtered[col]):
                            df_filtered[col].fillna(df_filtered[col].median(), inplace=True)
                    elif method == 'mode':
                        # Mode is safe for any data type. It returns a Series, so we take the first value.
                        if not df_filtered[col].mode().empty:
                            fill_value = df_filtered[col].mode()[0]
                            df_filtered[col].fillna(fill_value, inplace=True)

        except Exception as e:
            print(f"Error applying filter {f}: {e}")
            continue
    return df_filtered

def _apply_fuzzy_merge(df, column, threshold=90):
    """Groups similar values in a column and standardizes them."""
    col_series = df[column].astype(str).fillna('').str.strip()
    uniques = col_series[col_series != ''].unique().tolist()
    
    value_map = {}
    processed_values = set()

    for val in uniques:
        if val in processed_values:
            continue
        
        matches = [other for other in uniques if other not in processed_values and fuzz.ratio(val, other) >= threshold]
        
        if not matches:
            continue
            
        canonical_name = matches[0]
        for match in matches:
            value_map[match] = canonical_name
            processed_values.add(match)
            
    if not value_map:
        return df, "No similar values found to merge."

    original_series = df[column].copy()
    df[column] = df[column].map(value_map).fillna(original_series)
    
    merged_count = len(value_map) - len(set(value_map.values()))
    group_count = len(set(value_map.values()))
    
    if merged_count > 0:
        log_message = f"Merged {merged_count} variations into {group_count} groups in '{column}' (threshold: {threshold}%)."
    else:
        log_message = f"No values in '{column}' met the merge threshold of {threshold}%."
        
    return df, log_message


# --- Main Application Route ---
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if 'groups' not in session: 
        session['groups'] = {}

    if request.method == 'GET':
        groups_in_session = list(session.get('groups', {}).keys())
        stale_groups_found = False
        for group_id in groups_in_session:
            if group_id not in ACTIVE_DATAFRAMES:
                session['groups'].pop(group_id, None)
                stale_groups_found = True
        
        if stale_groups_found:
            session.modified = True
            flash("Some active file sessions were cleared due to a server restart. Please re-upload them if needed.", 'warning')

    if request.method == 'POST':
        group_id = request.form.get('group_id')
        if not group_id or 'file' not in request.files or request.files['file'].filename == '':
            flash("Please select a group and a file before uploading.", 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                with open(filepath, 'rb') as f:
                    raw_data = f.read(100000)
                    result = chardet.detect(raw_data)
                    encoding = result['encoding'] if result['confidence'] > 0.7 else 'utf-8'
                
                ext = filename.rsplit('.', 1)[1].lower()
                df = pd.read_csv(filepath, encoding=encoding) if ext in ['csv', 'txt'] else pd.read_excel(filepath)

                transformations = [f"Detected file encoding as '{encoding}'."]
                original_cols = set(df.columns)
                df = process_name_column(df)
                if set(df.columns) != original_cols: transformations.append("Processed and split name column(s).")
                df = format_phone_columns(df)
                transformations.append("Formatted phone numbers to (xxx) xxx-xxxx.")
                df = format_date_columns(df)
                transformations.append("Standardized date columns to YYYY-MM-DD.")
                
                total_rows = df.shape[0]
                duplicate_rows = int(df.duplicated().sum())
                
                results = {
                    "display_name": group_id.replace('_', ' ').title(), "filename": filename,
                    "filesize_kb": round(os.path.getsize(filepath) / 1024, 2), "rows": total_rows,
                    "columns": df.shape[1], "encoding": encoding, "missing_values": int(df.isnull().sum().sum()),
                    "duplicate_rows": duplicate_rows, "empty_rows": int(df.isnull().all(axis=1).sum()),
                    "preview": {
                        "columns": df.head().columns.tolist(),
                        "data": df.head().fillna('').values.tolist()
                    },
                    "column_types": {k: str(v) for k, v in df.dtypes.to_dict().items()},
                    "transformations": transformations,
                    "quality_stats": {
                        "total_rows": total_rows, "unique_rows": total_rows - duplicate_rows,
                        "duplicate_rows": duplicate_rows, "total_cells": df.size,
                        "valid_cells": df.size - int(df.isnull().sum().sum()), "missing_cells": int(df.isnull().sum().sum())
                    },
                    "filters": [],
                    "undo_count": 0, # NEW: Initial state
                    "redo_count": 0  # NEW: Initial state
                }
                
                ACTIVE_DATAFRAMES[group_id] = df.copy()
                session['groups'][group_id] = results
                session.modified = True
                flash(f"Successfully processed '{filename}' for group '{results['display_name']}'.", 'success')
            except Exception as e:
                flash(f"Error processing file: {e}", 'error')
            return redirect(url_for('upload_file'))
        else:
            flash("Unsupported file type.", 'error')
            return redirect(url_for('upload_file'))
            
    return render_template('index.html', groups=session.get('groups', {}))

# --- API & State Management Routes ---
@app.route('/rename-group', methods=['POST'])
def rename_group():
    data = request.get_json()
    old_id, new_display_name = data.get('old_id'), data.get('new_name')
    if not all([old_id, new_display_name]) or old_id not in session.get('groups', {}):
        return jsonify({'status': 'error', 'message': 'Invalid request.'}), 400
    
    new_id = slugify(new_display_name)
    if new_id in session['groups'] and new_id != old_id:
        return jsonify({'status': 'error', 'message': 'New name already exists.'}), 409
    
    group_data = session['groups'].pop(old_id)
    group_data['display_name'] = new_display_name
    session['groups'][new_id] = group_data
    if old_id in ACTIVE_DATAFRAMES:
        ACTIVE_DATAFRAMES[new_id] = ACTIVE_DATAFRAMES.pop(old_id)
    session.modified = True
    
    return jsonify({'status': 'success', 'new_id': new_id, 'new_display_name': new_display_name})

@app.route('/delete-group', methods=['POST'])
def delete_group():
    group_id = request.get_json().get('group_id')
    if group_id and group_id in session.get('groups', {}):
        session['groups'].pop(group_id)
        ACTIVE_DATAFRAMES.pop(group_id, None)
        UNDO_STACK.pop(group_id, None) # Clear undo/redo history
        REDO_STACK.pop(group_id, None)
        session.modified = True
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error', 'message': 'Group not found.'}), 404

@app.route('/apply-filters/<group_id>', methods=['POST'])
def apply_filters(group_id):
    if group_id not in session.get('groups', {}):
        return jsonify({'status': 'error', 'message': 'Group not found.'}), 404
    if group_id not in ACTIVE_DATAFRAMES:
         return jsonify({'status': 'error', 'message': 'Group data not found in active cache. Please re-upload file.'}), 404
    
    try:
        filters = request.get_json().get('filters', [])
        
        push_undo(group_id) # Save state before applying filter

        session['groups'][group_id]['filters'] = filters
        session.modified = True
        
        # NOTE: We use the *original* dataframe from before this filter was applied
        # This makes the "Reset" button work as expected, reverting to the last saved state.
        # To make filters stack, we would apply to the already-filtered df.
        # For this implementation, we re-apply ALL filters every time from a clean state.
        df_original = UNDO_STACK[group_id][-1] if UNDO_STACK.get(group_id) else ACTIVE_DATAFRAMES[group_id]
        df_filtered = _apply_filters_to_df(df_original, filters)
        ACTIVE_DATAFRAMES[group_id] = df_filtered.copy()
        
        payload = _build_reset_payload(group_id)
        socketio.emit('data_reset', payload)
        return jsonify(payload)

    except Exception as e:
        print(f"Error applying filters: {e}")
        # Attempt to roll back if something went wrong
        if UNDO_STACK.get(group_id):
            ACTIVE_DATAFRAMES[group_id] = UNDO_STACK[group_id].pop()
            REDO_STACK.pop(group_id, None) # Clear redo stack since the action failed
        return jsonify({'status': 'error', 'message': str(e)}), 500

# --- NEW: Undo/Redo Routes ---
@app.route('/undo/<group_id>', methods=['POST'])
def undo(group_id):
    if not UNDO_STACK.get(group_id):
        return jsonify({'status': 'error', 'message': 'Nothing to undo.'}), 400
    
    # Move current state to redo stack
    REDO_STACK.setdefault(group_id, []).append(ACTIVE_DATAFRAMES[group_id].copy())
    
    # Restore last state from undo stack
    ACTIVE_DATAFRAMES[group_id] = UNDO_STACK[group_id].pop()
    
    payload = _build_reset_payload(group_id)
    socketio.emit('data_reset', payload)
    return jsonify(payload)

@app.route('/redo/<group_id>', methods=['POST'])
def redo(group_id):
    if not REDO_STACK.get(group_id):
        return jsonify({'status': 'error', 'message': 'Nothing to redo.'}), 400
    
    # Move current state back to undo stack
    UNDO_STACK.setdefault(group_id, []).append(ACTIVE_DATAFRAMES[group_id].copy())
    
    # Restore state from redo stack
    ACTIVE_DATAFRAMES[group_id] = REDO_STACK[group_id].pop()

    payload = _build_reset_payload(group_id)
    socketio.emit('data_reset', payload)
    return jsonify(payload)


# --- NEW: Routes for Filter Recipes ---
@app.route('/export-filters/<group_id>', methods=['GET'])
def export_filters(group_id):
    """Exports the current filters for a group as a JSON file."""
    if group_id not in session.get('groups', {}):
        return jsonify({'status': 'error', 'message': 'Group not found.'}), 404

    filters = session['groups'][group_id].get('filters', [])
    if not filters:
        flash("No filters have been applied to this group yet.", "warning")
        return redirect(url_for('upload_file'))

    output_filename = f"{secure_filename(group_id)}_filter_recipe.json"
    
    buffer = io.StringIO()
    json.dump(filters, buffer, indent=2)
    buffer.seek(0)
    
    return Response(
        buffer.getvalue(),
        mimetype='application/json',
        headers={"Content-Disposition": f"attachment;filename={output_filename}"}
    )

@app.route('/import-filters/<group_id>', methods=['POST'])
def import_filters(group_id):
    """Imports a filter recipe JSON file and applies it to a group."""
    if group_id not in ACTIVE_DATAFRAMES:
        return jsonify({'status': 'error', 'message': 'Group data not in cache. Please re-upload file.'}), 404
    if 'filter_file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No filter recipe file provided.'}), 400

    file = request.files['filter_file']
    if file and file.filename.endswith('.json'):
        try:
            content = file.read().decode('utf-8')
            filters = json.loads(content)
            
            if not isinstance(filters, list) or not all('type' in f for f in filters):
                raise ValueError("Invalid filter file format.")

            push_undo(group_id) # Save state before importing

            # Replace session filters and apply them
            session['groups'][group_id]['filters'] = filters
            session.modified = True
            
            df_current = ACTIVE_DATAFRAMES[group_id]
            df_filtered = _apply_filters_to_df(df_current, filters)
            ACTIVE_DATAFRAMES[group_id] = df_filtered.copy()

            payload = _build_reset_payload(group_id)
            socketio.emit('data_reset', payload)
            
            # Send a specific response for the import action
            response_with_message = payload.copy()
            response_with_message['message'] = f'Successfully imported and applied {len(filters)} filters.'
            return jsonify(response_with_message)

        except (json.JSONDecodeError, ValueError) as e:
            return jsonify({'status': 'error', 'message': f'Invalid filter file: {e}'}), 400
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500
    return jsonify({'status': 'error', 'message': 'Invalid file type. Please upload a .json file.'}), 400

@app.route('/apply-column-operation/<group_id>', methods=['POST'])
def apply_column_operation(group_id):
    if group_id not in session.get('groups', {}):
        return jsonify({'status': 'error', 'message': 'Group not found.'}), 404
    if group_id not in ACTIVE_DATAFRAMES:
        return jsonify({'status': 'error', 'message': 'Error: Group data not found in active cache.'}), 404
    
    data = request.get_json()
    column_name = data.get('column_name')
    operation = data.get('operation_type')
    params = data.get('params', {})

    try:
        push_undo(group_id) # Save state before operation

        df = ACTIVE_DATAFRAMES[group_id]
        if column_name not in df.columns:
            return jsonify({'status': 'error', 'message': f"Column '{column_name}' not found."}), 404
        
        log_message = ""
        if operation == 'trim':
            df[column_name] = df[column_name].astype(str).str.strip()
            log_message = f"Trimmed whitespace from '{column_name}'."
        elif operation == 'lower':
            df[column_name] = df[column_name].astype(str).str.lower()
            log_message = f"Converted '{column_name}' to lowercase."
        elif operation == 'title':
            df[column_name] = df[column_name].astype(str).str.title()
            log_message = f"Converted '{column_name}' to title case."
        elif operation == 'fuzzy_merge':
            threshold = int(params.get('threshold', 90))
            df, log_message = _apply_fuzzy_merge(df, column_name, threshold)
        else:
            return jsonify({'status': 'error', 'message': 'Unsupported operation.'}), 400

        ACTIVE_DATAFRAMES[group_id] = df.copy()
        session['groups'][group_id]['transformations'].append(log_message)
        session.modified = True
        
        payload = _build_reset_payload(group_id)
        socketio.emit('data_reset', payload)

        return jsonify({'status': 'success', 'message': log_message})

    except Exception as e:
        print(f"Error applying column operation: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@socketio.on('edit_cell')
def handle_cell_edit(data):
    group_id = data.get('group_id')
    row_index = data.get('row_index')
    col_name = data.get('col_name')
    new_value = data.get('new_value')

    if group_id in ACTIVE_DATAFRAMES and row_index is not None and col_name is not None:
        try:
            push_undo(group_id) # Save state before cell edit

            df = ACTIVE_DATAFRAMES[group_id]
            actual_index = df.index[row_index]
            df.loc[actual_index, col_name] = new_value
            
            # Broadcast to all OTHER clients for immediate visual feedback.
            emit('cell_updated', data, broadcast=True, include_self=False)
            
            # Broadcast a full state reset to all clients (including self)
            # to update stats, charts, and undo/redo buttons.
            payload = _build_reset_payload(group_id)
            socketio.emit('data_reset', payload)

        except (IndexError, KeyError) as e:
            print(f"Error handling cell edit (likely stale index): {e}")
        except Exception as e:
            print(f"Error processing cell edit: {e}")

# --- NEW: Anomaly Detection Route ---
@app.route('/run-anomaly-detection/<group_id>', methods=['POST'])
def run_anomaly_detection(group_id):
    """
    Runs Isolation Forest algorithm on the numeric data of a group.
    NOTE: This is a read-only operation and does not modify the DataFrame.
    Therefore, it does not create an undo state.
    """
    data = request.get_json() or {}
    contamination = float(data.get('contamination', 0.05))

    # 1. Validate that the group and its DataFrame exist
    if group_id not in ACTIVE_DATAFRAMES:
        return jsonify({'status': 'error', 'message': 'Group data not found in active cache.'}), 404

    df = ACTIVE_DATAFRAMES[group_id]

    # 2. Select only numeric columns and fill missing values for the model
    df_numeric = df.select_dtypes(include=['number']).fillna(0)

    if df_numeric.empty:
        return jsonify({
            'status': 'error',
            'message': 'No numeric columns are available to run anomaly detection.'
        }), 400

    # 3. Fit the Isolation Forest model
    try:
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        predictions = iso_forest.fit_predict(df_numeric)

        # 4. Find the row indices flagged as anomalies (prediction == -1)
        anomaly_indices = [i for i, pred in enumerate(predictions) if pred == -1]

        # 5. Return the results as JSON
        return jsonify({
            'status': 'success',
            'count': len(anomaly_indices),
            'anomaly_indices': anomaly_indices
        })

    except Exception as e:
        print(f"Error during anomaly detection: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/export/<group_id>/<file_format>')
def export_file(group_id, file_format):
    if group_id not in ACTIVE_DATAFRAMES:
        flash("Processed data not found. It may have been cleared on server restart. Please re-upload or re-apply filters.", "error")
        return redirect(url_for('upload_file'))
    try:
        df = ACTIVE_DATAFRAMES[group_id]
        output_filename = f"{secure_filename(group_id)}_cleaned.{file_format}"
        if file_format == 'csv':
            buffer = io.StringIO()
            df.to_csv(buffer, index=False, encoding='utf-8')
            mimetype = 'text/csv'
            content = buffer.getvalue()
        elif file_format == 'xlsx':
            buffer = io.BytesIO()
            df.to_excel(buffer, index=False, engine='xlsxwriter')
            mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            content = buffer.getvalue()
        elif file_format == 'json':
            buffer = io.StringIO()
            df.to_json(buffer, orient='records', indent=4, date_format='iso')
            mimetype = 'application/json'
            content = buffer.getvalue()
        else:
            return "Unsupported format", 400
        
        buffer.seek(0)
        return Response(content, mimetype=mimetype, headers={"Content-Disposition": f"attachment;filename={output_filename}"})
    except Exception as e:
        print(f"Error exporting file: {e}")
        flash(f"Could not export file due to an error: {e}", "error")
        return redirect(url_for('upload_file'))

@app.route('/export-profile/<group_id>')
def export_profile(group_id):
    """Generates and serves a ydata-profiling HTML report."""
    if group_id not in ACTIVE_DATAFRAMES:
        flash("Processed data not found. It may have been cleared. Please re-upload.", "error")
        return redirect(url_for('upload_file'))

    df = ACTIVE_DATAFRAMES[group_id]
    display_name = session.get('groups', {}).get(group_id, {}).get('display_name', group_id)

    # Generate the report
    profile = ProfileReport(df, title=f"Profile Report: {display_name}", explorative=True)

    # Return HTML directly
    html = profile.to_html()

    resp = make_response(html)
    resp.headers['Content-Type'] = 'text/html'
    return resp

# --- Application Entry Point ---
if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    socketio.run(app, debug=True)