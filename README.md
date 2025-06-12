# PureView Project Suite — AI Data Engineering Tools

This repository contains multiple sub-projects focused on improving data quality, automation, and intelligence gathering for AI-driven data engineering workflows.

---

## 📌 Project 1: PureView

### 🔍 Why We’re Doing This

- **Quality at a Glance**: Instantly verify that every dataset has been cleaned to standard (e.g., duplicates removed, invalid rows dropped).
- **Time Saver**: Replaces manual script checks with one visual hub.
- **Reporting Ready**: Centralized metrics feed straight into reports.

---

### ✅ What Success Looks Like

| **Metric**        | **Raw** | **Cleaned** | **% Change** |
|-------------------|--------:|------------:|-------------:|
| Total records     |    •     |      •       |       •       |
| Duplicate rows    |    •     |      •       |       •       |

---

### 🔄 Data Flow Overview

1. **Post-Cleaning Analysis Script**: Generates summary CSV per dataset.
2. **Lightweight Harvester App**: Ingests summary CSVs into SQLite.
3. **Python Dashboard**: Reads SQLite to render interactive charts and tables.
4. *(Optional)* **Nightly Export**: Cron job outputs master CSV for audit or PowerBI/Tableau users.

---

### 🧱 Scope of Work

- Finalize metric list & CSV schema
- Build harvester + SQLite loader; test with 3 datasets
- Create dashboard with filters, bar charts, and duplicate metrics
- User acceptance testing, polish, and documentation hand-off

---

## 🔎 Project 2: ReconX

### 🧠 What It Is

**ReconX** is an autonomous tool designed to automate, supplement, and scale the effort of discovering leaked datasets across the web.

---

### ⚠️ Why It Matters

- Leaked data often vanishes quickly
- Manual monitoring is limited in scale
- ReconX ensures real-time, continuous discovery with full traceability

---

### 🛠️ What It Does

- Crawls breach forums and file-sharing hosts using crafted queries
- Collects and catalogs metadata
- Logs every decision and action for full auditability

---

### 🎯 Target Users

- Data Engineering Teams
- Threat Intelligence Teams

---

### 💡 Strategic Benefits

- Automated discovery of leaked data
- Greater efficiency over manual efforts
- Transparent governance with structured logs

---

## 🧽 Project 3: Scrub AI (Data Cleaning Agent)

### 🎯 Goal

Use AI to analyze a dataset sample and generate a Python data-cleaning script by auto-assigning variables and logic based on a provided template.

---

### 🧩 Key Features

1. **AI-Led Data Analysis**  
   - Identifies all columns
   - Flags key columns for cleaning logic

2. **Semi-Automated User-in-the-Loop Mode**  
   - Allows users to confirm or override AI-detected variables and logic  
   - Ideal for ambiguous or inconsistently labeled datasets

3. **Column Typing & Usefulness Classification**  
   - **Useful Columns**: Contain high-value, interpretable info (e.g., name, email, birthdate, hashes)  
   - **Non-Useful Columns**: All others

---

### 🧾 Expected Output

A Python script with:
- Pre-assigned variable roles
- Core logic to clean the dataset
- Clear, modifiable sections for user review

---

### 📁 Status

All projects are actively under development. Contributions and feedback are welcome.

______________________________________________________________________________________________________________________________________________________________________________________________________________________________
# 📊 Metrics for Post-Clean Analytics Dashboard (PureView)

**PureView** is a lightweight dashboard designed to visualize key data cleaning metrics, giving teams immediate insight into dataset quality—without reviewing code or logs manually.

The system pulls from post-cleaning script outputs (e.g., duplicates removed, garbage rows, validation success), stores results in a centralized **SQLite** database, and renders interactive dashboards with **Python**.

This approach helps:
- Save time
- Ensure consistency
- Support data quality reporting across teams

---

## 📐 Core Metrics

| **Metric**                     | **Raw Dataset** | **Clean Dataset** | **% of Raw** | **Why It Matters**                                                                 |
|--------------------------------|------------------|--------------------|--------------|--------------------------------------------------------------------------------------|
| **Duplicate Count**            | •                | •                  | •            | Verifies deduplication logic (should drop to 0).                                     |
| **Validation Pass**            | •                | •                  | •            | Confirms how much data survives quality checks.                                      |
| **Garbage Records (Rejected)** | •                | –                  | •            | Quick view of unusable data volume.                                                  |
| **Processing Runtime**         | –                | `hh:mm:ss`         | –            | Gauges script efficiency for each dataset.                                           |
| **Distinct IDs**               | •                | •                  | –            | Ensures no duplicate primary keys remain post-clean.                                 |

> ℹ️ Fill in the “•” programmatically from summary CSVs. Dashes “–” indicate metrics that don’t require before/after splits.

---

## 📈 Recommended Chart Visuals

| **Metric**            | **Chart Type**                              | **Why This Works**                                                                 |
|-----------------------|---------------------------------------------|-------------------------------------------------------------------------------------|
| Duplicate Count       | Side-by-side bar chart (Raw vs Clean)       | Clearly shows reduction in duplicates across datasets.                             |
| Validation Pass       | Stacked bar (Valid vs Invalid) OR Donut     | Visualizes what portion of data passed validation.                                 |
| Garbage Records       | Donut chart OR Bar chart by dataset         | Emphasizes unusable data and allows for dataset comparison.                        |
| Processing Runtime    | Horizontal bar OR Line chart over time      | Compares runtime per dataset and trends over multiple cleaning runs.               |
| Distinct IDs          | Side-by-side bar chart                      | Shows pre/post-cleaning uniqueness to ensure deduplication worked correctly.        |

---

## ✨ Optional Extras (Advanced)

- Tooltip overlays with raw counts & calculated percentages
- Trend lines to show quality improvement across time
- Data Quality Score heatmap for benchmarking across datasets

These extras make the dashboard more insightful for both technical and non-technical users.

---

## 🧭 Recommended Dashboard Layout

### 1️⃣ Summary Table View (Top Section)
A concise overview table displaying:
- Raw vs Clean counts
- Percentages
- Key indicators (runtime, unique IDs)

**Purpose**:  
Provides a high-level snapshot of cleaning performance.

**Bonus**:  
Enable CSV or PDF export for audit trails or business reports.

---

### 2️⃣ Visual Insights (Bottom or Side Section)
Interactive charts tied to each key metric:
- Garbage % → Donut chart  
- Runtime → Horizontal bar  
- Distinct IDs → Side-by-side bar  

**Purpose**:  
Makes trends, outliers, and improvements immediately visible. Useful in stakeholder presentations or data health dashboards.

---

PureView bridges the gap between raw cleaning logs and actionable insights—transforming technical outputs into a visual story everyone can understand.

