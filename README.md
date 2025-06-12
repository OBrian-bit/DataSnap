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

