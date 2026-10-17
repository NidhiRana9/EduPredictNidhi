# EduPredict — Student Performance Analysis and Prediction

EduPredict is an end-to-end Machine Learning and Data Analysis system designed to analyze student demographic and academic factors, predict student performance, and provide actionable insights for educators.

> **Project Phase:** **Phase 1 — Project Setup and MongoDB Connectivity** (Currently active).  
> In this phase, the workspace foundation, Python virtual environment, core dependencies, and a reusable, read-only MongoDB connection module are established and verified.

---

## 1. Technology Stack

- **Language:** Python (3.10+; verified with Python 3.14)
- **Database:** MongoDB Community Server (local unauthenticated instance on port `27017`)
- **Database Driver:** PyMongo (`>=4.6.0`)
- **Data Manipulation:** pandas (`>=2.2.0`)
- **Machine Learning:** scikit-learn (`>=1.4.0`)
- **Model Persistence:** joblib (`>=1.3.0`)
- **Visualization:** matplotlib (`>=3.8.0`)
- **Testing:** Python standard library `unittest` (no extra testing dependencies required)

---

## 2. Project Directory Structure

```text
EduPredict/
│
├── data/                      # Raw and processed CSV datasets (populated in Phase 2)
│   └── .gitkeep
│
├── src/                       # Application source code
│   ├── __init__.py            # Package initializer
│   └── database.py            # Reusable MongoDB connection manager and collections
│
├── models/                    # Serialized trained machine learning models
│   └── .gitkeep
│
├── results/                   # Metric outputs, predictions, and reports
│   ├── charts/                # Generated visualizations and plots
│   │   └── .gitkeep
│   └── .gitkeep
│
├── reports/                   # Documentation and final assignment reports
│   └── .gitkeep
│
├── tests/                     # Test suite
│   └── test_database.py       # MongoDB connectivity and handle verification tests
│
├── .gitignore                 # Git ignore rules for caches, venv, and large artifacts
├── requirements.txt           # Project dependencies and version bounds
└── README.md                  # Project overview and setup documentation
```

### Directory Roles
- **`data/`**: Stores raw CSV datasets and cleaned datasets for student tracking.
- **`src/`**: Houses production Python code, modules, and utilities.
- **`models/`**: Stores saved model files (`.joblib` or `.pkl`) for reuse in predictions.
- **`results/` & `results/charts/`**: Stores training metrics, prediction logs, and data charts.
- **`reports/`**: Contains academic assignment submissions and technical reports.
- **`tests/`**: Contains automated tests to ensure components behave correctly.

---

## 3. Prerequisites

Before running EduPredict, ensure you have installed:

1. **Python 3.10 or newer**:
   - Check in terminal:
     ```powershell
     python --version
     ```
2. **MongoDB Community Server**:
   - Running locally on the default port `27017`.
   - Default connection URI: `mongodb://localhost:27017/`

---

## 4. Virtual Environment Setup

Using a virtual environment isolates project dependencies from your global Python installation.

### A. Create the Virtual Environment

Run from the root of the `EduPredict` project:

**Windows PowerShell or Command Prompt:**
```powershell
python -m venv venv
```

This creates a local `venv/` folder containing an isolated Python interpreter and pip.

### B. Activate the Environment

#### Option 1: Windows PowerShell
```powershell
.\venv\Scripts\Activate.ps1
```

> **Note on PowerShell Execution Policy:**  
> If PowerShell displays an error such as `cannot be loaded because running scripts is disabled on this system`, you do **NOT** need to lower your system-wide security settings.  
> You can simply bypass the activation step and run the virtual environment's executables directly:
> ```powershell
> .\venv\Scripts\python.exe <script_path>
> .\venv\Scripts\pip.exe <command>
> ```
> Alternatively, enable script execution for only the current PowerShell window:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> .\venv\Scripts\Activate.ps1
> ```

#### Option 2: Windows Command Prompt (`cmd.exe`)
```cmd
venv\Scripts\activate.bat
```

### C. Verify Virtual Environment
Confirm that Python points to the project's `venv`:

**PowerShell:**
```powershell
Get-Command python | Select-Object -ExpandProperty Source
```
**Command Prompt:**
```cmd
where python
```
Expected path: `<path-to-EduPredict>\venv\Scripts\python.exe`.

---

## 5. Installing Dependencies

Install the pinned requirements inside the virtual environment:

**Using active virtual environment:**
```powershell
pip install -r requirements.txt
```

**Direct invocation (recommended if activation is bypassed):**
```powershell
.\venv\Scripts\pip.exe install -r requirements.txt
```

To verify the installation:
```powershell
.\venv\Scripts\python.exe -c "import pandas, pymongo, sklearn, joblib, matplotlib; print('All libraries imported successfully!')"
```

---

## 6. Checking MongoDB Server Status

MongoDB must be running on your system for EduPredict to connect.

### In Windows PowerShell:
Check if the MongoDB service is running:
```powershell
Get-Service -Name MongoDB
```
Or check if port 27017 is listening:
```powershell
Test-NetConnection -ComputerName localhost -Port 27017
```

If the service status is `Stopped`, start it with Administrator privileges:
```powershell
Start-Service MongoDB
```

### In Windows Command Prompt:
Check service status:
```cmd
sc query MongoDB
```
Start the service (Run as Administrator):
```cmd
net start MongoDB
```

---

## 7. Executing the MongoDB Connectivity Test

EduPredict includes a dedicated test to verify connection health without inserting, modifying, or deleting any documents.

### Method 1: Run the Standalone Test Runner (Recommended)

**PowerShell:**
```powershell
.\venv\Scripts\python.exe tests/test_database.py
```

**Command Prompt:**
```cmd
venv\Scripts\python.exe tests\test_database.py
```

### Method 2: Run via Python's Unittest Framework

**PowerShell:**
```powershell
.\venv\Scripts\python.exe -m unittest tests/test_database.py
```

**Command Prompt:**
```cmd
venv\Scripts\python.exe -m unittest tests\test_database.py
```

### Method 3: Run Database Module Directly
You can also run the database module directly for diagnostic output:
```powershell
.\venv\Scripts\python.exe src/database.py
```

### Expected Output
When MongoDB is running and reachable, you will see:
```text
test_01_client_initialization ... ok
test_02_database_ping ... ok
test_03_database_selection ... ok
test_04_collection_handles ... ok

============================================================
 EduPredict: Phase 1 - MongoDB Connectivity Test Runner
============================================================

[PASS] MongoDB ping succeeded! Server is active and reachable.
[PASS] Successfully selected target database: 'edupredict_db'.
[PASS] Defined collection handles: ['students', 'predictions', 'model_evaluations'] (no documents written or modified).
------------------------------------------------------------
[SUCCESS] All database tests passed! MongoDB is properly configured.
Project setup and Phase 1 database verification completed successfully.
```

---

## 8. Common MongoDB Connection Errors and Fixes

| Issue / Error | Cause | How to Fix |
|---|---|---|
| `ServerSelectionTimeoutError: localhost:27017: [WinError 10061] No connection could be made because the target machine actively refused it` | MongoDB server service is not running. | Start the service: <br>• PowerShell (Admin): `Start-Service MongoDB`<br>• CMD (Admin): `net start MongoDB` |
| `Cannot be loaded because running scripts is disabled on this system` | Windows PowerShell Execution Policy restricts running `.ps1` scripts. | Use `.\venv\Scripts\python.exe` directly, or run: <br>`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| `ModuleNotFoundError: No module named 'pymongo'` | The command was run using the global Python interpreter instead of the virtual environment. | Ensure you invoke `.\venv\Scripts\python.exe` or activate `venv` first. |
| `Connection refused on ::1 (IPv6)` | MongoDB is bound to IPv4 `127.0.0.1` by default on Windows. | The default URI `mongodb://localhost:27017/` in `src/database.py` automatically falls back to `127.0.0.1`. You can also specify `mongodb://127.0.0.1:27017/` directly if needed. |
| MongoDB service not found in `Get-Service` | MongoDB Community Server is not installed. | Download and install MongoDB Community Server and MongoDB Compass from the official MongoDB website: [mongodb.com/try/download/community](https://www.mongodb.com/try/download/community). |

---

## 9. Next Steps (Phase 2 Preview)

Once Phase 1 is reviewed and approved:
- **Phase 2:** Dataset acquisition, data schema inspection, and ingestion into the MongoDB `students` collection.
- **Phase 3:** Exploratory Data Analysis (EDA) and feature preprocessing with pandas.
- **Phase 4:** Model training (scikit-learn) and evaluation metrics storage.
- **Phase 5:** Prediction pipeline and assignment reporting.
