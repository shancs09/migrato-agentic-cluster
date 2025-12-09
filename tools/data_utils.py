"""
db_utils.py
------------
Unified data I/O utilities for labeling pipeline.
Supports:
 - CSV read/write (local workflow)
 - MySQL read/write (persistent storage)
"""

import os
import pandas as pd
import mysql.connector
from dotenv import load_dotenv
from fastapi import HTTPException

# ---------- LOAD ENV ----------
load_dotenv()

# ---------- CONFIG ----------
TABLE_NAME = os.getenv("TABLE_NAME", "core_assets")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./data")
CSV_PATH = os.path.join(OUTPUT_DIR, f"{TABLE_NAME}_sample.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "127.0.0.1"),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", "root"),
    "port": int(os.getenv("MYSQL_PORT", "3306")),
    "database": os.getenv("MYSQL_DATABASE", "opendata_rijksoverheid_dbs"),
}
## -----------------------------------------------------------------
# GENERIC DB EXECUTE
## -----------------------------------------------------------------
def db_execute(query: str, params: tuple = None) -> pd.DataFrame:
    """Run a MySQL query and return results as a DataFrame."""

    # Connect
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cur = conn.cursor()
    except mysql.connector.Error as e:
        raise HTTPException(
            status_code=500,
            detail=f"MySQL connection failed: {str(e)}"
        )

    # Execute
    try:
        print(query)
        cur.execute(query, params or ())
        rows = cur.fetchall()
        columns = [desc[0].lower() for desc in cur.description]
        return pd.DataFrame(rows, columns=columns)

    except mysql.connector.Error as e:
        raise HTTPException(
            status_code=500,
            detail=f"MySQL query failed: {str(e)}"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected MySQL error: {str(e)}"
        )

    finally:
        try:
            cur.close()
            conn.close()
        except:
            pass

def db_execute_write(query: str, params: tuple = None) -> int:
    """
    Execute INSERT/UPDATE/DELETE queries.
    Returns number of affected rows.
    """
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cur = conn.cursor()
    except mysql.connector.Error as e:
        raise HTTPException(
            status_code=500,
            detail=f"MySQL connection failed: {str(e)}"
        )

    try:
        cur.execute(query, params or ())
        conn.commit()
        return cur.rowcount

    except mysql.connector.Error as e:
        raise HTTPException(
            status_code=500,
            detail=f"MySQL write query failed: {str(e)}"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected MySQL error: {str(e)}"
        )

    finally:
        try:
            cur.close()
            conn.close()
        except:
            pass

# -----------------------------------------------------------------
# CSV FUNCTIONS
# -----------------------------------------------------------------
def read_from_csv() -> pd.DataFrame:
    """Read dataset from CSV file."""
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"❌ CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    print(f"📄 Loaded {len(df)} rows from CSV.")

    if "firstpagetxt" in df.columns:
        df["firstpagetxt"] = df["firstpagetxt"].fillna("").astype(str)
    return df


def save_to_csv(df: pd.DataFrame):
    """Save DataFrame back to CSV (overwrites in place)."""
    df.to_csv(CSV_PATH, index=False)
    print(f"💾 CSV updated: {CSV_PATH}")


# -----------------------------------------------------------------
# MYSQL FUNCTIONS
# -----------------------------------------------------------------
def read_from_mysql() -> pd.DataFrame:
    query = f"""
        SELECT cluster_id,cluster_label, label_status, labels_used
        FROM {TABLE_NAME}
        WHERE cluster_id IS NOT NULL
    """
    return db_execute(query)

def db_read_unlabeled_cluster() -> pd.DataFrame:
    query = f"""
        SELECT DISTINCT cluster_id,cluster_label
        FROM {TABLE_NAME}
        WHERE cluster_id IS NOT NULL 
          AND cluster_label IS NULL
    """
    return db_execute(query)

def db_read_single_cluster(cluster_id: int) -> pd.DataFrame:
    query = f"""
        SELECT *
        FROM {TABLE_NAME}
        WHERE cluster_id = %s
    """
    return db_execute(query, (cluster_id,))


def db_read_limit_cluster(limit: int) -> pd.DataFrame:

    query = f"""
        SELECT t1.*
        FROM {TABLE_NAME} AS t1
        JOIN (
            SELECT DISTINCT cluster_id
            FROM {TABLE_NAME}
            WHERE cluster_id IS NOT NULL AND cluster_label IS NULL
            LIMIT %s
        ) AS limited_clusters
        ON t1.cluster_id = limited_clusters.cluster_id;
    """
    print(query)
    return db_execute(query, (limit,))

def update_mysql_cluster_label(cluster_id: int, label: str, status: str, labels_used: str):
    query = f"""
        UPDATE {TABLE_NAME}
        SET cluster_label = %s,
            label_status = %s,
            labels_used = %s
        WHERE cluster_id = %s
    """
    return db_execute_write(query, (label, status, labels_used, cluster_id))
def update_mysql_reset_labels():
    query = f"""
        UPDATE {TABLE_NAME}
        SET cluster_label = NULL,
            label_status = NULL,
            labels_used = NULL
    """
    return db_execute_write(query)

def extend_mysql_schema():
    query = f"""
        ALTER TABLE {TABLE_NAME}
        ADD COLUMN IF NOT EXISTS cluster_label VARCHAR(255),
        ADD COLUMN IF NOT EXISTS label_status VARCHAR(50),
        ADD COLUMN IF NOT EXISTS labels_used TEXT;
    """

    db_execute_write(query)

    return {"message": f"✅ Table '{TABLE_NAME}' updated with label columns (safe add)."}


def get_data(source: str = "csv") -> pd.DataFrame:
    """
    Unified data reader.
    - source='csv' → reads from CSV
    - source='db' → reads from MySQL
    """
    if source == "csv":
         df= read_from_csv()
    elif source == "db":
         df= read_from_mysql()
    else:
        raise ValueError("❌ Unsupported source. Use 'csv' or 'mysql'.")
    # # Ensure label columns exist
    required_cols = ["cluster_label", "label_status", "labels_used"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = None
            print(f"Added missing column: {col}")
    return df

def update_mysql_reset_labels_limit(limit: int):
    query = f"""
        UPDATE {TABLE_NAME} AS t
        JOIN (
            SELECT DISTINCT cluster_id
            FROM {TABLE_NAME}
            WHERE cluster_id IS NOT NULL AND cluster_label IS NOT NULL
            LIMIT %s
        ) AS x ON t.cluster_id = x.cluster_id
        SET t.cluster_label = NULL,
            t.label_status = NULL,
            t.labels_used = NULL;
    """
    return db_execute_write(query, (limit,))