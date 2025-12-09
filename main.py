from fastapi import FastAPI, Query
from fastapi import Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from typing import Optional
from tools.data_utils import get_data,extend_mysql_schema,update_mysql_cluster_label,update_mysql_reset_labels,db_read_unlabeled_cluster,db_read_single_cluster,db_read_limit_cluster,update_mysql_reset_labels_limit
from tools.cluster_labeler import infer_cluster_label 
import pandas as pd,shutil
from datetime import datetime
import os,math,json
from pydantic import BaseModel
from typing import Dict, List, Optional


app = FastAPI(
    title="Cluster Labeling & AI Inference API",
    description=(
        "AI-powered document cluster labeling workflow using watsonx foundation models. "
        "Supports reading data, inferring cluster labels, exporting results and summaries."
    ),
    version="1.0.0",
    openapi_version="3.1.0",        
    contact={
        "name": "EMEA Build Lab",
        "email": "shanmsel@in.ibm.com"
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html"
    }
)

DATA_DIR = "./data"
RESULT_FILE = os.path.join(DATA_DIR, "core_assets_sample.csv")
os.makedirs(DATA_DIR, exist_ok=True)

DB_EXPORT_FILE = os.path.join(DATA_DIR, "core_assets_db_export.csv")

DEFAULT_DATA_SOURCE=os.getenv("DEFAULT_DATA_SOURCE","csv")
ENABLE_DATA_BACKUP=os.getenv("ENABLE_DATA_BACKUP","false")

# Mount the data directory so files can be served publicly
app.mount("/files", StaticFiles(directory=DATA_DIR), name="files")

# --------------------------------------------------------------------
# /data/read  →  Discover clusters and labeling status
# --------------------------------------------------------------------
class Overview(BaseModel):
    total_documents: int
    total_clusters: int
    labeled_clusters: int
    unlabeled_clusters: int
    coverage_percent: float
    data_source: str

class ReadDataResponse(BaseModel):
    overview: Overview
    doc_label_status_distribution: Dict[str, int]
    unlabeled_cluster_ids: List[int]

@app.get("/data/read", response_model=ReadDataResponse, operation_id="read_data")
async def read_data(source: str = Query(DEFAULT_DATA_SOURCE, description="Data source: csv or db")):
    df = get_data(source)

    if df.empty:
        return {"error": "No data found in source"}

    total_docs = len(df)
    total_clusters = df["cluster_id"].nunique()
    labeled_clusters = df[df["cluster_label"].notnull()]["cluster_id"].nunique()
    # print(len(labeled_clusters))
    unlabeled_clusters = total_clusters - labeled_clusters
    unlabeled_cluster_ids = (
        df[df["cluster_label"].isnull()]["cluster_id"].dropna().unique().tolist()
    )
    # print(len(unlabeled_cluster_ids))
    # print(unlabeled_clusters)
    
    coverage_percent = round((labeled_clusters / total_clusters) * 100, 2) if total_clusters else 0

    status_counts = (
        df["label_status"].fillna("Unlabeled").value_counts().to_dict()
        if "label_status" in df.columns else {}
    )

    return ReadDataResponse(
        overview=Overview(
            total_documents=total_docs,
            total_clusters=total_clusters,
            labeled_clusters=labeled_clusters,
            unlabeled_clusters=unlabeled_clusters,
            coverage_percent=coverage_percent,
            data_source=source
        ),
        doc_label_status_distribution=status_counts,
        unlabeled_cluster_ids=unlabeled_cluster_ids
    )

class UnlabeledClustersResponse(BaseModel):
    unlabeled_cluster_ids: List[int]

@app.get(
    "/data/unlabeled-clusters",
    response_model=UnlabeledClustersResponse,
    operation_id="get_unlabeled_clusters"
)
async def get_unlabeled_clusters(source: str = Query(DEFAULT_DATA_SOURCE, description="Data source: csv or db")):
    """
    Returns only the list of unlabeled cluster IDs for lightweight operations.
    """
    if source == "db":
        df = db_read_unlabeled_cluster()
        print(len(df))
    else:
        df = get_data(source)
    if df.empty:
        return {"error": "No data found in source"}

    unlabeled_cluster_ids = (
        df[df["cluster_label"].isnull()]["cluster_id"]
        .dropna()
        .unique()
        .tolist()
    )
    print(len(unlabeled_cluster_ids))
    return UnlabeledClustersResponse(
        unlabeled_cluster_ids=unlabeled_cluster_ids
    )


# --------------------------------------------------------------------
# Function →  Process for single cluster
# --------------------------------------------------------------------
def process_single_cluster(
    df,
    cluster_id: int,
    source: str
    # sample_size: Optional[int],
    # similarity_threshold: Optional[float],
):

    # Validate cluster existence
    if cluster_id not in df["cluster_id"].unique():
        return {"error": True, "message": f"Cluster {cluster_id} not found"}

    # Check if already labeled
    existing_label = df.loc[df["cluster_id"] == cluster_id, "cluster_label"].iloc[0]
    if pd.notnull(existing_label):
        return {
            "error": False,
            "skip": True,
            "message": f"Cluster {cluster_id} already labeled",
            "cluster_id": cluster_id,
            "cluster_label": existing_label
        }

    # Run inference
    cluster_df = df[df["cluster_id"] == cluster_id]

    result = infer_cluster_label(
        cluster_df
        # sample_size=sample_size,
        # similarity_threshold=similarity_threshold,
    )

    label = result.get("cluster_label", "Unknown")
    status = result.get("status", "Unknown")
    labels_used_json = json.dumps(result.get("labels", []), ensure_ascii=False)
    similarity = result.get("similarity_score", 0.0)

    # Update dataframe
    df.loc[df["cluster_id"] == cluster_id, "cluster_label"] = label
    df.loc[df["cluster_id"] == cluster_id, "label_status"] = status
    df.loc[df["cluster_id"] == cluster_id, "labels_used"] = labels_used_json

    # Save update
    if source == "db":
        update_mysql_cluster_label(cluster_id, label, status, labels_used_json)
    elif source == "csv":
        df.to_csv(RESULT_FILE, index=False)
    
    return {
        "error": False,
        "skip": False,
        "cluster_id": cluster_id,
        "cluster_label": label,
        "status": status,
        "similarity_score": similarity,
        "labels_used": result.get("labels", []),
    }
# --------------------------------------------------------------------
# /cluster/infersingle  →  Infer label for single cluster
# --------------------------------------------------------------------
def backup_result_file():
    """Perform backup if ENABLE_BACKUP=true and result file exists."""
    if not ENABLE_DATA_BACKUP:
        return None

    if not os.path.exists(RESULT_FILE):
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = RESULT_FILE.replace(".csv", f"_{timestamp}_backup.csv")
    shutil.copy(RESULT_FILE, backup_path)

    return backup_path

@app.get("/cluster/infersingle", operation_id="infer_labels_cluster_single")
async def infer_labels_single(
    cluster_id: int = Query(...)
    # sample_size: Optional[int] = None,
    # similarity_threshold: Optional[float] = None,
):
    source = DEFAULT_DATA_SOURCE
    if source=="db":
        df = db_read_single_cluster(cluster_id)
    else:
        df = get_data(source)
    if df.empty:
        return {"error": "No data found in source"}

    # Backup file
    backup_path = backup_result_file()

    # Shared core processing
    result = process_single_cluster(
        df, cluster_id, DEFAULT_DATA_SOURCE
        #sample_size, similarity_threshold
    )

    return {
        "message": f"Processed Cluster : {cluster_id}",
        "backup_file": backup_path,
        "updated_file": RESULT_FILE,
        # "sample_size": sample_size,
        # "similarity_threshold": similarity_threshold,
        "results": result,
    }
# --------------------------------------------------------------------
# /cluster/infer  →  Infer labels for clusters
# --------------------------------------------------------------------
@app.get("/cluster/infer", operation_id="infer_labels_cluster")
async def infer_labels(
    # cluster_id: Optional[int] = None,
    limit: int = 10,
    process_all: bool = False,
    source: str = Query(DEFAULT_DATA_SOURCE)
):
    if source=="db":
        if limit:
            df = db_read_limit_cluster(limit)
        elif process_all:
            df = db_read_unlabeled_cluster()
    else:
        df = get_data(source)

    if df.empty:
        return {"error": "No data found"}

    # Backup
    backup_path = backup_result_file()

    # Determine clusters
    unlabeled = df[df["cluster_label"].isnull()]["cluster_id"].unique().tolist()

    # if cluster_id is not None:
    #     target_clusters = [cluster_id]
    if limit and limit > 0:
        target_clusters = unlabeled[:limit]
    elif process_all:
        target_clusters = unlabeled

    results = []
    for cid in target_clusters:
        res = process_single_cluster(
            df, cid, source
        )
        results.append(res)

    return {
        "message": f"Processed {len(target_clusters)} clusters",
        "backup_file": backup_path,
        "updated_file": RESULT_FILE,
        "sample_size": sample_size,
        "similarity_threshold": similarity_threshold,
        "results": results,
    }

# --------------------------------------------------------------------
# /cluster/inferlimit  →  Infer labels for limited clusters
# --------------------------------------------------------------------
@app.get("/cluster/inferlimit", operation_id="infer_labels_cluster_limit")
async def infer_labels_limit(
    limit: int = 10,
    source: str = Query(DEFAULT_DATA_SOURCE)
):
    if source=="db":
        if limit:
            df = db_read_limit_cluster(limit)
    else:
        df = get_data(source)

    if df.empty:
        return {"error": "No data found"}

    # Backup
    backup_path = backup_result_file()

    # Determine clusters
    unlabeled = df[df["cluster_label"].isnull()]["cluster_id"].unique().tolist()

    # if cluster_id is not None:
    #     target_clusters = [cluster_id]
    if limit and limit > 0:
        target_clusters = unlabeled[:limit]
    else:
        target_clusters = unlabeled
    results = []
    for cid in target_clusters:
        res = process_single_cluster(
            df, cid, source
        )
        results.append(res)

    return {
        "message": f"Processed {len(target_clusters)} clusters",
        "backup_file": backup_path,
        "updated_file": RESULT_FILE,
        "results": results,
    }

# --------------------------------------------------------------------
# Export CSV directly as file (safe for rendering/download)
# --------------------------------------------------------------------
def get_export_file_url(request: Request, file_path: str):
    """Return a download URL for any given file."""
    if not os.path.exists(file_path):
        return None, f"File not found: {file_path}"

    file_name = os.path.basename(file_path)
    base_url = str(request.base_url).rstrip("/")
    download_url = f"{base_url}/download/{file_name}"

    return download_url, None

@app.get("/download/{filename}",operation_id="download_file")
async def download_file(filename: str):
    file_path = os.path.join(DATA_DIR, filename)

    if not os.path.exists(file_path):
        return {"error": "File not found"}
    # This sets the proper headers to force "Save As" dialog
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="text/csv",  # still correct content type
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )
# --------------------------------------------------------------------
# Export summarized JSON view for front-end rendering
# --------------------------------------------------------------------

@app.get("/results/export", operation_id="export_results_csv")
async def export_results(
            request: Request,
            format: str = Query("csv", description="Format: csv ,json(default)"),
            source: str = Query(DEFAULT_DATA_SOURCE, description="Format: db,csv)")
        ):
    if not os.path.exists(RESULT_FILE):
        return {"error": "No result file found. Please run /cluster/infer first."}

    if format == "csv":
        if source == "csv":
            download_url, error = get_export_file_url(request, RESULT_FILE)
        elif source == "db":
            download_url, error = get_export_file_url(request, DB_EXPORT_FILE)
        return {
            "file_source": download_url,
            "message": "File available for download"
        }

    elif format == "json":
        if source == "csv":
            df = pd.read_csv(RESULT_FILE)
        elif source == "db":
            df = pd.read_csv(DB_EXPORT_FILE)
        df = pd.read_csv(RESULT_FILE)
        records = df.replace({math.nan: None}).to_dict(orient="records")
        return JSONResponse(content={"records": records})

    else:
        return {"error": "Unsupported format. Use 'csv' or 'json'."}
    
@app.get("/results/export/summary",operation_id="results_summary")
async def export_summary( request: Request,source: str = Query(DEFAULT_DATA_SOURCE, description="Data source: csv or db"),filter: Optional[str] = Query(None), sort: Optional[str] = Query(None)):
    """
    Reads data and identifies which clusters have or lack labels.
    """
    df = get_data(source)

    if df.empty:
        return {"error": "No data found in source"}
    
    if source == "db":
        # fresh DB snapshot export
        db_export_path = os.path.join(DATA_DIR, "core_assets_db_export.csv")
        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(db_export_path, index=False)
        exported_file = db_export_path
    else:
        exported_file = RESULT_FILE

    for col in ["cluster_id", "cluster_label", "label_status"]:
        if col not in df.columns:
            return {"error": f"Missing required column: {col}"}

    total_clusters = int(df["cluster_id"].nunique())
    total_documents = len(df)
    labeled_clusters = int(df[df["cluster_label"].notnull()]["cluster_id"].nunique())
    unlabeled_clusters = total_clusters - labeled_clusters
    coverage_percent = round((labeled_clusters / total_clusters) * 100, 2)

    # 1️⃣ Summary by status
    status_counts = (
        df["label_status"].fillna("Unlabeled").value_counts().to_dict()
    )
    by_status = [{"status": k, "clusters": v} for k, v in status_counts.items()]

    # 2️⃣ Group by label
    label_groups = {}
    labeled_df = df[df["cluster_label"].notnull()]
    grouped = labeled_df.groupby("cluster_label")["cluster_id"].unique()

    for label, clusters in grouped.items():
        label_groups[label] = list(map(int, clusters))

    # optional filtering/sorting
    if sort == "label_count":
        label_groups = dict(sorted(label_groups.items(), key=lambda x: len(x[1]), reverse=True))

    if filter == "manual":
        df = df[df["label_status"].str.contains("Manual", na=False)]

    by_label = [
        {"label": label, "cluster_count": len(clusters), "cluster_ids": clusters}
        for label, clusters in label_groups.items()
    ]

    dominant_label = max(label_groups, key=lambda k: len(label_groups[k])) if label_groups else None
    dominant_label_ratio = round(
        len(label_groups.get(dominant_label, [])) / total_clusters * 100, 2
    ) if dominant_label else 0

    download_file_url, error = get_export_file_url(request, exported_file)

    
    summary = {
        "overview": {
            "total_clusters": total_clusters,
            "total_documents": total_documents,
            "labeled_clusters": labeled_clusters,
            "unlabeled_clusters": unlabeled_clusters,
            "coverage_percent": coverage_percent,
            "dominant_label": dominant_label,
            "dominant_label_ratio": dominant_label_ratio
        },
        "document_by_status": by_status,
        "cluster_by_label": by_label,
        "file_source": download_file_url
    }

    return JSONResponse(content=summary)

# --------------------------------------------------------------------
# /data/extend-schema  → Extend DB schema
# --------------------------------------------------------------------
@app.post("/db/extend-schema",operation_id="extend_db_schema")
def extend_schema():
    """
    Extends the database schema to add cluster labeling fields.
    """
    try:
        extend_mysql_schema()
        return {"message": "Database schema extended successfully."}
    except Exception as e:
        return {"error": str(e)}

# --------------------------------------------------------------------
# /data/reset  →  Reset all labels in data source
# --------------------------------------------------------------------

@app.post("/data/reset",operation_id="reset_labels")
async def reset_labels(
    source: str = Query(DEFAULT_DATA_SOURCE, description="Data source: csv or db"),
    confirm: bool = Query(False, description="Must be true to perform reset")
):
    if not confirm:
        return {"error": "Please confirm the reset by passing ?confirm=true"}

    # Save changes
    if source == "csv":
        df = get_data(source)
        if df.empty:
            return {"error": "No data found in source"}
        # Ensure columns exist
        for col in ["cluster_label", "label_status", "labels_used"]:
            if col not in df.columns:
                df[col] = None

        # Create backup
        backup_path = backup_result_file()

        # Reset the 3 label columns
        df["cluster_label"] = None
        df["label_status"] = None
        df["labels_used"] = None

        df.to_csv(RESULT_FILE, index=False)
        print(f"Reset CSV file: {RESULT_FILE}")
    else:
        update_mysql_reset_labels()
        print("Reset DB table")

    return JSONResponse(content={
        "message": f"Reset completed for {source.upper()}",
        "total_rows": len(df) if source == "csv" else "N/A",
        "backup_file": backup_path if source == "csv" else None,
        "columns_reset": ["cluster_label", "label_status", "labels_used"],
        "file_source": RESULT_FILE if source == "csv" else "MySQL DB"
    })


@app.post("/data/resetlimit",operation_id="reset_labels_limit")
async def reset_labels_limit(
    limit: int = 10,
    confirm: bool = Query(False, description="Must be true to perform reset")
):
    if not confirm:
        return {"error": "Please confirm the reset by passing ?confirm=true"}
    # Save changes
    update_mysql_reset_labels_limit(limit)
    return JSONResponse(content={
        "message": f"Reset completed for {limit} rows in DB",
    })