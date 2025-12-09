import mysql.connector
import csv

config = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "root",
    "port": 3306,
    "database": "opendata_rijksoverheid_dbs",
}

print("Connecting to MySQL...")
conn = mysql.connector.connect(**config)
cur = conn.cursor()

cur.execute("SHOW TABLES;")
tables = cur.fetchall()

print("\nTables:")
for (t,) in tables:
    print(" -", t)

if tables:
    table = tables[0][0]
    print(f"\nInspecting table: {table}")

    cur.execute(f"SELECT count(*) FROM {table} where cluster_id IS NOT NULL")
    print("Total rows:", cur.fetchone()[0])

    cur.execute(f"""
        SELECT  cluster_id,count(*) 
        FROM {table} 
        WHERE cluster_id IS NOT NULL
        GROUP BY cluster_id
    """)

    for row in cur.fetchall():
        print(repr(row))
    
    cur.execute(f"SELECT asset_id,filename,firstpagetxt,cluster_id,item_type,parent_id FROM {table} WHERE cluster_id IS NOT NULL")

    rows = cur.fetchall()
    columns = [desc[0].lower() for desc in cur.description]

    print("\nSample rows:")
    for row in rows:
        print(row)

    with open(f"{table}_sample.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)
    print(f"\nExported first 5 rows to {table}_sample.csv")

cur.close()
conn.close()
print("\nDone.")
