MySQL MEMORY Engine in Docker

You can spin up a **MySQL server** whose data only lives in memory.

### Steps

1. **Start an ephemeral MySQL container**

   ```bash
   docker run --name mysql-memory \
      -e MYSQL_ROOT_PASSWORD=root \
      -d -p 3306:3306 \
      mysql:latest \
      --default-storage-engine=MEMORY

   ```

   This starts a fresh MySQL server in memory.
   When you stop the container, everything disappears.

2. **Import your dump**

   ```bash
   docker exec -it mysql-memory mysql -u root -proot -e "SHOW DATABASES;"

   docker cp DemoDatabase.sql mysql-memory:/DemoDatabase.sql
   docker exec -i mysql-memory bash -c "mysql -u root -proot < /dump.sql"


   docker exec -it mysql-memory mysql -u root -proot -e "USE opendata_rijksoverheid_dbs; SHOW TABLES;" 

   ```
   <!-- docker exec -i mysql-memory bash -c "mysql -u root -proot -e 'CREATE DATABASE migratodb;'"
   docker exec -i mysql-memory bash -c "mysql -u root -proot migratodb < /DemoDatabase.sql" -->

3. **Query from Python**

   ```python
   import mysql.connector

   conn = mysql.connector.connect(
       host="localhost",
       user="root",
       password="root",
       database="your_database_name"
   )

   cur = conn.cursor()
   cur.execute("SHOW TABLES;")
   print("Tables:", cur.fetchall())

   cur.execute("SELECT COUNT(*) FROM your_table;")
   print("Row count:", cur.fetchone())

   # Sample 10 rows
   cur.execute("SELECT * FROM your_table LIMIT 10;")
   for row in cur.fetchall():
       print(row)

   cur.close()
   conn.close()
   ```

4. **Remove it when done**

   ```bash
   docker rm -f mysql-memory
   ```