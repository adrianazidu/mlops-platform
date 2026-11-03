#!/bin/bash
# Automatically creates distinct databases required by Airflow, MLflow, and Serving.
set -e
set -u

function create_user_and_database() {
    local database=$1
    echo "  [DB Init] Creating database '$database'"
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
        CREATE DATABASE $database;
        GRANT ALL PRIVILEGES ON DATABASE $database TO $POSTGRES_USER;
EOSQL
}

if [ -n "$POSTGRES_MULTIPLE_DATABASES" ]; then
    echo "[DB Init] Creating multiple databases: $POSTGRES_MULTIPLE_DATABASES"
    for db in $(echo $POSTGRES_MULTIPLE_DATABASES | tr ',' ' '); do
        create_user_and_database $db
    done
    echo "[DB Init] All databases provisioned successfully."
fi
