#!/bin/bash
# Auto-repair script for NAH Memory
set -e

echo "=== Setting up NAH Memory Schema ==="
psql -h localhost -U human -d notahuman -f /Users/human/notahuman/infrastructure/memory/nah-memory-unified/setup_schema.sql

echo ""
echo "=== Running Data Migration ==="
cd /Users/human/notahuman/infrastructure/memory/nah-memory-unified
./venv/bin/python migrate_data.py

echo ""
echo "=== Verification ==="
psql -h localhost -U human -d notahuman -c "SELECT schemaname, tablename FROM pg_tables WHERE schemaname = 'nah_memory';"

echo ""
echo "DONE! Schema created and data migrated."
