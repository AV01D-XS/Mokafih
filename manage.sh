#!/bin/bash

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: ./manage.sh [KEY_NAME] [COMPANY_NAME]"
    echo "Example: ./manage.sh TEST-KEY-99 'Jordan Bank'"
    exit 1
fi

KEY=$1
COMPANY=$2
DAYS=365

echo "🔑 Creating License for: $COMPANY..."

# We pipe this clean Python code directly into the container
docker compose exec -T bot python - <<EOF
import asyncio
from main import db

async def create_key():
    try:
        await db.init()
        # Create license: Key, Company, 100 Users, 365 Days
        await db.create_license('$KEY', '$COMPANY', 100, $DAYS)
        print(f"✅ SUCCESS: License '$KEY' created for '$COMPANY'")
    except Exception as e:
        print(f"❌ ERROR: {e}")

asyncio.run(create_key())
EOF
