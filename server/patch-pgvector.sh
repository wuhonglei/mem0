#!/bin/sh
# Patch pgvector list() to ORDER BY created_at DESC
python3 -c "
import re
f = '/usr/local/lib/python3.12/site-packages/mem0/vector_stores/pgvector.py'
with open(f) as fh:
    content = fh.read()
# Add ORDER BY before LIMIT in the list method's SELECT query
old = '''                SELECT id, payload
                FROM {}
                {}
                LIMIT %s'''
new = '''                SELECT id, payload
                FROM {}
                {}
                ORDER BY (payload->>'created_at')::timestamptz DESC
                LIMIT %s'''
if 'ORDER BY (payload' not in content and old in content:
    content = content.replace(old, new)
    with open(f, 'w') as fh:
        fh.write(content)
    print('Patched pgvector.py successfully')
else:
    print('Already patched or pattern not found')
"
