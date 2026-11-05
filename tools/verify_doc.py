import docx

doc = docx.Document(r'c:\Users\capar\Downloads\Group10_AIDeclaration-1.docx')
print('=== PARAGRAPHS ===')
for i, p in enumerate(doc.paragraphs):
    print(f'{i}: {p.text}')

print('\n=== TABLES ===')
for t_idx, table in enumerate(doc.tables):
    print(f'\nTable {t_idx}: {len(table.rows)} rows, {len(table.columns)} cols')
    for r_idx, row in enumerate(table.rows):
        print(f'  Row {r_idx}: {[c.text.replace(chr(10), " ")[:60] for c in row.cells]}')
