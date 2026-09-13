from pathlib import Path
from pypdf import PdfReader, PdfWriter

source = Path("documents/qualification/Carta_NU.pdf")
target = Path("documents/qualification/Carta_NU_normalized.pdf")

reader = PdfReader(source)
writer = PdfWriter()

for page in reader.pages:
    # Force inherited page geometry to be resolved
    _ = page.mediabox
    writer.add_page(page)

with target.open("wb") as output:
    writer.write(output)

print("Created:", target)
print("Pages:", len(reader.pages))
