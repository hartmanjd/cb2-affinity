# Original ChEMBL downloads

The existing snapshot `chembl_cb2_20260922T165756862384Z/` is human CB2
(CHEMBL253, UniProt P34972), ChEMBL 37, retrieved September 22, 2026 UTC.
Its six original JSON files preserve 22,523 activities, 1,551 assays, 653 documents,
target/release metadata, and the exact acquisition queries. No raw bytes changed.

Notebook 00 creates a new timestamped download of the latest available API data.
Only completed acquisitions have an acquisition manifest. Notebook 01 uses the
newest completed acquisition automatically. Existing snapshots remain available
while preserved model results reference them.

The [data provenance workbook](../CB2_data_provenance.xlsx) lists the download
version, date, record counts and checksums, and accounts for every downloaded
activity as kept or removed. Non-Ki activities
remain in the source so every exclusion can be accounted for.

The active project does not use BindingDB or IUPHAR/BPS Guide to Pharmacology
extracts. These sources were explored in an earlier separate draft; their origin
sites are https://www.bindingdb.org/ and https://www.guidetopharmacology.org/.
