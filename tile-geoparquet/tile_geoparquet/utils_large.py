from __future__ import annotations
import pyarrow as pa
from pyarrow import types as pat

def ensure_large_types(tbl: pa.Table, geom_col: str) -> pa.Table:
    """
    Upgrade narrow variable-width types to their 64-bit offset 'large_*' versions
    to avoid Arrow 'offset overflow' during take/concat/write on big batches.
      - WKB column (geometry) -> large_binary
      - string columns        -> large_string
    """
    schema = tbl.schema
    new_fields = []
    changed = False

    for f in schema:
        t = f.type
        nf = f
        if f.name == geom_col and pat.is_binary(t):
            nf = pa.field(f.name, pa.large_binary(), nullable=f.nullable, metadata=f.metadata)
            changed = True
        elif pat.is_string(t):
            nf = pa.field(f.name, pa.large_string(), nullable=f.nullable, metadata=f.metadata)
            changed = True
        new_fields.append(nf)

    if not changed:
        return tbl
    return tbl.cast(pa.schema(new_fields), safe=False)