from typing import Iterable
import logging
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

class DataSource:
    def schema(self) -> pa.Schema: raise NotImplementedError
    def iter_tables(self) -> Iterable[pa.Table]: raise NotImplementedError

class GeoParquetSource(DataSource):
    def __init__(self, path: str):
        self._pf = pq.ParquetFile(path)
        self._schema = self._pf.schema_arrow
        self._num_row_groups = self._pf.num_row_groups
        logger.info(f"GeoParquetSource opened {path} with {self._num_row_groups} row groups")

    def schema(self) -> pa.Schema:
        logger.info(f"Source schema: {self._schema.metadata}")  
        return self._schema

    def iter_tables(self) -> Iterable[pa.Table]:
        for i in range(self._num_row_groups):
            logger.debug(f"Reading row group {i}/{self._num_row_groups}")
            yield self._pf.read_row_group(i)
