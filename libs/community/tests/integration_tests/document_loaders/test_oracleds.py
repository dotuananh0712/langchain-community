# Authors:
#   Sudhir Kumar (sudhirkk)
#
# -----------------------------------------------------------------------------
# test_oracleds.py
# -----------------------------------------------------------------------------
import sys

from langchain_community.document_loaders.oracleai import (
    OracleDocLoader,
    OracleTextSplitter,
)
from langchain_community.vectorstores.oraclevs import (
    _table_exists,
    drop_table_purge,
)

uname = "hr"
passwd = "hr"
# uname = "LANGCHAINUSER"
# passwd = "langchainuser"
v_dsn = "100.70.107.245:1521/cdb1_pdb1.regress.rdbms.dev.us.oracle.com"


### Test loader #####
def test_loader_test() -> None:
    try:
        import oracledb
    except ImportError:
        return

    try:
        # oracle connection
        connection = oracledb.connect(user=uname, password=passwd, dsn=v_dsn)
        cursor = connection.cursor()

        if _table_exists(connection, "LANGCHAIN_DEMO"):
            drop_table_purge(connection, "LANGCHAIN_DEMO")

        cursor.execute("CREATE TABLE langchain_demo(id number, text varchar2(25))")

        rows = [
            (1, "First"),
            (2, "Second"),
            (3, "Third"),
            (4, "Fourth"),
            (5, "Fifth"),
            (6, "Sixth"),
            (7, "Seventh"),
        ]

        cursor.executemany("insert into LANGCHAIN_DEMO(id, text) values (:1, :2)", rows)

        connection.commit()

        # local file, local directory, database  column
        loader_params = {
            "owner": uname,
            "tablename": "LANGCHAIN_DEMO",
            "colname": "TEXT",
        }

        # instantiate
        loader = OracleDocLoader(conn=connection, params=loader_params)

        # load
        docs = loader.load()

        # verify
        if len(docs) == 0:
            sys.exit(1)

        if _table_exists(connection, "LANGCHAIN_DEMO"):
            drop_table_purge(connection, "LANGCHAIN_DEMO")

    except Exception:
        sys.exit(1)

    try:
        # expectation : ORA-00942
        loader_params = {
            "owner": uname,
            "tablename": "COUNTRIES1",
            "colname": "COUNTRY_NAME",
        }

        # instantiate
        loader = OracleDocLoader(conn=connection, params=loader_params)

        # load
        docs = loader.load()
        if len(docs) == 0:
            pass

    except Exception:
        pass

    try:
        # expectation : file "SUDHIR" doesn't exist.
        loader_params = {"file": "SUDHIR"}

        # instantiate
        loader = OracleDocLoader(conn=connection, params=loader_params)

        # load
        docs = loader.load()
        if len(docs) == 0:
            pass

    except Exception:
        pass

    try:
        # expectation : path "SUDHIR" doesn't exist.
        loader_params = {"dir": "SUDHIR"}

        # instantiate
        loader = OracleDocLoader(conn=connection, params=loader_params)

        # load
        docs = loader.load()
        if len(docs) == 0:
            pass

    except Exception:
        pass


### Test splitter ####
def test_splitter_test() -> None:
    try:
        import oracledb
    except ImportError:
        return

    try:
        # oracle connection
        connection = oracledb.connect(user=uname, password=passwd, dsn=v_dsn)
        doc = """Langchain is a wonderful framework to load, split, chunk
                and embed your data!!"""

        # by words , max = 1000
        splitter_params = {
            "by": "words",
            "max": "1000",
            "overlap": "200",
            "split": "custom",
            "custom_list": [","],
            "extended": "true",
            "normalize": "all",
        }

        # instantiate
        splitter = OracleTextSplitter(conn=connection, params=splitter_params)

        # generate chunks
        chunks = splitter.split_text(doc)

        # verify
        if len(chunks) == 0:
            sys.exit(1)

        # by chars , max = 4000
        splitter_params = {
            "by": "chars",
            "max": "4000",
            "overlap": "800",
            "split": "NEWLINE",
            "normalize": "all",
        }

        # instantiate
        splitter = OracleTextSplitter(conn=connection, params=splitter_params)

        # generate chunks
        chunks = splitter.split_text(doc)

        # verify
        if len(chunks) == 0:
            sys.exit(1)

        # by words , max = 10
        splitter_params = {
            "by": "words",
            "max": "10",
            "overlap": "2",
            "split": "SENTENCE",
        }

        # instantiate
        splitter = OracleTextSplitter(conn=connection, params=splitter_params)

        # generate chunks
        chunks = splitter.split_text(doc)

        # verify
        if len(chunks) == 0:
            sys.exit(1)

        # by chars , max = 50
        splitter_params = {
            "by": "chars",
            "max": "50",
            "overlap": "10",
            "split": "SPACE",
            "normalize": "all",
        }

        # instantiate
        splitter = OracleTextSplitter(conn=connection, params=splitter_params)

        # generate chunks
        chunks = splitter.split_text(doc)

        # verify
        if len(chunks) == 0:
            sys.exit(1)

    except Exception:
        sys.exit(1)

    try:
        # ORA-20003: invalid value xyz for BY parameter
        splitter_params = {"by": "xyz"}

        # instantiate
        splitter = OracleTextSplitter(conn=connection, params=splitter_params)

        # generate chunks
        chunks = splitter.split_text(doc)

        # verify
        if len(chunks) == 0:
            pass

    except Exception:
        pass

    try:
        # Expectation: ORA-30584: invalid text chunking MAXIMUM - '10'
        splitter_params = {
            "by": "chars",
            "max": "10",
            "overlap": "2",
            "split": "SPACE",
            "normalize": "all",
        }

        # instantiate
        splitter = OracleTextSplitter(conn=connection, params=splitter_params)

        # generate chunks
        chunks = splitter.split_text(doc)

        # verify
        if len(chunks) == 0:
            pass

    except Exception:
        pass

    try:
        # Expectation: ORA-30584: invalid text chunking MAXIMUM - '5'
        splitter_params = {
            "by": "words",
            "max": "5",
            "overlap": "2",
            "split": "SPACE",
            "normalize": "all",
        }

        # instantiate
        splitter = OracleTextSplitter(conn=connection, params=splitter_params)

        # generate chunks
        chunks = splitter.split_text(doc)

        # verify
        if len(chunks) == 0:
            pass

    except Exception:
        pass

    try:
        # Expectation: ORA-30586: invalid text chunking SPLIT BY - SENTENCE
        splitter_params = {
            "by": "words",
            "max": "50",
            "overlap": "2",
            "split": "SENTENCE",
            "normalize": "all",
        }

        # instantiate
        splitter = OracleTextSplitter(conn=connection, params=splitter_params)

        # generate chunks
        chunks = splitter.split_text(doc)

        # verify
        if len(chunks) == 0:
            pass

    except Exception:
        pass
