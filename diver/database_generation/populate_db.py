import os
import argparse
import logging
from sqlalchemy import create_engine
import pandas as pd
from tqdm.contrib.logging import logging_redirect_tqdm
from p_tqdm import p_map

from diver.database_generation.msd_parser import msd_h5_to_df


# Probably should have this in a config
POSTGRES_USER = "diver"
POSTGRES_PASSWORD = "diver"
POSTGRES_DB = "dumpster"
POSTGRES_PORT = 5432
POSTGRES_HOST = "localhost"


if __name__ == "__main__":
    # Setup a quick logger
    logging.basicConfig(level=logging.DEBUG)

    # Parse the arguments
    logging.info("Parsing arguments")
    parser = argparse.ArgumentParser(
        prog="Dump Truck",
        description="Parse the MSD to populate the database"
    )
    parser.add_argument("-f", "--msd-filepath", metavar="FILEPATH", type=str)
    parser.add_argument("-n", "--num-threads", metavar="NUMTHREADS", type=int, choices=range(0, os.cpu_count()))
    parser.add_argument("-t", "--truncate-db", default=False, required=False, action="store_true")

    args = vars(parser.parse_args())

    # Connect to the database
    logging.info("Connecting to the database...")
    try:
        engine = create_engine(
            url="postgresql://{0}:{1}@{2}:{3}/{4}".format(
                POSTGRES_USER,
                POSTGRES_PASSWORD,
                POSTGRES_HOST,
                POSTGRES_PORT,
                POSTGRES_DB
            )
        )
        # Make sure we are connected to the database
        with engine.connect() as conn:
            results = conn.execute("SELECT version()")
        db_version = results.fetchone()
        logging.info("PostgreSQL Version {}".format(db_version))
        logging.info("Connected to the database!")
    except Exception as error:
        logging.error("Encountered an error connecting to the databse!")
        raise error
    
    # Clear the database if we were asked too
    if args["truncate_db"]:
        logging.info("Emptying the dump truck")
        with engine.connect() as conn:
            trans = conn.begin()
            conn.execute("TRUNCATE TABLE msd_table")
            trans.commit()
        logging.info("Truncation completed")

    # Parse the directory for all of the .h5 files
    logging.info("Searching for all .h5 files in {}".format(args["msd_filepath"]))
    h5_files = []
    for root, dirs, files in os.walk(args["msd_filepath"]):
        if len(files) > 0:
            h5_files.extend([os.path.join(root, x) for x in filter(lambda f: f.endswith(".h5"), files)])
    logging.info("Found {} .h5 files".format(len(h5_files)))

    # Read all the h5 files
    logging.info("Parsing the .h5 files")
    with logging_redirect_tqdm():
        dfs = p_map(msd_h5_to_df, h5_files, **{"num_cpus": args["num_threads"]})
    logging.info("Creating a single datafrom")
    df = pd.concat(dfs)
    
    # Writing to the database
    logging.info("Inserting into the database...")
    num_rows = df.to_sql("msd_table", con=engine, if_exists="replace", index=False)
    logging.info("Wrote {} rows".format(num_rows))

    # Close the engine and exit
    engine.dispose()
    logging.info("Finished!")
