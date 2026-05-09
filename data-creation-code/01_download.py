import urllib.request
import zipfile
import os
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logging/data_creation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def download_movielens(output_dir="../raw_data"):
    # Function to download and extract the MovieLens 1M dataset.
    # Returns the filepath where the data was downloaded.

    try:
        url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
        logger.info(f"Starting MovieLens dataset download from {url}")
        
        # Create and log output file path
        script_dir = Path(__file__).parent.absolute()
        output_path = script_dir / output_dir
        
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_path}")
        
        zip_path = output_path / "ml-1m.zip"
        
        # Attempt to download the file
        try:
            logger.info("Downloading file...")
            urllib.request.urlretrieve(url, zip_path)
            logger.info(f"Successfully downloaded to {zip_path}")
        except Exception as e:
            logger.error(f"Unexpected error during download: {e}")
        
        # Verify file exists and has content
        if not zip_path.exists() or zip_path.stat().st_size == 0:
            logger.error("Downloaded file is empty or missing.")
            return
        
        logger.info(f"File size: {zip_path.stat().st_size / (1024*1024):.2f} MB")
        
        # Attempt to extract the zip file
        try:
            logger.info("Extracting files...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_path)
            logger.info("Extraction completed successfully")
        except Exception as e:
            logger.error(f"Error during extraction: {e}")
            return
        
        logger.info(f"Dataset successfully downloaded. File ready at: {output_path / 'ml-1m'}")
        return str(output_path / 'ml-1m')
        
    except Exception as e:
        logger.error(f"Error in download process: {e}")
        return

if __name__ == "__main__":
    # Execute the download function and log the result
    try:
        data_path = download_movielens()
        logger.info(f"Download complete! Data path: {data_path}")
    except Exception as e:
        logger.error(f"Download failed: {e}")