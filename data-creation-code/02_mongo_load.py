import pandas as pd
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from dotenv import load_dotenv
import os
import logging
import sys
from pathlib import Path
from datetime import datetime

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

def load_movielens_to_mongo(raw_data_path="../raw_data/ml-1m"):
    # Load MovieLens 1M data into MongoDB, with documents representing users.
    # Each document represents one user and has the following "structure":
    
    #{
    #    user_id: int,
    #    demographics: {gender, age_group, occupation, zip_code},
    #    ratings: [{movie_id, title, genres, rating, timestamp}],
    #    num_ratings: int,
    #    avg_rating: float
    #}
    
    # Convert to absolute path relative to script location
    script_dir = Path(__file__).parent.absolute()
    data_dir = script_dir / raw_data_path
    
    logger.info("\nStarting MovieLens data load process...")
    logger.info(f"Data directory: {data_dir}\n")
    
    client = None
    
    try:
        # Verify data files exist
        logger.info("Checking for data files")
        required_files = ['users.dat', 'ratings.dat', 'movies.dat']
        for file in required_files:
            file_path = data_dir / file
            if not file_path.exists():
                logger.error(f"Required file not found: {file_path}")
                return
            logger.info(f"  Found {file}")
        
        # Load CSV files into pandas dataframes
        logger.info("Loading CSV files into pandas DataFrames")
        
        try:
            logger.info("  Loading users.dat...")
            users = pd.read_csv(
                data_dir / 'users.dat', 
                sep='::', 
                engine='python', 
                header=None,
                names=['user_id', 'gender', 'age', 'occupation', 'zip_code'],
                encoding='latin-1'
            )
            logger.info(f"    Loaded {len(users)} users")
        except Exception as e:
            logger.error(f"Failed to load users.dat: {e}")
            return
        
        try:
            logger.info("  Loading ratings.dat...")
            ratings = pd.read_csv(
                data_dir / 'ratings.dat', 
                sep='::',
                engine='python', 
                header=None,
                names=['user_id', 'movie_id', 'rating', 'timestamp'],
                encoding='latin-1'
            )
            logger.info(f"    Loaded {len(ratings)} ratings")
        except Exception as e:
            logger.error(f"Failed to load ratings.dat: {e}")
            return
        
        try:
            logger.info("  Loading movies.dat...")
            movies = pd.read_csv(
                data_dir / 'movies.dat', 
                sep='::',
                engine='python', 
                header=None, 
                encoding='latin-1',
                names=['movie_id', 'title', 'genres']
            )
            logger.info(f"    Loaded {len(movies)} movies")
        except Exception as e:
            logger.error(f"Failed to load movies.dat: {e}")
            return
        
        # Connect to MongoDB
        logger.info("Connecting to MongoDB Atlas")
        
        # Load environment variables
        load_dotenv()
        mongo_uri = os.getenv("MONGO_URI")
        db_name = os.getenv("DB_NAME")
        
        if not mongo_uri:
            logger.error("MONGO_URI not found in .env file")
            return
        if not db_name:
            logger.error("DB_NAME not found in .env file")
            return
        
        try:
            client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
            # Test connection
            client.admin.command('ping')
            logger.info("  Connected to MongoDB Atlas")
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            return
        
        db = client[db_name]
        collection = db["users"]
        
        # Clear existing data if it exists
        logger.info("Preparing collection for new data")
        existing_count = collection.count_documents({})
        if existing_count > 0:
            logger.info(f"  Found {existing_count} existing documents")
            result = collection.delete_many({})
            logger.info(f"  Cleared {result.deleted_count} documents from collection")
        else:
            logger.info("  Collection is empty, ready for insertion")
        
        # Transform data into document format
        logger.info("Transforming data into document format")
        
        # Set age mapping buckets 
        age_map = {
            1: "Under 18", 
            18: "18-24", 
            25: "25-34", 
            35: "35-44", 
            45: "45-49", 
            50: "50-55", 
            56: "56+"
        }
        
        # Group ratings by user
        ratings_by_user = ratings.groupby('user_id')
        logger.info("  Grouped ratings by user_id")
        
        user_documents = []
        total_ratings_processed = 0
        error_users = []
        
        # Process each user
        for idx, user_id in enumerate(users['user_id'].unique()):
            try:
                # Print progress every 1000 users
                if idx % 1000 == 0 and idx > 0:
                    logger.info(f"  Processed {idx} of {len(users)} users...")
                
                # Get user's ratings and build ratings array
                user_ratings = ratings_by_user.get_group(user_id)
                
                ratings_array = []
                for _, row in user_ratings.iterrows():
                    # Get movie details and append to ratings array
                    movie = movies[movies['movie_id'] == row['movie_id']]
                    if len(movie) > 0:
                        movie = movie.iloc[0]
                        ratings_array.append({
                            'movie_id': int(row['movie_id']),
                            'title': movie['title'],
                            'genres': movie['genres'].split('|'),
                            'rating': float(row['rating']),
                            'timestamp': int(row['timestamp'])
                        })
                    else:
                        # Log warning and continue if movie not found
                        logger.warning(f"  Movie ID {row['movie_id']} not found for user {user_id}")
                
                # Get user demographics and create document
                user_info = users[users['user_id'] == user_id].iloc[0]
                
                doc = {
                    'user_id': int(user_id),
                    'demographics': {
                        'gender': user_info['gender'],
                        'age_group': age_map.get(user_info['age'], "Unknown"),
                        'occupation': int(user_info['occupation']),
                        'zip_code': str(user_info['zip_code'])
                    },
                    'ratings': ratings_array,
                    'num_ratings': len(ratings_array),
                    'avg_rating': sum(r['rating'] for r in ratings_array) / len(ratings_array) if ratings_array else 0,
                    'created_at': datetime.now()
                }
                
                user_documents.append(doc)
                total_ratings_processed += len(ratings_array)
                
            except Exception as e:
                logger.error(f"  Error processing user {user_id}: {e}")
                error_users.append(user_id)
                continue
        
        logger.info(f"  Successfully processed {len(user_documents)} users")
        logger.info(f"  Total ratings processed: {total_ratings_processed:,}")
        
        if error_users:
            logger.warning(f"  Failed to process {len(error_users)} users: {error_users[:10]}")
        
        # Begin inserting into MongoDB
        logger.info("Inserting documents into MongoDB")
        
        if user_documents:
            try:
                result = collection.insert_many(user_documents, ordered=False)
                logger.info(f"  Successfully inserted {len(result.inserted_ids)} documents")
                logger.info(f"  Total ratings stored: {total_ratings_processed:,}")
                logger.info(f"  Collection now has {collection.count_documents({})} documents")
            except Exception as e:
                logger.error(f"Failed to insert documents: {e}")
                raise
        else:
            logger.error("No documents were created from the data, data list after transformation was empty!")
            return
                
        # Verify insertion
        logger.info("Verifying insertion")
        final_count = collection.count_documents({})
        if final_count == len(user_documents):
            logger.info(f"  Verification passed: {final_count} documents in database")
        else:
            logger.warning(f"  Verification mismatch: Expected {len(user_documents)}, found {final_count}")
        
        logger.info("\n MovieLens data successfully loaded into MongoDB")
        logger.info(f"   Users: {final_count:,}")
        logger.info(f"   Ratings: {total_ratings_processed:,}")
        
        return final_count
        
    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        logger.error("   Run download.py first to get the dataset")
        return
        
    except ConnectionFailure as e:
        logger.error(f"Database connection error: {e}")
        logger.error("   Check your internet connection and MongoDB Atlas credentials")
        return
        
    except Exception as e:
        logger.error(f"Unexpected error during load process: {e}")
        return
        
    finally:
        # Close the connection
        if client:
            client.close()
            logger.info("MongoDB connection closed")

if __name__ == "__main__":
    # Execute the load function to load MovieLens data into MongoDB
    # Log the result
    try:
        num_docs = load_movielens_to_mongo()
        logger.info(f"Script completed successfully with {num_docs} documents")
    except Exception as e:
        logger.error(f"Script failed: {e}")