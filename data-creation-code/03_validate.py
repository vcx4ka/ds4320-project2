from pymongo import MongoClient
from dotenv import load_dotenv
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


def get_mongo_connection():
    # Establishes a connection to MongoDB, returns a MongoDB client

    try:
        # Get variables from .env, use to connect to MongoDB
        mongo_uri = os.getenv("MONGO_URI")
        db_name = os.getenv("DB_NAME")
        
        if not mongo_uri:
            logger.error("MONGO_URI not found in environment variables")
            return
        if not db_name:
            logger.error("DB_NAME not found in environment variables")
            return
        
        logger.info(f"Connecting to MongoDB database: {db_name}")
        client = MongoClient(mongo_uri)
        
        # Test connection
        client.admin.command('ping')
        logger.info("MongoDB connection successful")
        
        return client, client[db_name]
        
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        return

def validate_and_clean():
    # Perform data quality checks on the MongoDB collection and log results.

    client = None
    try:
        # Connect to MongoDB
        client, db = get_mongo_connection()
        collection = db["users"]
        
        # Check count of users in collection
        total_users = collection.count_documents({})
        logger.info(f"Total users in collection: {total_users}")
        
        if total_users == 0:
            logger.warning("Collection is empty! Run mongo_load.py first.")
            return {"status": "empty"}
        
        validation_results = {
            "total_users": total_users,
            "status": "success"
        }
        
        # Check for users with no ratings
        no_ratings = collection.count_documents({"num_ratings": 0})
        logger.info(f"Users with 0 ratings: {no_ratings}")
        validation_results["users_without_ratings"] = no_ratings
        
        # Check for missing demographics
        missing_demo = collection.count_documents({
            "$or": [
                {"demographics.gender": {"$exists": False}},
                {"demographics.age_group": {"$exists": False}},
                {"demographics.occupation": {"$exists": False}}
            ]
        })
        logger.info(f"Users missing demographics: {missing_demo}")
        validation_results["users_missing_demographics"] = missing_demo
        
        # Check rating distribution
        try:
            pipeline = [
                {"$unwind": "$ratings"},
                {"$group": {
                    "_id": None,
                    "avg_rating": {"$avg": "$ratings.rating"},
                    "min_rating": {"$min": "$ratings.rating"},
                    "max_rating": {"$max": "$ratings.rating"},
                    "total_ratings": {"$sum": 1}
                }}
            ]
            stats = list(collection.aggregate(pipeline))
            
            if stats:
                validation_results["global_avg_rating"] = stats[0]['avg_rating']
                validation_results["rating_range"] = (stats[0]['min_rating'], stats[0]['max_rating'])
                validation_results["total_ratings"] = stats[0]['total_ratings']
                logger.info(f"Global average rating: {stats[0]['avg_rating']:.2f}")
                logger.info(f"Rating range: {stats[0]['min_rating']} - {stats[0]['max_rating']}")
                logger.info(f"Total ratings in database: {stats[0]['total_ratings']:,}")
        except Exception as e:
            logger.warning(f"Could not compute rating statistics: {e}")
        
        # Sample a document for validation
        sample = collection.find_one()
        if sample:
            logger.info("Sample document structure validated")
            required_fields = ['user_id', 'demographics', 'ratings', 'num_ratings', 'avg_rating']
            missing_fields = [f for f in required_fields if f not in sample]
            if missing_fields:
                logger.warning(f"Sample document missing fields: {missing_fields}")
                validation_results["missing_fields"] = missing_fields
            else:
                logger.info("Document schema validation passed")
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return {"status": "error", "message": str(e)}
        
    finally:
        if client:
            client.close()
            logger.info("MongoDB connection closed")

if __name__ == "__main__":
    # Execute the validation function and log the results.

    try:
        # Load environment variables
        load_dotenv()
        logger.info("Environment variables loaded successfully")

        logger.info("\nStarting data validation process...\n")
        results = validate_and_clean()
        
        if results.get("status") == "success":
            logger.info("Data validation completed successfully")
            logger.info(f"Summary: {results}")
        elif results.get("status") == "empty":
            logger.warning("Database is empty. Run mongo_load.py first.")
        else:
            logger.error("Data validation failed")

    except Exception as e:
        logger.error(f"Failed to load .env file: {e}")