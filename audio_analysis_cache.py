import os
import json
import hashlib
import pickle
import logging
import numpy as np

logger = logging.getLogger("AnalysisCache")

class AudioAnalysisCache:
    """Cache system for storing and retrieving audio analysis results"""
    
    def __init__(self, cache_dir="analysis_cache"):
        """Initialize the cache system
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(cache_dir):
            try:
                os.makedirs(cache_dir)
                logger.info(f"Created cache directory: {cache_dir}")
            except Exception as e:
                logger.error(f"Failed to create cache directory: {e}")
    
    def _get_file_hash(self, file_path):
        """Generate a hash for the file to use as a cache key
        
        Args:
            file_path: Path to the audio file
        
        Returns:
            str: Hash of the file
        """
        # Get file metadata
        file_stat = os.stat(file_path)
        file_size = file_stat.st_size
        file_mtime = file_stat.st_mtime
        
        # Create a hash using the file path, size, and modification time
        # This avoids having to read the entire file content
        hasher = hashlib.md5()
        hasher.update(file_path.encode())
        hasher.update(str(file_size).encode())
        hasher.update(str(file_mtime).encode())
        
        return hasher.hexdigest()
    
    def _get_cache_path(self, file_path):
        """Get the cache file path for an audio file
        
        Args:
            file_path: Path to the audio file
        
        Returns:
            str: Path to the cache file
        """
        file_hash = self._get_file_hash(file_path)
        file_name = os.path.basename(file_path)
        base_name = os.path.splitext(file_name)[0]
        
        # Create a cache file name using the base name and hash
        cache_name = f"{base_name}_{file_hash}.pkl"
        return os.path.join(self.cache_dir, cache_name)
    
    def has_cache(self, file_path):
        """Check if analysis results are cached for a file
        
        Args:
            file_path: Path to the audio file
        
        Returns:
            bool: True if cached results exist, False otherwise
        """
        cache_path = self._get_cache_path(file_path)
        return os.path.exists(cache_path)
    
    def save_analysis(self, file_path, features):
        """Save analysis results to cache
        
        Args:
            file_path: Path to the original audio file
            features: Dictionary of audio features
        
        Returns:
            bool: True if saved successfully, False otherwise
        """
        cache_path = self._get_cache_path(file_path)
        
        try:
            # Convert numpy arrays to lists for serialization
            serializable_features = {}
            for key, value in features.items():
                if isinstance(value, np.ndarray):
                    serializable_features[key] = value.tolist()
                elif isinstance(value, dict):
                    serializable_features[key] = {}
                    for k, v in value.items():
                        if isinstance(v, np.ndarray):
                            serializable_features[key][k] = v.tolist()
                        else:
                            serializable_features[key][k] = v
                else:
                    serializable_features[key] = value
            
            # Save to file
            with open(cache_path, 'wb') as f:
                pickle.dump(serializable_features, f)
            
            logger.info(f"Saved analysis to cache: {cache_path}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to save analysis to cache: {e}")
            return False
    
    def load_analysis(self, file_path):
        """Load analysis results from cache
        
        Args:
            file_path: Path to the audio file
        
        Returns:
            dict: Dictionary of audio features or None if not found
        """
        cache_path = self._get_cache_path(file_path)
        
        if not os.path.exists(cache_path):
            logger.warning(f"No cache found for: {file_path}")
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                serialized_features = pickle.load(f)
            
            # Convert lists back to numpy arrays
            features = {}
            for key, value in serialized_features.items():
                if isinstance(value, list):
                    features[key] = np.array(value)
                elif isinstance(value, dict):
                    features[key] = {}
                    for k, v in value.items():
                        if isinstance(v, list):
                            features[key][k] = np.array(v)
                        else:
                            features[key][k] = v
                else:
                    features[key] = value
            
            logger.info(f"Loaded analysis from cache: {cache_path}")
            return features
        
        except Exception as e:
            logger.error(f"Failed to load analysis from cache: {e}")
            return None
    
    def clear_cache(self, file_path=None):
        """Clear cache for a specific file or all files
        
        Args:
            file_path: Path to the audio file or None to clear all
        
        Returns:
            int: Number of cache files removed
        """
        if file_path:
            # Clear cache for a specific file
            cache_path = self._get_cache_path(file_path)
            if os.path.exists(cache_path):
                os.remove(cache_path)
                logger.info(f"Cleared cache for: {file_path}")
                return 1
            return 0
        else:
            # Clear all cache files
            count = 0
            for file_name in os.listdir(self.cache_dir):
                if file_name.endswith(".pkl"):
                    os.remove(os.path.join(self.cache_dir, file_name))
                    count += 1
            
            logger.info(f"Cleared {count} cache files")
            return count
