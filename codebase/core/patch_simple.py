from .cache_simple import DynamicCache as newCache
import transformers.cache_utils

transformers.cache_utils.DynamicCache = newCache