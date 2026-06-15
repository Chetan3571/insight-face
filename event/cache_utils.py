import logging

from django.core.cache import cache

logger = logging.getLogger('event')


def invalidate_search_cache() -> None:
    """Clear cached face-search results after embeddings change."""
    try:
        if hasattr(cache, 'delete_pattern'):
            cache.delete_pattern('search:*')
            logger.info('Search cache invalidated')
    except Exception as exc:
        logger.warning('Search cache invalidation skipped: %s', exc)
