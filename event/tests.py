from unittest.mock import patch

from django.test import SimpleTestCase, override_settings

from .api_views import _extract_search_embeddings


class SearchEmbeddingExtractorTests(SimpleTestCase):
    @override_settings(USE_RUNPOD=True)
    @patch('event.face_utils.extract_embeddings_runpod')
    @patch('event.face_utils.extract_embeddings')
    def test_search_uses_runpod_when_enabled(self, local_extract, runpod_extract):
        runpod_extract.return_value = [[['embedding']]]

        result = _extract_search_embeddings('/tmp/query.jpg')

        self.assertEqual(result, [['embedding']])
        runpod_extract.assert_called_once_with(['/tmp/query.jpg'])
        local_extract.assert_not_called()

    @override_settings(USE_RUNPOD=False)
    @patch('event.face_utils.extract_embeddings')
    def test_search_uses_local_extractor_when_runpod_disabled(self, local_extract):
        local_extract.return_value = [['embedding']]

        result = _extract_search_embeddings('/tmp/query.jpg')

        self.assertEqual(result, [['embedding']])
        local_extract.assert_called_once_with('/tmp/query.jpg')

    @override_settings(USE_RUNPOD=True)
    @patch('event.face_utils.extract_embeddings_runpod')
    def test_search_rejects_unexpected_runpod_result_count(self, runpod_extract):
        runpod_extract.return_value = [[], []]

        with self.assertRaisesMessage(RuntimeError, '2 result(s) for 1 image'):
            _extract_search_embeddings('/tmp/query.jpg')
