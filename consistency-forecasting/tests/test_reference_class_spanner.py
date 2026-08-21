"""
Test individual outputs and overall pipline for NewsAPI -> FQs
"""

import sys
from common.path_utils import get_src_path

sys.path.append(str(get_src_path()))

import pytest
from dotenv import load_dotenv
import asyncio
from datetime import datetime
from common.datatypes import ForecastingQuestion
from generate_fqs_using_reference_class import BinaryFQReferenceClassSpanner

load_dotenv()

sample_forecasting_question_dicts = [
    # TODO - add support for spanning the reference classes of questions that have their resolutions between a date such as between 2024 and 2030.
    {
        "id": "5d8a3198-dd56-4a7b-ac3b-b464cadc94e4",
        "title": "Will the United Kingdom have a new Prime Minister by January 1, 2028?",
        "body": "This question will resolve as Yes if, by January 1, 2028, an individual other than the Prime Minister in office as of July 1, 2024 is officially serving as the Prime Minister of the United Kingdom. The change in leadership must be confirmed by official announcements or credible news reports.",
        "resolution_date": "2028-01-01T00:00:00Z",
        "question_type": "binary",
        "data_source": "synthetic",
        "url": None,
        "metadata": None,
        "resolution": None,
    },
    {
        "id": "114dda02-c4ca-432b-99a3-0d6687c9a55e",
        "title": "Will Germany have a new Chancellor by December 31, 2028?",
        "body": "This question will resolve as Yes if, by December 31, 2028, an individual other than the Chancellor in office as of July 1, 2024 is officially serving as the Chancellor of Germany. The change in leadership must be confirmed by official announcements or credible news reports.",
        "resolution_date": "2028-12-31T00:00:00Z",
        "question_type": "binary",
        "data_source": "synthetic",
        "url": None,
        "metadata": None,
        "resolution": None,
    },
]


@pytest.mark.asyncio
async def test_reference_class_spanned_questions():
    num_spanned_questions = 10
    tasks = []
    for source_fq_dict in sample_forecasting_question_dicts:
        source_fq = ForecastingQuestion(**source_fq_dict)
        tasks.append(
            BinaryFQReferenceClassSpanner.generate_spanned_fqs(
                source_fq,
                "openai/gpt-5.4-mini",
                num_spanned_questions,
                datetime(2024, 7, 1),
                "basic",
            )
        )

    results = await asyncio.gather(*tasks)

    print(sample_forecasting_question_dicts)
    print(results)

    for result in results:
        assert (
            len(result) >= 3
        )  # sanity check for verification module.


if __name__ == "__main__":
    asyncio.run(pytest.main([__file__]))
