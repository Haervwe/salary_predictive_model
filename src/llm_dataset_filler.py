import pandas as pd
import asyncio
import aiohttp
import logging
import json
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Default configuration (using Ollama defaults)
DEFAULT_CONFIG = {
    'base_url': 'http://localhost:11434',
    'model_name': 'hermes3:8b-llama3.1-q6_K',
    'api_key': "",
}

async def infer_missing_value(
    session, 
    row, 
    index, 
    field, 
    base_url: str = DEFAULT_CONFIG['base_url'],
    model_name: str = DEFAULT_CONFIG['model_name'],
    api_key: str = DEFAULT_CONFIG['api_key'],
    debug: bool = False
):
    """
    Asynchronously infers a missing value using OpenAI-compatible endpoints.
    """
    description = row.get('Description', '')
    if pd.isna(description) or description.strip() == '':
        logger.debug(f"Index {index}: No description available to infer {field}.")
        return index, field, None

    prompt = f"""Extract the {field} from the following employee description, if it is explicitly mentioned. If the {field} is not mentioned, reply with 'Not found', if the filed is AGE be sure to not put years of experience, its different.

for education use only Bachelor's , Master's , PhD , dont use degree

Description:
{description}

Your response should be just the value of {field}, without any additional text.
"""

    # OpenAI-compatible payload
    payload = {
        'model': model_name,
        'messages': [
            {'role': 'user', 'content': prompt}
        ],
        'temperature': 0.7
    }

    url = f"{base_url}/v1/chat/completions"
    headers = {'Authorization': f'Bearer {api_key}'} if api_key else {}

    try:
        async with session.post(url, json=payload, headers=headers) as response:
            if response.status != 200:
                logger.error(f"Index {index}: Failed to infer {field}. HTTP status code: {response.status}")
                return index, field, None

            data = await response.json()
            inferred_value = data['choices'][0]['message']['content'].strip()
            
            if debug:
                logger.debug(f"Index {index}: Inferred {field}: {inferred_value}")
            
            if 'Not found' in inferred_value or inferred_value.lower() == 'not found' or inferred_value == '':
                return index, field, None
            else:
                return index, field, inferred_value

    except Exception as e:
        logger.error(f"Index {index}: Error inferring {field}: {e}")
        return index, field, None

async def infer_missing_values_in_dataframe(
    df, 
    fields_to_infer: Optional[list] = None, 
    description_field: str = 'Description',
    base_url: str = DEFAULT_CONFIG['base_url'],
    model_name: str = DEFAULT_CONFIG['model_name'],
    api_key: str = DEFAULT_CONFIG['api_key'],
    debug: bool = False
):
    """
    Asynchronously infers missing values in the DataFrame using OpenAI-compatible endpoints.
    """
    if fields_to_infer is None:
        fields_to_infer = ['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience', 'Salary']

    missing_data_df = df[df.drop(columns=[description_field]).isnull().any(axis=1)]
    if debug:
        logger.debug(f"Total rows with missing values: {len(missing_data_df)}")

    tasks = []
    async with aiohttp.ClientSession() as session:
        for index, row in missing_data_df.iterrows():
            for field in fields_to_infer:
                if pd.isna(row.get(field)):
                    task = asyncio.ensure_future(
                        infer_missing_value(
                            session, row, index, field, 
                            base_url, model_name, api_key, debug
                        )
                    )
                    tasks.append(task)

        results = await asyncio.gather(*tasks)

    for index, field, inferred_value in results:
        if inferred_value is not None:
            if field in ['Age', 'Years of Experience', 'Salary']:
                try:
                    df.at[index, field] = float(inferred_value)
                    if debug:
                        logger.debug(f"Index {index}: Updated {field} with value {inferred_value}")
                except ValueError:
                    logger.error(f"Index {index}: Could not convert '{inferred_value}' to a number for field '{field}'.")
            else:
                df.at[index, field] = inferred_value
                if debug:
                    logger.debug(f"Index {index}: Updated {field} with value {inferred_value}")
        else:
            if debug:
                logger.debug(f"Index {index}: Could not infer {field}.")

    return df