import pandas as pd
import asyncio
import aiohttp
import logging
import json

# Configure logging
logging.basicConfig(level=logging.DEBUG)  # Set to DEBUG for more verbose output
logger = logging.getLogger(__name__)

# Configuration variables
BASE_URL = 'http://localhost:11434'  # Replace with your actual server URL and port
MODEL_NAME = 'hermes3:8b-llama3.1-q6_K'       # Replace with your model name

async def infer_missing_value(session, row, index, field, base_url=BASE_URL, model_name=MODEL_NAME):
    """
    Asynchronously infers a missing value for a specific field in a DataFrame row using a local LLM based on the 'Description' field.

    Parameters:
    - session: The aiohttp ClientSession used for making HTTP requests.
    - row: pandas Series representing the row with a missing value.
    - index: The index of the row in the DataFrame.
    - field: The name of the missing field to infer.
    - base_url: URL to your local LLM API.
    - model_name: Name of the LLM model to use.

    Returns:
    - A tuple (index, field, inferred_value)
    """
    description = row.get('Description', '')
    if pd.isna(description) or description.strip() == '':
        logger.debug(f"Index {index}: No description available to infer {field}.")
        return index, field, None  # Cannot infer without a description

    # Construct the prompt
    prompt = f"""Extract the {field} from the following employee description, if it is explicitly mentioned. If the {field} is not mentioned, reply with 'Not found', if the filed is AGE be sure to not put years of experience, its different.

for education use only Bachelor's , Master's , PhD , dont use degree

Description:
{description}

Your response should be just the value of {field}, without any additional text.
Your response should be just the value of {field}, without any additional text. 
"""

    payload = {
        'model': model_name,
        'prompt': prompt,
    }

    url = f"{base_url}/api/generate"

    try:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                logger.error(f"Index {index}: Failed to infer {field}. HTTP status code: {response.status}")
                return index, field, None

            content_type = response.headers.get('Content-Type', '')
            if 'application/x-ndjson' in content_type:
                logger.debug(f"Index {index}: Received NDJSON response for {field}.")

                # Read the NDJSON response line by line
                inferred_value = ''
                async for line in response.content:
                    line = line.decode('utf-8')
                    if line:
                        data = json.loads(line)
                        # Depending on the API, extract the relevant field
                        answer = data.get('response', '')
                        if answer:
                            inferred_value += answer
                inferred_value = inferred_value.strip()
            else:
                # Fallback if not NDJSON
                data = await response.json()
                inferred_value = data.get('response', '').strip()

            logger.debug(f"Index {index}: Inferred {field}: {inferred_value}")

            if 'Not found' in inferred_value or inferred_value.lower() == 'not found' or inferred_value == '':
                return index, field, None  # The field was not found in the description
            else:
                return index, field, inferred_value
    except Exception as e:
        logger.error(f"Index {index}: Error inferring {field}: {e}")
        return index, field, None

async def infer_missing_values_in_dataframe(df, fields_to_infer=None, description_field='Description', base_url=BASE_URL, model_name=MODEL_NAME):
    """
    Asynchronously infers missing values in the DataFrame using the local LLM based on the 'Description' field.

    Parameters:
    - df: pandas DataFrame containing the data.
    - fields_to_infer: List of fields (columns) to infer. If None, defaults to predetermined fields.
    - description_field: Name of the field containing the description.
    - base_url: URL to your local LLM API.
    - model_name: Name of the LLM model to use.

    Returns:
    - A DataFrame with missing values inferred where possible.
    """

    if fields_to_infer is None:
        # Default fields to consider (excluding the description itself)
        fields_to_infer = ['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience', 'Salary']

    # Identify rows with missing values, excluding the description field itself
    missing_data_df = df[df.drop(columns=[description_field]).isnull().any(axis=1)]

    logger.debug(f"Total rows with missing values: {len(missing_data_df)}")

    tasks = []
    async with aiohttp.ClientSession() as session:
        for index, row in missing_data_df.iterrows():
            for field in fields_to_infer:
                if pd.isna(row.get(field)):
                    task = asyncio.ensure_future(
                        infer_missing_value(session, row, index, field, base_url, model_name)
                    )
                    tasks.append(task)

        # Gather results concurrently
        results = await asyncio.gather(*tasks)

    # Update the DataFrame with inferred values
    for index, field, inferred_value in results:
        if inferred_value is not None:
            # Handle data type conversion
            if field in ['Age', 'Years of Experience', 'Salary']:
                try:
                    df.at[index, field] = float(inferred_value)
                    logger.debug(f"Index {index}: Updated {field} with value {inferred_value}")
                except ValueError:
                    logger.error(f"Index {index}: Could not convert '{inferred_value}' to a number for field '{field}'.")
            else:
                df.at[index, field] = inferred_value
                logger.debug(f"Index {index}: Updated {field} with value {inferred_value}")
        else:
            logger.debug(f"Index {index}: Could not infer {field}.")

    return df