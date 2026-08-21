# LLM Utils
# Run tests in this file with python -m common.llm_utils

# %%
import os
import logging
from typing import Coroutine, Optional, List, Literal
from openai import AsyncOpenAI, OpenAI
import instructor
from instructor import Instructor
from instructor.mode import Mode
import asyncio
from pydantic import BaseModel
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from dotenv import load_dotenv, dotenv_values
# from mistralai.models.chat_completion import ChatMessage
from anthropic import AsyncAnthropic, Anthropic
import logfire
from costly import CostlyResponse, costly, Costlog
from costly.simulators.llm_simulator_faker import LLM_Simulator_Faker
from .datatypes import (
    PlainText,
    Prob,
    Forecast,
    ForecastingQuestion,
    ForecastingQuestion_stripped,
    Consequence_ClassifyOutput,
    Consequence_ConsequenceType,
)
from .path_utils import get_src_path, get_root_path, get_data_path, get_logs_path
from .perplexity_client import AsyncPerplexityClient, SyncPerplexityClient

from .perscache import (
    Cache,
    JSONPydanticResponseSerializer,
    JSONSerializer,
    PickleSerializer,
    RedisStorage,
    LocalFileStorage,
    ValueWrapperDictInspectArgs,
)  # If no redis, use LocalFileStorage

GLOBAL_COST_LOG = Costlog(mode="jsonl", discard_extras=True)
SIMULATE = os.getenv("SIMULATE", "False").lower() == "true"
DISABLE_COSTLY = os.getenv("DISABLE_COSTLY", "False").lower() == "true"

CACHE_FLAGS = ["NO_CACHE", "NO_READ_CACHE", "NO_WRITE_CACHE", "LOCAL_CACHE"]
print(f"LOCAL_CACHE: {os.getenv('LOCAL_CACHE')}")

load_dotenv(override=False, dotenv_path=get_root_path() / ".env")

# We override all keys and tokens (bc those could have been set globally in the user's system). Other flags stay if they are set.
env_path = get_root_path() / ".env"
env_vars = dotenv_values(env_path)
KEYS = [k for k in env_vars.keys() if "KEY" in k or "TOKEN" in k]
override_env_vars = {k: v for k, v in env_vars.items() if k in KEYS}
os.environ.update(override_env_vars)

max_concurrent_queries = int(os.getenv("MAX_CONCURRENT_QUERIES", 25))
print(f"max_concurrent_queries set for global semaphore: {max_concurrent_queries}")

if os.getenv("OPENAI_JSON_STRICT") == "True":
    raise ValueError(
        "OPENAI_JSON_STRICT is set to True. There is an ongoing issue with this flag, so please set it to False."
    )
## All logging settings here
if os.getenv("USE_LOGFIRE") == "True":
    print("Setting up Pydantic Logfire")

    def scrubbing_callback(m: logfire.ScrubMatch):
        """
        Need to disable some security measures of logfire.
        Those trigges depending on whether some substrings like "auth" are present as param *values*;
        and our param values are *prompts* and such, so no need to scrub them.
        """
        if m.pattern_match.group(0) == "auth":
            return m.value

    logfire.configure(
        pydantic_plugin=logfire.PydanticPlugin(record="failure"),
        scrubbing=logfire.ScrubbingOptions(callback=scrubbing_callback),
    )

if os.getenv("LOGGING_DEBUG") == "True":
    print("Setting logging level to DEBUG")
    logging.basicConfig(level=logging.DEBUG, force=True)


logs_dir = get_logs_path()
if not logs_dir.exists():
    logs_dir.mkdir(parents=True)
    print(f"Created generic logs directory at {logs_dir}")
else:
    print(f"Generic logs directory already exists at {logs_dir}")


def reset_global_semaphore():
    """
    Use if your code uses asyncio.run()
    """
    global global_llm_semaphore
    global_llm_semaphore = asyncio.Semaphore(max_concurrent_queries)
    print(
        f"Resetting global semaphore, max concurrent queries: {max_concurrent_queries}"
    )


reset_global_semaphore()


pydantic_cache = Cache(
    serializer=JSONPydanticResponseSerializer(),
    storage=(
        LocalFileStorage(location=get_src_path().parent / os.getenv("LOCAL_CACHE"))
        if os.getenv("LOCAL_CACHE")
        else RedisStorage(namespace="llm_utils")
    ),
    value_wrapper=ValueWrapperDictInspectArgs(),
)

embeddings_cache = Cache(
    serializer=PickleSerializer(),
    storage=(
        LocalFileStorage(location=get_src_path().parent / os.getenv("LOCAL_CACHE"))
        if os.getenv("LOCAL_CACHE")
        else RedisStorage(namespace="llm_utils")
    ),
    value_wrapper=ValueWrapperDictInspectArgs(),
)

text_cache = Cache(
    serializer=JSONSerializer(),
    storage=(
        LocalFileStorage(location=get_src_path().parent / os.getenv("LOCAL_CACHE"))
        if os.getenv("LOCAL_CACHE")
        else RedisStorage(namespace="llm_utils")
    ),
    value_wrapper=ValueWrapperDictInspectArgs(),
)

embeddings_cache = Cache(
    serializer=PickleSerializer(),
    storage=(
        LocalFileStorage(location=get_src_path().parent / os.getenv("LOCAL_CACHE"))
        if os.getenv("LOCAL_CACHE")
        else RedisStorage(namespace="llm_utils")
    ),
    value_wrapper=ValueWrapperDictInspectArgs(),
)

FLAGS = CACHE_FLAGS + ["SINGLE_THREAD"] + ["VERBOSE", "LOGGING_DEBUG", "USE_LOGFIRE"]


client = None
PROVIDERS = [
    "openrouter",
    "openai",
    "openai_o1",
    "openai_strict",
    "mistral",
    "anthropic",
    "gemini",
    "qwen",
    "minimax",
    "togetherai",
    "huggingface_local",
]


class LLM_Simulator(LLM_Simulator_Faker):
    fqs_path = get_data_path() / "fq" / "real" / "test_formatted.jsonl"

    @staticmethod
    def pick_random_fq(file_path: str, strip=False):
        import random

        with open(file_path, "r") as file:
            lines = file.readlines()
        random_line = random.choice(lines)
        fq = ForecastingQuestion.model_validate_json(random_line)
        if strip:
            fq = fq.cast_stripped()
        return fq

    @classmethod
    def _fake_custom(cls, t: type):
        import random

        t_name = getattr(t, "__name__", "")
        if t_name in ("VerificationResult", "ValidationResult"):
            return t(reasoning="Simulated valid result", valid=True)
        elif t_name == "RelevanceResult":
            return t(reasons=["Simulated relevant result"], conclusion="relevant", score=1.0)
        elif t_name == "BodyAndDate":
            return t(resolution_date=datetime(2028, 12, 31), resolution_criteria="Simulated criteria")
        elif t_name == "ResolutionDate":
            return t(resolution_date=datetime(2028, 12, 31))
        elif t_name == "Prob_cot":
            return t(chain_of_thought="Simulated CoT reasoning...", prob=random.random())
        elif t_name == "ForecastingQuestion_stripped_list":
            return t(questions=[cls.pick_random_fq(cls.fqs_path, strip=True) for _ in range(5)])
        elif t_name == "ForecastingQuestions":
            return t(questions=[cls.pick_random_fq(cls.fqs_path, strip=False) for _ in range(5)])
        elif t_name == "ForecastingQuestion_stripped_with_resolution_list":
            elem_type = t.model_fields['questions'].annotation.__args__[0]
            fqs = [cls.pick_random_fq(cls.fqs_path, strip=True) for _ in range(5)]
            return t(
                questions=[
                    elem_type(title=fq.title, body=fq.body, resolution=True)
                    for fq in fqs
                ]
            )
        elif t_name == "ForecastingQuestion_stripped_with_resolution":
            fq = cls.pick_random_fq(cls.fqs_path, strip=True)
            return t(title=fq.title, body=fq.body, resolution=True)
        elif t_name == "ForecastingQuestion_title_body":
            fq = cls.pick_random_fq(cls.fqs_path, strip=True)
            return t(title=fq.title, body=fq.body)
        elif t_name in ("ForecastingQuestionGroundTruthResolutionStrict", "ForecastingQuestionGroundTruthResolutionLax"):
            return t(resolution=True, reasoning="Simulated resolution reasoning")
        elif issubclass(t, Prob):
            return t(prob=random.random())
        elif issubclass(t, Forecast):
            return t(prob=random.random(), metadata=None)
        elif issubclass(t, ForecastingQuestion):
            return cls.pick_random_fq(cls.fqs_path, strip=False)
        elif issubclass(t, ForecastingQuestion_stripped):
            return cls.pick_random_fq(cls.fqs_path, strip=True)
        elif issubclass(t, Consequence_ConsequenceType):
            return random.choice(list(Consequence_ConsequenceType))
        elif issubclass(t, Consequence_ClassifyOutput):
            return Consequence_ClassifyOutput(
                consequence_type=[
                    random.choice(list(Consequence_ConsequenceType))
                    for _ in range(random.randint(1, 4))
                ]
            )
        else:
            raise NotImplementedError(f"{t} is not a known custom type")


def singleton_constructor(get_instance_func):
    instances = {}

    def wrapper(*args, **kwargs):
        env_key = (
            os.getenv("OPENAI_API_KEY"),
            os.getenv("OPENAI_BASE_URL"),
            os.getenv("ANTHROPIC_API_KEY"),
            os.getenv("ANTHROPIC_BASE_URL"),
            os.getenv("GEMINI_API_KEY"),
            os.getenv("MISTRAL_API_KEY"),
        )
        key = (get_instance_func, args, frozenset(kwargs.items()), env_key)
        if key not in instances:
            instances[key] = get_instance_func(*args, **kwargs)
        return instances[key]

    return wrapper


@singleton_constructor
def get_async_perplexity_client() -> AsyncPerplexityClient:
    load_dotenv()
    api_key = os.getenv("PERPLEXITY_API_TOKEN")
    if not api_key:
        raise ValueError("PERPLEXITY_API_TOKEN not found in environment variables")
    client = AsyncPerplexityClient(api_key)
    # If you have a logging/instrumentation library like logfire, you can add it here
    # logfire.instrument_perplexity(client)
    return client


@singleton_constructor
def get_sync_perplexity_client() -> SyncPerplexityClient:
    load_dotenv()
    api_key = os.getenv("PERPLEXITY_API_TOKEN")
    if not api_key:
        raise ValueError("PERPLEXITY_API_TOKEN not found in environment variables")
    client = SyncPerplexityClient(api_key)
    # If you have a logging/instrumentation library like logfire, you can add it here
    # logfire.instrument_perplexity(client)
    return client


@singleton_constructor
def get_async_openai_client_pydantic(
    mode: Literal[Mode.TOOLS, Mode.TOOLS_STRICT, Mode.JSON_O1] = Mode.TOOLS,
    api_key: str = None,
    base_url: str = None,
) -> Instructor:
    key = api_key or os.getenv("OPENAI_API_KEY")
    url = base_url or os.getenv("OPENAI_BASE_URL")
    _client = AsyncOpenAI(api_key=key, base_url=url)
    logfire.instrument_openai(_client)

    return instructor.from_openai(_client, mode=mode)


@singleton_constructor
def get_async_openai_client_native(api_key: str = None, base_url: str = None) -> AsyncOpenAI:
    key = api_key or os.getenv("OPENAI_API_KEY")
    url = base_url or os.getenv("OPENAI_BASE_URL")
    client = AsyncOpenAI(api_key=key, base_url=url)
    logfire.instrument_openai(client)
    return client


@singleton_constructor
def get_openai_client_pydantic(
    mode: Literal[Mode.TOOLS, Mode.TOOLS_STRICT, Mode.JSON_O1] = Mode.TOOLS,
    api_key: str = None,
    base_url: str = None,
) -> Instructor:
    key = api_key or os.getenv("OPENAI_API_KEY")
    url = base_url or os.getenv("OPENAI_BASE_URL")
    _client = OpenAI(api_key=key, base_url=url)
    logfire.instrument_openai(_client)
    return instructor.from_openai(_client, mode=mode)


@singleton_constructor
def get_openai_client_native(api_key: str = None, base_url: str = None) -> OpenAI:
    key = api_key or os.getenv("OPENAI_API_KEY")
    url = base_url or os.getenv("OPENAI_BASE_URL")
    client = OpenAI(api_key=key, base_url=url)
    logfire.instrument_openai(client)
    return client


@singleton_constructor
def get_async_openrouter_client_pydantic(**kwargs) -> Instructor:
    print(
        "Only some OpenRouter endpoints will work. If you encounter errors, please check on the OpenRouter website."
    )
    api_key = os.getenv("OPENROUTER_API_KEY")
    print(f"OPENROUTER_API_KEY: {api_key}")
    _client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    logfire.instrument_openai(_client)
    return instructor.from_openai(_client, mode=Mode.MD_JSON, **kwargs)


@singleton_constructor
def get_async_openrouter_client_native() -> AsyncOpenAI:
    print("Calling models through OpenRouter")
    api_key = os.getenv("OPENROUTER_API_KEY")
    print(f"OPENROUTER_API_KEY: {api_key}")
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    logfire.instrument_openai(client)
    return client


@singleton_constructor
def get_openrouter_client_pydantic(**kwargs) -> Instructor:
    print(
        "Only some OpenRouter endpoints have `response_format`. If you encounter errors, please check on the OpenRouter website."
    )
    print("Calling models through OpenRouter")
    _client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    print(f"OPENROUTER_API_KEY: {os.getenv('OPENROUTER_API_KEY')}")
    logfire.instrument_openai(_client)
    return instructor.from_openai(_client, mode=Mode.TOOLS, **kwargs)


@singleton_constructor
def get_openrouter_client_native() -> OpenAI:
    print("Calling models through OpenRouter")
    api_key = os.getenv("OPENROUTER_API_KEY")
    print(f"OPENROUTER_API_KEY: {api_key}")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    logfire.instrument_openai(client)
    return client


@singleton_constructor
def get_anthropic_async_client_pydantic() -> Instructor:
    api_key = os.getenv("ANTHROPIC_KEY")
    base_url = os.getenv("ANTHROPIC_BASE_URL")
    _client = AsyncAnthropic(api_key=api_key, base_url=base_url)
    # As of 27 Aug 2024, cannot setup logfire for anthropic client, because of version mismatch.
    return instructor.from_anthropic(_client, mode=instructor.Mode.ANTHROPIC_JSON)


@singleton_constructor
def get_anthropic_async_client_native() -> AsyncAnthropic:
    api_key = os.getenv("ANTHROPIC_KEY")
    base_url = os.getenv("ANTHROPIC_BASE_URL")
    _client = AsyncAnthropic(api_key=api_key, base_url=base_url)
    # As of 27 Aug 2024, cannot setup logfire for anthropic client, because of version mismatch.
    return _client


@singleton_constructor
def get_anthropic_client_pydantic() -> Instructor:
    api_key = os.getenv("ANTHROPIC_KEY")
    base_url = os.getenv("ANTHROPIC_BASE_URL")
    _client = Anthropic(api_key=api_key, base_url=base_url)
    # As of 27 Aug 2024, cannot setup logfire for anthropic client, because of version mismatch.
    return instructor.from_anthropic(_client, mode=instructor.Mode.ANTHROPIC_JSON)


@singleton_constructor
def get_anthropic_client_native() -> Anthropic:
    api_key = os.getenv("ANTHROPIC_KEY")
    base_url = os.getenv("ANTHROPIC_BASE_URL")
    _client = Anthropic(api_key=api_key, base_url=base_url)
    # As of 27 Aug 2024, cannot setup logfire for anthropic client, because of version mismatch.
    return _client


@singleton_constructor
def get_async_openai_client_pydantic_for_anthropic(api_key: str = None, base_url: str = None) -> Instructor:
    key = api_key or os.getenv("ANTHROPIC_KEY") or os.getenv("OPENAI_API_KEY")
    url = base_url or os.getenv("ANTHROPIC_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    _client = AsyncOpenAI(api_key=key, base_url=url)
    logfire.instrument_openai(_client)
    mode = Mode.TOOLS
    openai_mode_str = os.getenv("OPENAI_INSTRUCTOR_MODE")
    if openai_mode_str:
        try:
            mode = getattr(Mode, openai_mode_str)
        except AttributeError:
            pass
    return instructor.from_openai(_client, mode=mode)


@singleton_constructor
def get_openai_client_pydantic_for_anthropic(api_key: str = None, base_url: str = None) -> Instructor:
    key = api_key or os.getenv("ANTHROPIC_KEY") or os.getenv("OPENAI_API_KEY")
    url = base_url or os.getenv("ANTHROPIC_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    _client = OpenAI(api_key=key, base_url=url)
    logfire.instrument_openai(_client)
    mode = Mode.TOOLS
    openai_mode_str = os.getenv("OPENAI_INSTRUCTOR_MODE")
    if openai_mode_str:
        try:
            mode = getattr(Mode, openai_mode_str)
        except AttributeError:
            pass
    return instructor.from_openai(_client, mode=mode)


@singleton_constructor
def get_async_openai_client_native_for_anthropic(api_key: str = None, base_url: str = None) -> AsyncOpenAI:
    key = api_key or os.getenv("ANTHROPIC_KEY") or os.getenv("OPENAI_API_KEY")
    url = base_url or os.getenv("ANTHROPIC_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    _client = AsyncOpenAI(api_key=key, base_url=url)
    logfire.instrument_openai(_client)
    return _client


@singleton_constructor
def get_openai_client_native_for_anthropic(api_key: str = None, base_url: str = None) -> OpenAI:
    key = api_key or os.getenv("ANTHROPIC_KEY") or os.getenv("OPENAI_API_KEY")
    url = base_url or os.getenv("ANTHROPIC_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    _client = OpenAI(api_key=key, base_url=url)
    logfire.instrument_openai(_client)
    return _client


@singleton_constructor
def get_async_gemini_client_pydantic() -> Instructor:
    api_key = os.getenv("GEMINI_KEY")
    base_url = os.getenv("GEMINI_BASE_URL")
    _client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    logfire.instrument_openai(_client)
    return instructor.from_openai(_client, mode=Mode.MD_JSON)


@singleton_constructor
def get_gemini_client_pydantic() -> Instructor:
    api_key = os.getenv("GEMINI_KEY")
    base_url = os.getenv("GEMINI_BASE_URL")
    _client = OpenAI(api_key=api_key, base_url=base_url)
    logfire.instrument_openai(_client)
    return instructor.from_openai(_client, mode=Mode.MD_JSON)


@singleton_constructor
def get_async_gemini_client_native() -> AsyncOpenAI:
    api_key = os.getenv("GEMINI_KEY")
    base_url = os.getenv("GEMINI_BASE_URL")
    _client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    logfire.instrument_openai(_client)
    return _client


@singleton_constructor
def get_gemini_client_native() -> OpenAI:
    api_key = os.getenv("GEMINI_KEY")
    base_url = os.getenv("GEMINI_BASE_URL")
    _client = OpenAI(api_key=api_key, base_url=base_url)
    logfire.instrument_openai(_client)
    return _client


@singleton_constructor
def get_async_qwen_client_pydantic() -> Instructor:
    api_key = os.getenv("QWEN_KEY")
    base_url = os.getenv("QWEN_BASE_URL")
    _client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    logfire.instrument_openai(_client)
    return instructor.from_openai(_client, mode=Mode.TOOLS)


@singleton_constructor
def get_qwen_client_pydantic() -> Instructor:
    api_key = os.getenv("QWEN_KEY")
    base_url = os.getenv("QWEN_BASE_URL")
    _client = OpenAI(api_key=api_key, base_url=base_url)
    logfire.instrument_openai(_client)
    return instructor.from_openai(_client, mode=Mode.TOOLS)


@singleton_constructor
def get_async_qwen_client_native() -> AsyncOpenAI:
    api_key = os.getenv("QWEN_KEY")
    base_url = os.getenv("QWEN_BASE_URL")
    _client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    logfire.instrument_openai(_client)
    return _client


@singleton_constructor
def get_qwen_client_native() -> OpenAI:
    api_key = os.getenv("QWEN_KEY")
    base_url = os.getenv("QWEN_BASE_URL")
    _client = OpenAI(api_key=api_key, base_url=base_url)
    logfire.instrument_openai(_client)
    return _client


@singleton_constructor
def get_async_minimax_client_pydantic() -> Instructor:
    api_key = os.getenv("MINIMAX_KEY")
    base_url = os.getenv("MINIMAX_BASE_URL")
    _client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    logfire.instrument_openai(_client)
    return instructor.from_openai(_client, mode=Mode.TOOLS)


@singleton_constructor
def get_minimax_client_pydantic() -> Instructor:
    api_key = os.getenv("MINIMAX_KEY")
    base_url = os.getenv("MINIMAX_BASE_URL")
    _client = OpenAI(api_key=api_key, base_url=base_url)
    logfire.instrument_openai(_client)
    return instructor.from_openai(_client, mode=Mode.TOOLS)


@singleton_constructor
def get_async_minimax_client_native() -> AsyncOpenAI:
    api_key = os.getenv("MINIMAX_KEY")
    base_url = os.getenv("MINIMAX_BASE_URL")
    _client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    logfire.instrument_openai(_client)
    return _client


@singleton_constructor
def get_minimax_client_native() -> OpenAI:
    api_key = os.getenv("MINIMAX_KEY")
    base_url = os.getenv("MINIMAX_BASE_URL")
    _client = OpenAI(api_key=api_key, base_url=base_url)
    logfire.instrument_openai(_client)
    return _client


@singleton_constructor
def get_async_mistral_client_pydantic() -> Instructor:
    api_key = os.getenv("MISTRAL_KEY")
    base_url = os.getenv("MISTRAL_BASE_URL")
    _client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    logfire.instrument_openai(_client)
    return instructor.from_openai(_client, mode=Mode.MD_JSON)


@singleton_constructor
def get_mistral_client_pydantic() -> Instructor:
    api_key = os.getenv("MISTRAL_KEY")
    base_url = os.getenv("MISTRAL_BASE_URL")
    _client = OpenAI(api_key=api_key, base_url=base_url)
    logfire.instrument_openai(_client)
    return instructor.from_openai(_client, mode=Mode.MD_JSON)


@singleton_constructor
def get_async_mistral_client_native() -> AsyncOpenAI:
    api_key = os.getenv("MISTRAL_KEY")
    base_url = os.getenv("MISTRAL_BASE_URL")
    _client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    logfire.instrument_openai(_client)
    return _client


@singleton_constructor
def get_mistral_client_native() -> OpenAI:
    api_key = os.getenv("MISTRAL_KEY")
    base_url = os.getenv("MISTRAL_BASE_URL")
    _client = OpenAI(api_key=api_key, base_url=base_url)
    logfire.instrument_openai(_client)
    return _client


@singleton_constructor
def get_togetherai_client_native() -> OpenAI:
    url = "https://api.together.xyz/v1"
    api_key = os.getenv("TOGETHER_API_KEY")
    _client = OpenAI(api_key=api_key, base_url=url)
    logfire.instrument_openai(_client)
    return _client


def is_openai(model: str) -> bool:
    if "/" in model and not model.startswith(("openai/", "anthropic/", "gemini/", "qwen/", "minimax/", "mistral/")):
        return True
    keywords = [
        "ft:gpt",
        "o1",
        "o3",
        "o4",
        "gpt-5",
        "gpt-4o-mini",
        "gpt-4",
        "gpt-3.5",
        "gpt-",
        "babbage",
        "davinci",
        "openai",
        "open-ai",
    ]
    return any(keyword in model for keyword in keywords)


def is_minimax(model: str) -> bool:
    keywords = ["minimax"]
    return any(keyword.lower() in model.lower() for keyword in keywords)


def is_mistral(model: str) -> bool:
    keywords = ["mistral", "pixtral", "codestral"]
    return any(keyword in model for keyword in keywords)


def is_perplexity_ai(model: str) -> bool:
    keywords = ["perplexity", "sonar"]
    return any(keyword.lower() in model.lower() for keyword in keywords)


def is_togetherai(model: str) -> bool:
    keywords = ["together", "llama", "phi", "orca", "Hermes", "Yi"]
    return any(keyword in model for keyword in keywords)


def is_anthropic(model: str) -> bool:
    keywords = ["anthropic", "claude"]
    return any(keyword in model for keyword in keywords)


def is_gemini(model: str) -> bool:
    keywords = ["gemini"]
    return any(keyword in model for keyword in keywords)


def is_qwen(model: str) -> bool:
    keywords = ["qwen", "deepseek"]
    return any(keyword in model for keyword in keywords)


def is_huggingface_local(model: str) -> bool:
    keywords = ["huggingface", "hf"]
    return any(keyword in model for keyword in keywords)


def map_model_name(model: str) -> str:
    # Preserve provider prefix if present
    prefix = ""
    for p in ["openai/", "anthropic/", "gemini/", "qwen/", "minimax/", "mistral/"]:
        if model.startswith(p):
            prefix = p
            model = model[len(p):]
            break
            
    if "gpt-4o" in model:
        mapped = os.getenv("DEFAULT_MODEL", "gpt-5.4-mini")
    elif "claude-3.5-sonnet" in model:
        mapped = "claude-opus-4-6-thinking"
    else:
        mapped = model
        
    return prefix + mapped


def get_provider(model: str) -> str:
    model = map_model_name(model)
    if os.getenv("USE_OPENROUTER") and os.getenv("USE_OPENROUTER") != "False":
        return "openrouter"
    elif is_minimax(model):
        return "minimax"
    elif is_mistral(model):
        return "mistral"
    elif is_openai(model):
        print(
            f"Using OpenAI provider for model {model}, key {os.getenv('OPENAI_API_KEY')}"
        )
        if "o1" in model or "o3" in model or "o4" in model:
            if os.getenv("ALLOW_OPENAI_O1", "False") == "True":
                return "openai_o1"
            else:
                raise ValueError("OPENAI_O1 is not allowed")
        elif os.getenv("OPENAI_JSON_STRICT") == "True":
            return "openai_strict"
        else:
            return "openai"
    elif is_perplexity_ai(model):
        return "perplexity"
    elif is_anthropic(model):
        return "anthropic"
    elif is_gemini(model):
        return "gemini"
    elif is_qwen(model):
        return "qwen"
    elif is_togetherai(model):
        return "togetherai"
    elif is_huggingface_local(model):
        return "huggingface_local"
    else:
        print(
            f"Model {model} is not supported with a provider; USE_OPENROUTER should be True"
        )
        assert False


def get_client_pydantic(model: str, use_async=True, api_key: str = None, base_url: str = None) -> tuple[Instructor, str, str]:
    model = map_model_name(model)
    provider: str = get_provider(model)
    final_model_name = model

    print(f"Using {provider} provider for model {model}")
    if provider == "openrouter":
        kwargs = {}
        client = (
            get_async_openrouter_client_pydantic(**kwargs)
            if use_async
            else get_openrouter_client_pydantic(**kwargs)
        )
    elif provider in ["openai", "openai_strict", "openai_o1"]:
        dict_modes = {
            "openai": Mode.TOOLS,
            "openai_strict": Mode.TOOLS_STRICT,
            "openai_o1": Mode.JSON_O1,
        }
        mode = dict_modes[provider]
        openai_mode_str = os.getenv("OPENAI_INSTRUCTOR_MODE")
        if openai_mode_str:
            try:
                mode = getattr(Mode, openai_mode_str)
            except AttributeError:
                pass
        final_model_name = model.replace("openai/", "")
        client = (
            get_async_openai_client_pydantic(mode=mode, api_key=api_key, base_url=base_url)
            if use_async
            else get_openai_client_pydantic(mode=mode, api_key=api_key, base_url=base_url)
        )
    elif provider == "anthropic":
        final_model_name = model.replace("anthropic/", "")
        if final_model_name in ANTHROPIC_DEFAULT_MODEL_NAME_MAP:
            final_model_name = ANTHROPIC_DEFAULT_MODEL_NAME_MAP[final_model_name]
        base_url_env = os.getenv("ANTHROPIC_BASE_URL") or base_url
        if base_url_env and ("llm.wokushop.com" in base_url_env or "api.xah.io" in base_url_env or "/v1" in base_url_env):
            client = (
                get_async_openai_client_pydantic_for_anthropic(api_key=api_key, base_url=base_url)
                if use_async
                else get_openai_client_pydantic_for_anthropic(api_key=api_key, base_url=base_url)
            )
        else:
            client = (
                get_anthropic_async_client_pydantic()
                if use_async
                else get_anthropic_client_pydantic()
            )
    elif provider == "gemini":
        final_model_name = model.replace("gemini/", "")
        client = (
            get_async_gemini_client_pydantic()
            if use_async
            else get_gemini_client_pydantic()
        )
    elif provider == "qwen":
        final_model_name = model.replace("qwen/", "")
        client = (
            get_async_qwen_client_pydantic()
            if use_async
            else get_qwen_client_pydantic()
        )
    elif provider == "minimax":
        final_model_name = model.replace("minimax/", "")
        client = (
            get_async_minimax_client_pydantic()
            if use_async
            else get_minimax_client_pydantic()
        )
    elif provider == "mistral":
        final_model_name = model.replace("mistral/", "")
        client = (
            get_async_mistral_client_pydantic()
            if use_async
            else get_mistral_client_pydantic()
        )
    else:
        raise NotImplementedError(
            f"Model {model} Pydantic client is not supported for now outside of OpenRouter"
        )

    return client, provider, final_model_name


def get_client_native(
    model: str, use_async=True, api_key: str = None, base_url: str = None
) -> tuple[AsyncOpenAI | OpenAI | AsyncAnthropic | Anthropic, str, str]:
    model = map_model_name(model)
    provider = get_provider(model)
    final_model_name = model

    print(f"Using {provider} provider for model {model}")
    if provider == "openrouter":
        client = (
            get_async_openrouter_client_native()
            if use_async
            else get_openrouter_client_native()
        )
    elif provider in ["openai", "openai_strict", "openai_o1"]:
        final_model_name = model.replace("openai/", "")
        client = (
            get_async_openai_client_native(api_key=api_key, base_url=base_url)
            if use_async
            else get_openai_client_native(api_key=api_key, base_url=base_url)
        )
    elif provider == "anthropic":
        final_model_name = model.replace("anthropic/", "")
        if final_model_name in ANTHROPIC_DEFAULT_MODEL_NAME_MAP:
            final_model_name = ANTHROPIC_DEFAULT_MODEL_NAME_MAP[final_model_name]
        base_url_env = os.getenv("ANTHROPIC_BASE_URL") or base_url
        if base_url_env and ("llm.wokushop.com" in base_url_env or "api.xah.io" in base_url_env or "/v1" in base_url_env):
            client = (
                get_async_openai_client_native_for_anthropic(api_key=api_key, base_url=base_url)
                if use_async
                else get_openai_client_native_for_anthropic(api_key=api_key, base_url=base_url)
            )
        else:
            client = (
                get_anthropic_async_client_native()
                if use_async
                else get_anthropic_client_native()
            )
    elif provider == "gemini":
        final_model_name = model.replace("gemini/", "")
        client = (
            get_async_gemini_client_native()
            if use_async
            else get_gemini_client_native()
        )
    elif provider == "qwen":
        final_model_name = model.replace("qwen/", "")
        client = (
            get_async_qwen_client_native()
            if use_async
            else get_qwen_client_native()
        )
    elif provider == "minimax":
        final_model_name = model.replace("minimax/", "")
        client = (
            get_async_minimax_client_native()
            if use_async
            else get_minimax_client_native()
        )
    elif provider == "mistral":
        final_model_name = model.replace("mistral/", "")
        client = (
            get_async_mistral_client_native()
            if use_async
            else get_mistral_client_native()
        )
    elif provider == "togetherai":
        if use_async:
            raise NotImplementedError(
                "Only synchronous calls are supported for TogetherAI"
            )
        client = get_togetherai_client_native()
    elif provider == "perplexity":
        if use_async:
            client = get_async_perplexity_client()
        else:
            client = get_sync_perplexity_client()
    else:
        raise NotImplementedError(f"Model {model} is not supported for now")

    return client, provider, final_model_name


def is_llama2_tokenized(model: str) -> bool:
    keywords = ["Llama-2", "pythia"]
    return any(keyword in model for keyword in keywords)


def _mistral_message_transform(messages):
    try:
        from mistralai.models.chat_completion import ChatMessage
    except ImportError:
        return messages
    try:
        mistral_messages = []
        for message in messages:
            mistral_message = ChatMessage(role=message["role"], content=message["content"])
            mistral_messages.append(mistral_message)
        return mistral_messages
    except Exception:
        return messages


def _o1_message_params_transform(messages, options):
    o1_messages = []
    if messages[0]["role"] == "system":
        o1_messages.append({"role": "user", "content": messages[0]["content"]})
        o1_messages.append(
            {"role": "assistant", "content": "System message acknowledged"}
        )
        o1_messages.extend(messages[1:])
    else:
        o1_messages.extend(messages)

    options["temperature"] = 1
    return o1_messages, options


def supports_system_message(model: str, client_name: str) -> bool:
    """
    There might be other models that don't support system messages; check if there is an error when running the code.
    """
    if any(m in model for m in ["o1", "o3", "o4"]):
        return False
    return True


ANTHROPIC_DEFAULT_MODEL_NAME_MAP = {}


@pydantic_cache
@costly(simulator=LLM_Simulator.simulate_llm_call)
@logfire.instrument("query_api_chat", extract_args=True)
async def query_api_chat(
    messages: list[dict[str, str]],
    verbose=False,
    model: str | None = None,
    cost_log: Costlog = GLOBAL_COST_LOG,  # need to give explicitly because of cache
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs,
) -> BaseModel:
    """
    Query the API (through instructor.Instructor) with the given messages.

    Order of precedence for model:
    1. `model` argument
    2. `model` in `kwargs`
    3. Default model
    """
    try:
        if not os.getenv("NO_CACHE"):
            assert (
                kwargs.get("response_model", -1) is not None
            ), "Cannot pass response_model=None if caching is enabled"

        default_options = {
            "model": "gpt-4o-mini-2024-07-18",
            "response_model": PlainText,
        }
        options = default_options | kwargs
        options["model"] = model or options["model"]
        client, client_name, final_model_name = get_client_pydantic(
            options["model"], use_async=True, api_key=api_key, base_url=base_url
        )
        options["model"] = final_model_name
        if options.get("n", 1) != 1:
            raise NotImplementedError("Multiple queries not supported yet")

        call_messages = (
            _mistral_message_transform(messages) if client_name == "mistral" else messages
        )
        call_messages, options = (
            _o1_message_params_transform(call_messages, options)
            if not supports_system_message(options["model"], client_name)
            else (call_messages, options)
        )

        if client_name == "anthropic":
            options["max_tokens"] = options.get("max_tokens", 1024)

        if verbose or os.getenv("VERBOSE") == "True":
            print(f"{options=}, {len(messages)=}")

        response, completion = await client.chat.completions.create_with_completion(
            messages=call_messages,
            **options,
        )
        try:
            cost_info = {
                "input_tokens": completion.usage.prompt_tokens,
                "output_tokens": completion.usage.completion_tokens,
            }
        except AttributeError:
            cost_info = {
                "input_tokens": completion.usage.input_tokens,
                "output_tokens": completion.usage.output_tokens,
            }
        if verbose or os.getenv("VERBOSE") == "True":
            print(f"...\nText: {messages[-1]['content']}\nResponse: {response}")
            
        try:
            import streamlit as st
            if "api_working_status" not in st.session_state or not st.session_state["api_working_status"].get("fallback"):
                st.session_state["api_working_status"] = {
                    "working": True,
                    "model_used": model or options.get("model", "unknown"),
                    "fallback": False
                }
        except:
            pass
            
        return CostlyResponse(output=response, cost_info=cost_info)
        
    except Exception as e:
        fallback_key = os.getenv("FALLBACK_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        fallback_url = os.getenv("FALLBACK_OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL", "https://llm.wokushop.com/v1")
        current_api_key = api_key or os.getenv("OPENAI_API_KEY")
        target_model = model or (options.get("model") if 'options' in locals() else "unknown")
        if (target_model == "gpt-4o-mini" and (not fallback_key or current_api_key == fallback_key)) or not fallback_key:
            try:
                import streamlit as st
                st.session_state["api_working_status"] = {
                    "working": False,
                    "error": str(e)
                }
            except:
                pass
            raise e
            
        print(f"Query failed with model {target_model}: {e}. Falling back to gpt-4o-mini...")
        
        orig_key = os.environ.get("OPENAI_API_KEY")
        orig_url = os.environ.get("OPENAI_BASE_URL")
        orig_model = os.environ.get("DEFAULT_MODEL")
        
        os.environ["OPENAI_API_KEY"] = fallback_key
        os.environ["OPENAI_BASE_URL"] = fallback_url
        os.environ["DEFAULT_MODEL"] = "gpt-4o-mini"
        
        try:
            import streamlit as st
            st.session_state["api_working_status"] = {
                "working": True,
                "model_used": "gpt-4o-mini",
                "fallback": True,
                "original_model": target_model,
                "error": str(e)
            }
        except:
            pass
            
        try:
            if 'model' in kwargs:
                kwargs['model'] = "gpt-4o-mini"
            return await query_api_chat(
                messages,
                verbose,
                model="gpt-4o-mini",
                cost_log=cost_log,
                api_key=fallback_key,
                base_url=fallback_url,
                **kwargs
            )
        finally:
            if orig_key is not None: os.environ["OPENAI_API_KEY"] = orig_key
            else: os.environ.pop("OPENAI_API_KEY", None)
            if orig_url is not None: os.environ["OPENAI_BASE_URL"] = orig_url
            else: os.environ.pop("OPENAI_BASE_URL", None)
            if orig_model is not None: os.environ["DEFAULT_MODEL"] = orig_model
            else: os.environ.pop("DEFAULT_MODEL", None)


@text_cache
@costly(simulator=LLM_Simulator.simulate_llm_call)
@logfire.instrument("query_api_chat_native", extract_args=True)
async def query_api_chat_native(
    messages: list[dict[str, str]],
    verbose=False,
    model: str | None = None,
    cost_log: Costlog = GLOBAL_COST_LOG,  # need to give explicitly because of cache
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs,
) -> str:
    try:
        default_options = {
            "model": "gpt-4o-mini-2024-07-18",
        }
        options = default_options | kwargs
        options["model"] = model or options["model"]

        client, client_name, final_model_name = get_client_native(
            options["model"], use_async=True, api_key=api_key, base_url=base_url
        )
        options["model"] = final_model_name
        call_messages = (
            _mistral_message_transform(messages) if client_name == "mistral" else messages
        )
        call_messages, options = (
            _o1_message_params_transform(call_messages, options)
            if not supports_system_message(options["model"], client_name)
            else (call_messages, options)
        )

        if verbose or os.getenv("VERBOSE") == "True":
            print(f"{options=}, {len(messages)=}")

        max_retries = 5
        backoff = 2.0
        for attempt in range(max_retries):
            try:
                if client_name == "mistral" and not isinstance(client, (AsyncOpenAI, OpenAI, Instructor)) and not os.getenv("USE_OPENROUTER"):
                    response = await client.chat(
                        messages=call_messages,
                        **options,
                    )
                else:
                    response = await client.chat.completions.create(
                        messages=call_messages,
                        **options,
                    )
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                import asyncio
                sleep_time = backoff * (2 ** attempt)
                print(f"API call failed: {e}. Retrying in {sleep_time:.1f}s (attempt {attempt + 1}/{max_retries})...")
                await asyncio.sleep(sleep_time)


        text_response = response.choices[0].message.content

        if verbose or os.getenv("VERBOSE") == "True":
            print(f"...\nText: {messages[-1]['content']}\nResponse: {text_response}\n")

        try:
            cost_info = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }
        except AttributeError:
            cost_info = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }

        try:
            import streamlit as st
            if "api_working_status" not in st.session_state or not st.session_state["api_working_status"].get("fallback"):
                st.session_state["api_working_status"] = {
                    "working": True,
                    "model_used": model or options.get("model", "unknown"),
                    "fallback": False
                }
        except:
            pass

        return CostlyResponse(output=text_response, cost_info=cost_info)

    except Exception as e:
        fallback_key = os.getenv("FALLBACK_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        fallback_url = os.getenv("FALLBACK_OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL", "https://llm.wokushop.com/v1")
        current_api_key = api_key or os.getenv("OPENAI_API_KEY")
        target_model = model or (options.get("model") if 'options' in locals() else "unknown")
        if (target_model == "gpt-4o-mini" and (not fallback_key or current_api_key == fallback_key)) or not fallback_key:
            try:
                import streamlit as st
                st.session_state["api_working_status"] = {
                    "working": False,
                    "error": str(e)
                }
            except:
                pass
            raise e
            
        print(f"Query failed with model {target_model}: {e}. Falling back to gpt-4o-mini...")
        
        orig_key = os.environ.get("OPENAI_API_KEY")
        orig_url = os.environ.get("OPENAI_BASE_URL")
        orig_model = os.environ.get("DEFAULT_MODEL")
        
        os.environ["OPENAI_API_KEY"] = fallback_key
        os.environ["OPENAI_BASE_URL"] = fallback_url
        os.environ["DEFAULT_MODEL"] = "gpt-4o-mini"
        
        try:
            import streamlit as st
            st.session_state["api_working_status"] = {
                "working": True,
                "model_used": "gpt-4o-mini",
                "fallback": True,
                "original_model": target_model,
                "error": str(e)
            }
        except:
            pass
            
        try:
            if 'model' in kwargs:
                kwargs['model'] = "gpt-4o-mini"
            return await query_api_chat_native(
                messages,
                verbose,
                model="gpt-4o-mini",
                cost_log=cost_log,
                api_key=fallback_key,
                base_url=fallback_url,
                **kwargs
            )
        finally:
            if orig_key is not None: os.environ["OPENAI_API_KEY"] = orig_key
            else: os.environ.pop("OPENAI_API_KEY", None)
            if orig_url is not None: os.environ["OPENAI_BASE_URL"] = orig_url
            else: os.environ.pop("OPENAI_BASE_URL", None)
            if orig_model is not None: os.environ["DEFAULT_MODEL"] = orig_model
            else: os.environ.pop("DEFAULT_MODEL", None)


@pydantic_cache
@costly(simulator=LLM_Simulator.simulate_llm_call)
@logfire.instrument("query_api_chat_sync", extract_args=True)
def query_api_chat_sync(
    messages: list[dict[str, str]],
    verbose=False,
    model: str | None = None,
    cost_log: Costlog = GLOBAL_COST_LOG,  # need to give explicitly because of cache
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs,
) -> BaseModel:
    try:
        if not os.getenv("NO_CACHE"):
            assert (
                kwargs.get("response_model", -1) is not None
            ), "Cannot pass response_model=None if caching is enabled"

        default_options = {
            "model": "gpt-4o-mini-2024-07-18",
            "response_model": PlainText,
        }
        options = default_options | kwargs
        options["model"] = model or options["model"]
        client, client_name, final_model_name = get_client_pydantic(
            options["model"], use_async=False, api_key=api_key, base_url=base_url
        )
        options["model"] = final_model_name
        if options.get("n", 1) != 1:
            raise NotImplementedError("Multiple structured queries not supported yet")

        call_messages = (
            _mistral_message_transform(messages) if client_name == "mistral" else messages
        )
        call_messages, options = (
            _o1_message_params_transform(call_messages, options)
            if not supports_system_message(options["model"], client_name)
            else (call_messages, options)
        )

        if client_name == "anthropic":
            options["max_tokens"] = options.get("max_tokens", 1024)

        if verbose or os.getenv("VERBOSE") == "True":
            print(f"{options=}, {len(messages)=}")

        response, completion = client.chat.completions.create_with_completion(
            messages=call_messages,
            **options,
        )
        # print(f"Completion: {completion}")

        if verbose or os.getenv("VERBOSE") == "True":
            print(f"...\nText: {messages[-1]['content']}\nResponse: {response}")

        try:
            cost_info = {
                "input_tokens": completion.usage.prompt_tokens,
                "output_tokens": completion.usage.completion_tokens,
            }
        except AttributeError:
            cost_info = {
                "input_tokens": completion.usage.input_tokens,
                "output_tokens": completion.usage.output_tokens,
            }

        try:
            import streamlit as st
            if "api_working_status" not in st.session_state or not st.session_state["api_working_status"].get("fallback"):
                st.session_state["api_working_status"] = {
                    "working": True,
                    "model_used": model or options.get("model", "unknown"),
                    "fallback": False
                }
        except:
            pass

        return CostlyResponse(output=response, cost_info=cost_info)

    except Exception as e:
        fallback_key = os.getenv("FALLBACK_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        fallback_url = os.getenv("FALLBACK_OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL", "https://llm.wokushop.com/v1")
        current_api_key = api_key or os.getenv("OPENAI_API_KEY")
        target_model = model or (options.get("model") if 'options' in locals() else "unknown")
        if (target_model == "gpt-4o-mini" and (not fallback_key or current_api_key == fallback_key)) or not fallback_key:
            try:
                import streamlit as st
                st.session_state["api_working_status"] = {
                    "working": False,
                    "error": str(e)
                }
            except:
                pass
            raise e
            
        print(f"Query failed with model {target_model}: {e}. Falling back to gpt-4o-mini...")
        
        orig_key = os.environ.get("OPENAI_API_KEY")
        orig_url = os.environ.get("OPENAI_BASE_URL")
        orig_model = os.environ.get("DEFAULT_MODEL")
        
        os.environ["OPENAI_API_KEY"] = fallback_key
        os.environ["OPENAI_BASE_URL"] = fallback_url
        os.environ["DEFAULT_MODEL"] = "gpt-4o-mini"
        
        try:
            import streamlit as st
            st.session_state["api_working_status"] = {
                "working": True,
                "model_used": "gpt-4o-mini",
                "fallback": True,
                "original_model": target_model,
                "error": str(e)
            }
        except:
            pass
            
        try:
            if 'model' in kwargs:
                kwargs['model'] = "gpt-4o-mini"
            return query_api_chat_sync(
                messages,
                verbose,
                model="gpt-4o-mini",
                cost_log=cost_log,
                api_key=fallback_key,
                base_url=fallback_url,
                **kwargs
            )
        finally:
            if orig_key is not None: os.environ["OPENAI_API_KEY"] = orig_key
            else: os.environ.pop("OPENAI_API_KEY", None)
            if orig_url is not None: os.environ["OPENAI_BASE_URL"] = orig_url
            else: os.environ.pop("OPENAI_BASE_URL", None)
            if orig_model is not None: os.environ["DEFAULT_MODEL"] = orig_model
            else: os.environ.pop("DEFAULT_MODEL", None)


@text_cache
@costly(simulator=LLM_Simulator.simulate_llm_call)
@logfire.instrument("query_api_chat_sync_native", extract_args=True)
def query_api_chat_sync_native(
    messages: list[dict[str, str]],
    verbose=False,
    model: str | None = None,
    cost_log: Costlog = GLOBAL_COST_LOG,  # need to give explicitly because of cache
    api_key: str | None = None,
    base_url: str | None = None,
    **kwargs,
) -> str:
    try:
        default_options = {
            "model": "gpt-4o-mini-2024-07-18",
        }
        options = default_options | kwargs
        options["model"] = model or options["model"]
        client, client_name, final_model_name = get_client_native(
            options["model"], use_async=False, api_key=api_key, base_url=base_url
        )
        options["model"] = final_model_name
        call_messages = (
            _mistral_message_transform(messages) if client_name == "mistral" else messages
        )
        call_messages, options = (
            _o1_message_params_transform(call_messages, options)
            if not supports_system_message(options["model"], client_name)
            else (call_messages, options)
        )

        if verbose or os.getenv("VERBOSE") == "True":
            print(f"{options=}, {len(messages)=}")

        max_retries = 5
        backoff = 2.0
        for attempt in range(max_retries):
            try:
                if client_name == "mistral" and not isinstance(client, (AsyncOpenAI, OpenAI, Instructor)) and not os.getenv("USE_OPENROUTER"):
                    response = client.chat(
                        messages=call_messages,
                        **options,
                    )
                else:
                    response = client.chat.completions.create(
                        messages=call_messages,
                        **options,
                    )
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                import time
                sleep_time = backoff * (2 ** attempt)
                print(f"API call failed: {e}. Retrying in {sleep_time:.1f}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(sleep_time)


        text_response = response.choices[0].message.content

        if verbose or os.getenv("VERBOSE") == "True":
            print(f"...\nText: {messages[-1]['content']}\nResponse: {text_response}")

        try:
            cost_info = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            }
        except AttributeError:
            cost_info = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }

        try:
            import streamlit as st
            if "api_working_status" not in st.session_state or not st.session_state["api_working_status"].get("fallback"):
                st.session_state["api_working_status"] = {
                    "working": True,
                    "model_used": model or options.get("model", "unknown"),
                    "fallback": False
                }
        except:
            pass

        return CostlyResponse(output=text_response, cost_info=cost_info)

    except Exception as e:
        fallback_key = os.getenv("FALLBACK_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        fallback_url = os.getenv("FALLBACK_OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL", "https://llm.wokushop.com/v1")
        current_api_key = api_key or os.getenv("OPENAI_API_KEY")
        target_model = model or (options.get("model") if 'options' in locals() else "unknown")
        if (target_model == "gpt-4o-mini" and (not fallback_key or current_api_key == fallback_key)) or not fallback_key:
            try:
                import streamlit as st
                st.session_state["api_working_status"] = {
                    "working": False,
                    "error": str(e)
                }
            except:
                pass
            raise e
            
        print(f"Query failed with model {target_model}: {e}. Falling back to gpt-4o-mini...")
        
        orig_key = os.environ.get("OPENAI_API_KEY")
        orig_url = os.environ.get("OPENAI_BASE_URL")
        orig_model = os.environ.get("DEFAULT_MODEL")
        
        os.environ["OPENAI_API_KEY"] = fallback_key
        os.environ["OPENAI_BASE_URL"] = fallback_url
        os.environ["DEFAULT_MODEL"] = "gpt-4o-mini"
        
        try:
            import streamlit as st
            st.session_state["api_working_status"] = {
                "working": True,
                "model_used": "gpt-4o-mini",
                "fallback": True,
                "original_model": target_model,
                "error": str(e)
            }
        except:
            pass
            
        try:
            if 'model' in kwargs:
                kwargs['model'] = "gpt-4o-mini"
            return query_api_chat_sync_native(
                messages,
                verbose,
                model="gpt-4o-mini",
                cost_log=cost_log,
                api_key=fallback_key,
                base_url=fallback_url,
                **kwargs
            )
        finally:
            if orig_key is not None: os.environ["OPENAI_API_KEY"] = orig_key
            else: os.environ.pop("OPENAI_API_KEY", None)
            if orig_url is not None: os.environ["OPENAI_BASE_URL"] = orig_url
            else: os.environ.pop("OPENAI_BASE_URL", None)
            if orig_model is not None: os.environ["DEFAULT_MODEL"] = orig_model
            else: os.environ.pop("DEFAULT_MODEL", None)


@dataclass_json
@dataclass
class Example:
    user: str | BaseModel
    assistant: str | BaseModel


def serialize_if_pydantic(obj: str | BaseModel) -> str:
    """
    Idempotent function to convert a BaseModel to a string.
    If the object is already a string, it returns the object as is.
    """
    if isinstance(obj, BaseModel):
        return obj.model_dump_json()
    return obj


def prepare_messages(
    prompt: str | BaseModel | None,
    preface: str | None = None,
    examples: list[Example] | None = None,
) -> list[dict[str, str]]:
    preface = preface or "You are a helpful assistant."
    examples = examples or []
    messages = [{"role": "system", "content": preface}]
    for example in examples:
        example.user = serialize_if_pydantic(example.user)
        example.assistant = serialize_if_pydantic(example.assistant)
        messages.append({"role": "user", "content": example.user})
        # Convert assistant's response to string if it's not already
        assistant_content = (
            str(example.assistant)
            if isinstance(example.assistant, (float, int))
            else example.assistant
        )
        messages.append({"role": "assistant", "content": assistant_content})
    if prompt is not None:
        prompt = serialize_if_pydantic(prompt)
        messages.append({"role": "user", "content": prompt})
    return messages


def prepare_messages_alt(
    prompt: str | BaseModel | None,
    preface: str | None = None,
    examples: list[Example] | None = None,
) -> list[dict[str, str]]:
    sys_preface = "You are a helpful assistant."
    messages = [{"role": "system", "content": sys_preface}]
    examples = examples or []
    if not preface:
        preface = ""
    for example in examples:
        example.user = serialize_if_pydantic(example.user)
        example.assistant = serialize_if_pydantic(example.assistant)
        messages.append({"role": "user", "content": example.user})
        example.user = preface + "\n\n" + example.user
        # Convert assistant's response to string if it's not already
        assistant_content = (
            str(example.assistant)
            if isinstance(example.assistant, (float, int))
            else example.assistant
        )
        messages.append({"role": "assistant", "content": assistant_content})
    if prompt is not None:
        prompt = serialize_if_pydantic(prompt)
        prompt = preface + "\n\n" + prompt
        messages.append({"role": "user", "content": prompt})
    return messages


@logfire.instrument("answer", extract_args=True)
async def answer(
    prompt: str,
    preface: Optional[str] = None,
    examples: Optional[List[Example]] = None,
    prepare_messages_func=prepare_messages,
    with_parsing: bool = False,
    **kwargs,
) -> BaseModel:
    messages = prepare_messages_func(prompt, preface, examples)
    default_options = {
        "model": "gpt-4o-mini-2024-07-18",
        "temperature": 0.5,
        "response_model": PlainText,
    }
    options = default_options | kwargs  # override defaults with kwargs

    if os.getenv("VERBOSE") == "True":
        print(f"{options=}, {len(messages)=}")

    async with global_llm_semaphore:
        if with_parsing:
            return await query_api_chat_with_parsing(messages=messages, **options)
        else:
            return await query_api_chat(messages=messages, **options)


@logfire.instrument("answer_sync", extract_args=True)
def answer_sync(
    prompt: str,
    preface: str | None = None,
    examples: list[Example] | None = None,
    prepare_messages_func=prepare_messages,
    with_parsing: bool = False,
    **kwargs,
) -> BaseModel:
    messages = prepare_messages_func(prompt, preface, examples)
    options = {
        "model": "gpt-4o-mini-2024-07-18",
        "temperature": 0.5,
        "response_model": PlainText,
    } | kwargs
    if with_parsing:
        return query_api_chat_sync_with_parsing(messages=messages, **options)
    else:
        return query_api_chat_sync(messages=messages, **options)


@logfire.instrument("answer_native", extract_args=True)
async def answer_native(
    prompt: str,
    preface: Optional[str] = None,
    examples: Optional[List[Example]] = None,
    prepare_messages_func=prepare_messages,
    **kwargs,
) -> str:
    messages = prepare_messages_func(prompt, preface, examples)
    default_options = {
        "model": "gpt-4o-mini-2024-07-18",
        "temperature": 0.5,
    }
    options = default_options | kwargs  # override defaults with kwargs

    if os.getenv("VERBOSE") == "True":
        print(f"{options=}, {len(messages)=}")

    async with global_llm_semaphore:
        response = await query_api_chat_native(messages=messages, **options)
        return response


@logfire.instrument("answer_native_sync", extract_args=True)
def answer_native_sync(
    prompt: str,
    preface: str | None = None,
    examples: list[Example] | None = None,
    prepare_messages_func=prepare_messages,
    **kwargs,
) -> str:
    messages = prepare_messages_func(prompt, preface, examples)
    options = {
        "model": "gpt-4o-mini-2024-07-18",
        "temperature": 0.5,
    } | kwargs
    response = query_api_chat_sync_native(messages=messages, **options)
    return response


async def answer_messages(
    messages: List[dict[str, str] | dict[str, BaseModel]],
    **kwargs,
) -> BaseModel:
    default_options = {
        "model": "gpt-4o-mini-2024-07-18",
        "temperature": 0.5,
        "response_model": PlainText,
    }
    options = default_options | kwargs  # override defaults with kwargs

    for message in messages:
        assert (
            isinstance(message, dict) and "content" in message
        ), "Messages must be dictionaries with a 'content' key"
        message["content"] = serialize_if_pydantic(message["content"])

    print(f"options: {options}")
    print(f"messages: {messages}")
    async with global_llm_semaphore:
        return await query_api_chat(messages=messages, **options)


def answer_messages_sync(
    messages: List[dict[str, str] | dict[str, BaseModel]],
    **kwargs,
) -> BaseModel:
    for message in messages:
        assert (
            isinstance(message, dict) and "content" in message
        ), "Messages must be dictionaries with a 'content' key"
        message["content"] = serialize_if_pydantic(message["content"])

    options = {
        "model": "gpt-4o-mini-2024-07-18",
        "temperature": 0.5,
        "response_model": PlainText,
    } | kwargs

    return query_api_chat_sync(messages=messages, **options)


@pydantic_cache
@costly(simulator=LLM_Simulator.simulate_llm_call)
@logfire.instrument("query_api_text", extract_args=True)
async def query_api_text(
    model: str, text: str, verbose=False, cost_log: Costlog = GLOBAL_COST_LOG, **kwargs
) -> str:
    client, client_name = get_client_pydantic(model, use_async=True)
    response, completion = await client.completions.create_with_completion(
        model=model, prompt=text, **kwargs
    )
    response_text = response.choices[0].text
    if verbose or os.getenv("VERBOSE") == "True":
        print("Text:", text[:30], "\nResponse:", response_text[:30])

    try:
        cost_info = {
            "input_tokens": completion.usage.prompt_tokens,
            "output_tokens": completion.usage.completion_tokens,
        }
    except AttributeError:
        cost_info = {
            "input_tokens": completion.usage.input_tokens,
            "output_tokens": completion.usage.output_tokens,
        }

    return CostlyResponse(output=response_text, cost_info=cost_info)


@costly(simulator=LLM_Simulator.simulate_llm_call)
@logfire.instrument("query_api_text_sync", extract_args=True)
def query_api_text_sync(model: str, text: str, verbose=False, **kwargs) -> str:
    client, client_name = get_client_pydantic(model, use_async=False)
    response, completion = client.completions.create_with_completion(
        model=model, prompt=text, **kwargs
    )
    response_text = response.choices[0].text
    if verbose or os.getenv("VERBOSE") == "True":
        print("Text:", text, "\nResponse:", response_text)

    try:
        cost_info = {
            "input_tokens": completion.usage.prompt_tokens,
            "output_tokens": completion.usage.completion_tokens,
        }
    except AttributeError:
        cost_info = {
            "input_tokens": completion.usage.input_tokens,
            "output_tokens": completion.usage.output_tokens,
        }

    return CostlyResponse(output=response_text, cost_info=cost_info)


@logfire.instrument("query_parse_last_response_into_format", extract_args=True)
async def query_parse_last_response_into_format(
    messages: list[dict[str, str]],
    response_model: BaseModel,
    verbose: bool = False,
    model: str | None = None,
    **kwargs,
) -> BaseModel:
    parsing_messages = messages + [
        {
            "role": "user",
            "content": (
                "Now parse the latest response into the specified Pydantic model:\n\n"
                f"{response_model.model_fields=}"
            ),
        },
    ]

    parsed_response = await query_api_chat(
        messages=parsing_messages,
        response_model=response_model,
        verbose=verbose,
        model=model,
        **kwargs,
    )

    return parsed_response


@logfire.instrument("query_parse_last_response_into_format_sync", extract_args=True)
def query_parse_last_response_into_format_sync(
    messages: list[dict[str, str]],
    response_model: BaseModel,
    verbose: bool = False,
    model: str | None = None,
    **kwargs,
) -> BaseModel:
    parsing_messages = messages + [
        {
            "role": "user",
            "content": (
                "Now parse the latest response into the specified Pydantic model:\n\n"
                f"{response_model.model_fields=}"
            ),
        },
    ]

    response = query_api_chat_sync(
        messages=parsing_messages,
        response_model=response_model,
        verbose=verbose,
        model=model,
        **kwargs,
    )
    return response


def system_message_addition_for_parsing(response_model: BaseModel) -> str:
    return f"""\
Note: unless explicitly stated in the prompt, do not worry about the exact formatting of the output.
There will be an extra step that will summarize your output into the final answer format.
For context, the final answer format is described by the following Pydantic model:
{response_model.model_fields=}\n
Again, just try to answer the question as best as you can, with all the necessary information; the output will be cleaned up in the final step.
"""


@logfire.instrument("query_api_chat_with_parsing", extract_args=True)
async def query_api_chat_with_parsing(
    messages: list[dict[str, str]],
    response_model: BaseModel,
    verbose: bool = False,
    model: str | None = None,
    parsing_model: str | None = None,
    **kwargs,
) -> BaseModel:
    """
    Runs a native call using the specified model, then parses the output into the desired Pydantic model.
    """
    system_message_addition = system_message_addition_for_parsing(response_model)
    if messages[0]["role"] != "system":
        messages = [{"role": "system", "content": system_message_addition}] + messages
    else:
        messages[0]["content"] += "\n\n" + system_message_addition

    native_output: str = await query_api_chat_native(
        messages=messages,
        verbose=verbose,
        model=model,
        **kwargs,
    )

    if verbose or os.getenv("VERBOSE") == "True":
        print(f"Native output: {native_output}")

    messages.append({"role": "assistant", "content": native_output})

    parsed_response = await query_parse_last_response_into_format(
        messages=messages,
        response_model=response_model,
        verbose=verbose,
        model=parsing_model,
        **kwargs,
    )
    if verbose or os.getenv("VERBOSE") == "True":
        print(f"Parsed response: {parsed_response}")

    return parsed_response


@logfire.instrument("query_api_chat_sync_with_parsing", extract_args=True)
def query_api_chat_sync_with_parsing(
    messages: list[dict[str, str]],
    response_model: BaseModel,
    verbose: bool = False,
    model: str | None = None,
    parsing_model: str | None = None,
    **kwargs,
) -> BaseModel:
    """
    Runs a native call using the specified model, then parses the output into the desired Pydantic model.
    """
    system_message_addition = system_message_addition_for_parsing(response_model)

    system_message_addition = system_message_addition_for_parsing(response_model)

    if messages[0]["role"] != "system":
        messages = [{"role": "system", "content": system_message_addition}] + messages
    else:
        messages[0]["content"] += "\n\n" + system_message_addition

    native_output: str = query_api_chat_sync_native(
        messages=messages, verbose=verbose, model=model, **kwargs
    )
    if verbose or os.getenv("VERBOSE") == "True":
        print(f"Native output: {native_output}")

    messages.append({"role": "assistant", "content": native_output})
    parsed_response = query_parse_last_response_into_format_sync(
        messages=messages,
        response_model=response_model,
        verbose=verbose,
        model=parsing_model,
        **kwargs,
    )
    if verbose or os.getenv("VERBOSE") == "True":
        print(f"Parsed response: {parsed_response}")
    return parsed_response


async def parallelized_call(
    func: Coroutine,
    data: list[str],
    max_concurrent_queries: int = 100,
) -> list[any]:
    """
    Run async func in parallel on the given data.
    func will usually be a partial which uses query_api or whatever in some way.

    Example usage:
        partial_eval_method = functools.partial(eval_method, model=model, **kwargs)
        results = await parallelized_call(partial_eval_method, [format_post(d) for d in data])
    """

    if os.getenv("SINGLE_THREAD"):
        print(f"Running {func} on {len(data)} datapoints sequentially")
        return [await func(d) for d in data]

    max_concurrent_queries = min(
        max_concurrent_queries,
        int(os.getenv("MAX_CONCURRENT_QUERIES", max_concurrent_queries)),
    )

    print(
        f"Running {func} on {len(data)} datapoints with {max_concurrent_queries} concurrent queries"
    )

    local_semaphore = asyncio.Semaphore(max_concurrent_queries)

    async def call_func(sem, func, datapoint):
        async with sem:
            return await func(datapoint)

    tasks = [call_func(local_semaphore, func, d) for d in data]
    return await asyncio.gather(*tasks)


def get_deterministic_dummy_embedding(text: str, dimension: int = 1536) -> list[float]:
    import hashlib
    import random
    import math
    # Use MD5 to get a deterministic seed from the text
    hasher = hashlib.md5(text.encode('utf-8'))
    seed = int(hasher.hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    
    # Generate random values
    vec = [rng.gauss(0, 1) for _ in range(dimension)]
    # Normalize to unit vector
    norm = math.sqrt(sum(x*x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    else:
        vec = [0.0] * dimension
        vec[0] = 1.0
    return vec


@embeddings_cache
async def get_embedding(
    text: str,
    embedding_model: str = "text-embedding-3-small",
    model: str = "gpt-4o-mini-2024-07-18",
) -> list[float]:
    # model is largely ignored because we currently can't use the same model for both the embedding and the completion
    client, _, _ = get_client_pydantic(model, use_async=True)
    try:
        response = await client.client.embeddings.create(input=text, model=embedding_model)
        return response.data[0].embedding
    except Exception as e:
        safe_e = str(e).encode('ascii', errors='backslashreplace').decode('ascii')
        print(f"[get_embedding] Warning: Embedding API failed ({safe_e}). Falling back to deterministic dummy embedding.")
        dim = 3072 if "large" in embedding_model else 1536
        return get_deterministic_dummy_embedding(text, dimension=dim)


@embeddings_cache
def get_embeddings_sync(
    texts: list[str],
    embedding_model: str = "text-embedding-3-small",
    model: str = "gpt-4o-mini-2024-07-18",
) -> list[list[float]]:
    # model is largely ignored because we currently can't use the same model for both the embedding and the completion
    client, _, _ = get_client_pydantic(model, use_async=False)
    try:
        response = client.client.embeddings.create(input=texts, model=embedding_model)
        return [e.embedding for e in response.data]
    except Exception as e:
        safe_e = str(e).encode('ascii', errors='backslashreplace').decode('ascii')
        print(f"[get_embeddings_sync] Warning: Embedding API failed ({safe_e}). Falling back to deterministic dummy embeddings.")
        dim = 3072 if "large" in embedding_model else 1536
        return [get_deterministic_dummy_embedding(t, dimension=dim) for t in texts]


@embeddings_cache
def get_embedding_sync(
    text: str,
    embedding_model: str = "text-embedding-3-small",
    model: str = "gpt-4o-mini-2024-07-18",
) -> list[float]:
    return get_embeddings_sync([text], embedding_model, model)[0]


# %%
def get_all_cached_requests():
    all_cached = pydantic_cache.storage.get_all(namespace="llm_utils")
    for key, value in all_cached.items():
        print(key, value.decode("utf-8"))
        break
    # connect to redis for any other operations
    import redis
    from .perscache import REDIS_CONFIG_DEFAULT

    r = redis.StrictRedis(**REDIS_CONFIG_DEFAULT)
    keys = r.keys("llm_utils:*")
    print(keys)


if __name__ == "__main__":
    get_all_cached_requests()


# %%
