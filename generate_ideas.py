import json
import os
import os.path as osp
import time
from typing import List, Dict, Union

import backoff
import requests
from pydantic import BaseModel
from utils import get_response_from_llm, extract_json_between_markers

S2_API_KEY = os.getenv("S2_API_KEY")


class QuestionGenerationResponse(BaseModel):
    # Define the schema for the response
    question: str = ""
    motivation: str = ""
    interestingness: int = 0
    feasibility: int = 0
    novelty: int = 0


idea_first_prompt = """{topic_description}
Here are the research questions that you have already generated:
'''
{prev_ideas_string}
'''

Come up with the next impactful and creative research question within the scope of the provided topic. Make sure the question is novel, feasible, and interesting.
Respond in the following format:
```json
{{
    "question": <The research question you propose>,
    "motivation": <Why is this question important?>,
    "interestingness": <A rating from 1 to 10 (lowest to highest)>,
    "feasibility": <A rating from 1 to 10 (lowest to highest)>,
    "novelty": <A rating from 1 to 10 (lowest to highest)>
}}
```

Be cautious and realistic on your ratings.
This JSON will be automatically parsed, so ensure the format is precise.
You will have {num_reflections} rounds to iterate on the idea, but do not need to use them all.
"""

idea_reflection_prompt = """Round {current_round}/{num_reflections}.
In your thoughts, first carefully consider the quality, novelty, and feasibility of the question you just created.
Include any other factors that you think are important in evaluating the question.
Ensure the question is clear and concise, and the JSON is the correct format.
Do not make things overly complicated.
In the next attempt, try and refine and improve your question.
Stick to the spirit of the original question unless there are glaring issues.
Respond in the same format as before.

If there is nothing to improve, simply repeat the previous JSON EXACTLY."""


# GENERATE IDEAS
async def generate_ideas(
    base_dir,
    result_dir,
    result_file,
    client,
    model,
    skip_generation=False,
    max_num_generations=20,
    num_reflections=5,
):
    if skip_generation:
        # Load existing ideas from file
        try:
            with open(osp.join(result_dir, result_file), "r") as f:
                ideas = json.load(f)
            print("Loaded existing ideas:")
            for idea in ideas:
                print(idea)
            return ideas
        except FileNotFoundError:
            print("No existing ideas found. Generating new ideas.")
        except json.JSONDecodeError:
            print("Error decoding existing ideas. Generating new ideas.")

    idea_str_archive = []
    with open(osp.join(base_dir, "seed_ideas.json"), "r") as f:
        seed_ideas = json.load(f)
    for seed_idea in seed_ideas:
        idea_str_archive.append(json.dumps(seed_idea))

    with open(osp.join(base_dir, "prompt.json"), "r") as f:
        prompt = json.load(f)

    idea_system_prompt = prompt["system"]

    for _ in range(max_num_generations):
        print()
        print(f"Generating idea {_ + 1}/{max_num_generations}")
        try:
            prev_ideas_string = "\n\n".join(idea_str_archive)

            msg_history = []
            print(f"Iteration 1/{num_reflections}")
            output, msg_history = await get_response_from_llm(
                msg=idea_first_prompt.format(
                    topic_description=prompt["topic_description"],
                    prev_ideas_string=prev_ideas_string,
                    num_reflections=num_reflections,
                ),
                client=client,
                model=model,
                system_message=idea_system_prompt,
                msg_history=msg_history,
                response_format=QuestionGenerationResponse,
            )
            ## PARSE OUTPUT
            json_output = output.model_dump()
            assert json_output is not None, "Failed to extract JSON from LLM output"
            output_question = json_output["question"]

            # Iteratively improve task.
            if num_reflections > 1:
                for j in range(num_reflections - 1):
                    print(f"Iteration {j + 2}/{num_reflections}")
                    text, msg_history = await get_response_from_llm(
                        msg=idea_reflection_prompt.format(
                            current_round=j + 2,
                            num_reflections=num_reflections,
                        ),
                        client=client,
                        model=model,
                        system_message=idea_system_prompt,
                        msg_history=msg_history,
                        response_format=QuestionGenerationResponse,
                    )
                    ## PARSE OUTPUT
                    json_output = text.model_dump()
                    assert (
                        json_output is not None
                    ), "Failed to extract JSON from LLM output"
                    if json_output["question"] == output_question:
                        print(f"Idea generation converged after {j + 2} iterations.")
                        break
                    output_question = json_output["question"]

            idea_str_archive.append(json.dumps(json_output))
        except Exception as e:
            print(f"Failed to generate idea: {e}")
            continue

    ## SAVE IDEAS
    ideas = []
    for idea_str in idea_str_archive:
        ideas.append(json.loads(idea_str))

    with open(osp.join(result_dir, result_file), "w") as f:
        json.dump(ideas, f, indent=4)

    return ideas


# GENERATE IDEAS OPEN-ENDED
async def generate_next_idea(
    base_dir,
    result_dir,
    result_file,
    client,
    model,
    prev_idea_archive=[],
    num_reflections=5,
    max_attempts=10,
):
    idea_archive = prev_idea_archive
    original_archive_size = len(idea_archive)

    print(f"Generating idea {original_archive_size + 1}")

    if len(prev_idea_archive) == 0:
        print(f"First iteration, taking seed ideas")
        # seed the archive on the first run with pre-existing ideas
        with open(osp.join(base_dir, "seed_ideas.json"), "r") as f:
            seed_ideas = json.load(f)
        for seed_idea in seed_ideas[:1]:
            idea_archive.append(seed_idea)
    else:
        with open(osp.join(base_dir, "prompt.json"), "r") as f:
            prompt = json.load(f)
        idea_system_prompt = prompt["system"]

        for _ in range(max_attempts):
            try:
                idea_strings = []
                for idea in idea_archive:
                    idea_strings.append(json.dumps(idea))
                prev_ideas_string = "\n\n".join(idea_strings)

                msg_history = []
                print(f"Iteration 1/{num_reflections}")
                output, msg_history = await get_response_from_llm(
                    msg=idea_first_prompt.format(
                        topic_description=prompt["topic_description"],
                        prev_ideas_string=prev_ideas_string,
                        num_reflections=num_reflections,
                    ),
                    client=client,
                    model=model,
                    system_message=idea_system_prompt,
                    msg_history=msg_history,
                    response_format=QuestionGenerationResponse,
                )
                ## PARSE OUTPUT
                json_output = output.model_dump()
                assert json_output is not None, "Failed to extract JSON from LLM output"
                output_question = json_output["question"]

                # Iteratively improve task.
                if num_reflections > 1:
                    for j in range(num_reflections - 1):
                        print(f"Iteration {j + 2}/{num_reflections}")
                        output, msg_history = await get_response_from_llm(
                            msg=idea_reflection_prompt.format(
                                current_round=j + 2, num_reflections=num_reflections
                            ),
                            client=client,
                            model=model,
                            system_message=idea_system_prompt,
                            msg_history=msg_history,
                            response_format=QuestionGenerationResponse,
                        )
                        ## PARSE OUTPUT
                        json_output = output.model_dump()
                        assert (
                            json_output is not None
                        ), "Failed to extract JSON from LLM output"
                        if json_output["question"] == output_question:
                            print(
                                f"Idea generation converged after {j + 2} iterations."
                            )
                            break
                        output_question = json_output["question"]

                idea_archive.append(json_output)
                break
            except Exception as e:
                print(f"Failed to generate idea: {e}")
                continue

    ## SAVE IDEAS
    with open(osp.join(result_dir, result_file), "w") as f:
        json.dump(idea_archive, f, indent=4)

    return idea_archive


def on_backoff(details):
    print(
        f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries "
        f"calling function {details['target'].__name__} at {time.strftime('%X')}"
    )


@backoff.on_exception(
    backoff.expo, requests.exceptions.HTTPError, on_backoff=on_backoff
)
def search_for_papers(
    query, result_limit=10, engine="semanticscholar"
) -> Union[None, List[Dict]]:
    if not query:
        return None
    if engine == "semanticscholar":
        assert (
            S2_API_KEY is not None and S2_API_KEY != ""
        ), "Please set S2_API_KEY environment variable!"
        rsp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            headers={"X-API-KEY": S2_API_KEY} if S2_API_KEY else {},
            params={
                "query": query,
                "limit": result_limit,
                "fields": "title,authors,venue,year,abstract,citationStyles,citationCount",
            },
        )
        print(f"Response Status Code: {rsp.status_code}")
        print(
            f"Response Content: {rsp.text[:500]}"
        )  # Print the first 500 characters of the response content
        rsp.raise_for_status()
        results = rsp.json()
        total = results["total"]
        time.sleep(1.0)
        if not total:
            return None

        papers = results["data"]
        return papers
    elif engine == "openalex":
        import pyalex
        from pyalex import Work, Works

        mail = os.environ.get("OPENALEX_MAIL_ADDRESS", None)
        if mail is None:
            print(
                "[WARNING] Please set OPENALEX_MAIL_ADDRESS for better access to OpenAlex API!"
            )
        else:
            pyalex.config.email = mail

        def extract_info_from_work(
            work: Work, max_abstract_length: int = 1000
        ) -> dict[str, str]:
            # "Unknown" is returned when venue is unknown...
            venue = "Unknown"
            for i, location in enumerate(work["locations"]):
                if location["source"] is not None:
                    venue = location["source"]["display_name"]
                    if venue != "":
                        break
            title = work["title"]
            abstract = work["abstract"]
            if abstract is None:
                abstract = ""
            if len(abstract) > max_abstract_length:
                # To avoid context length exceed error.
                print(
                    f"[WARNING] {title=}: {len(abstract)=} is too long! Use first {max_abstract_length} chars."
                )
                abstract = abstract[:max_abstract_length]
            authors_list = [
                author["author"]["display_name"] for author in work["authorships"]
            ]
            authors = (
                " and ".join(authors_list)
                if len(authors_list) < 20
                else f"{authors_list[0]} et al."
            )
            paper = dict(
                title=title,
                authors=authors,
                venue=venue,
                year=work["publication_year"],
                abstract=abstract,
                citationCount=work["cited_by_count"],
            )
            return paper

        works: List[Dict] = Works().search(query).get(per_page=result_limit)
        papers: List[Dict[str, str]] = [extract_info_from_work(work) for work in works]
        return papers
    else:
        raise NotImplementedError(f"{engine=} not supported!")


novelty_system_msg = """You are an ambitious AI PhD student who is looking to publish a paper that will contribute significantly to the field.
You have an idea and you want to check if it is novel or not. I.e., not overlapping significantly with existing literature or already well explored.
Be a harsh critic for novelty, ensure there is a sufficient contribution in the idea for a new conference or workshop paper.
You will be given access to the Semantic Scholar API, which you may use to survey the literature and find relevant papers to help you make your decision.
The top 10 results for any search query will be presented to you with the abstracts.

You will be given {num_rounds} to decide on the paper, but you do not need to use them all.
At any round, you may exit early and decide on the novelty of the idea.
Decide a paper idea is novel if after sufficient searching, you have not found a paper that significantly overlaps with your idea.
Decide a paper idea is not novel, if you have found a paper that significantly overlaps with your idea.

{topic_description}
"""

novelty_prompt = '''Round {current_round}/{num_rounds}.
You have this idea:

"""
{idea}
"""

The results of the last query are (empty on first round):
"""
{last_query_results}
"""

Respond in the following format:

THOUGHT:
<THOUGHT>

RESPONSE:
```json
<JSON>
```

In <THOUGHT>, first briefly reason over the idea and identify any query that could help you make your decision.
If you have made your decision, add "Decision made: novel." or "Decision made: not novel." to your thoughts.

In <JSON>, respond in JSON format with ONLY the following field:
- "Query": An optional search query to search the literature (e.g. attention is all you need). You must make a query if you have not decided this round.

A query will work best if you are able to recall the exact name of the paper you are looking for, or the authors.
This JSON will be automatically parsed, so ensure the format is precise.'''


async def check_idea_novelty(
    ideas,
    base_dir,
    result_dir,
    result_file,
    client,
    model,
    max_num_iterations=10,
    engine="semanticscholar",
):
    with open(osp.join(base_dir, "prompt.json"), "r") as f:
        prompt = json.load(f)
        topic_description = prompt["topic_description"]

    for idx, idea in enumerate(ideas):
        if "novel" in idea:
            print(f"Skipping idea {idx}, already checked.")
            continue

        print(f"\nChecking novelty of idea {idx}: {idea['Name']}")

        novel = False
        msg_history = []
        papers_str = ""

        for j in range(max_num_iterations):
            try:
                text, msg_history = await get_response_from_llm(
                    novelty_prompt.format(
                        current_round=j + 1,
                        num_rounds=max_num_iterations,
                        idea=idea,
                        last_query_results=papers_str,
                    ),
                    client=client,
                    model=model,
                    system_message=novelty_system_msg.format(
                        num_rounds=max_num_iterations,
                        topic_description=topic_description,
                    ),
                    msg_history=msg_history,
                )
                if "decision made: novel" in text.lower():
                    print("Decision made: novel after round", j)
                    novel = True
                    break
                if "decision made: not novel" in text.lower():
                    print("Decision made: not novel after round", j)
                    break

                ## PARSE OUTPUT
                json_output = extract_json_between_markers(text)
                assert json_output is not None, "Failed to extract JSON from LLM output"

                ## SEARCH FOR PAPERS
                query = json_output["Query"]
                papers = search_for_papers(query, result_limit=10, engine=engine)
                if papers is None:
                    papers_str = "No papers found."
                else:
                    paper_strings = []
                    for i, paper in enumerate(papers):
                        paper_strings.append(
                            """{i}: {title}. {authors}. {venue}, {year}.\nNumber of citations: {cites}\nAbstract: {abstract}""".format(
                                i=i,
                                title=paper["title"],
                                authors=paper["authors"],
                                venue=paper["venue"],
                                year=paper["year"],
                                cites=paper["citationCount"],
                                abstract=paper["abstract"],
                            )
                        )
                    papers_str = "\n\n".join(paper_strings)

            except Exception as e:
                print(f"Error: {e}")
                continue

        idea["novel"] = novel

    # Save results to JSON file
    results_file = osp.join(result_dir, result_file)
    with open(results_file, "w") as f:
        json.dump(ideas, f, indent=4)

    return ideas
