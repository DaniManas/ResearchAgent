from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import Tool
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()



search = TavilySearchResults()
search_tool = Tool(
    name="search",
    func=search.run,
    description="Search the web for information"
)

api_wrapper = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=1000)
wiki_tool=WikipediaQueryRun(api_wrapper=api_wrapper)
