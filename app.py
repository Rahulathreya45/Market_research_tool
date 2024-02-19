import getpass
import os
import json
import bs4
import re
import nest_asyncio
from fpdf import FPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()

nest_asyncio.apply()  # for parallel scraping of text from the websites
os.environ["TAVILY_API_KEY"] = "your api key"


class PDF(FPDF):
    """This class extends FPDF for creating basic PDF documents with chapters.

    Attributes:
        None

    Methods:
        chapter_title(self, title):
            Creates a formatted chapter title in the PDF.
        chapter_body(self, body):
            Creates a formatted chapter body text in the PDF.
    """

    def chapter_title(self, title):
        """
        Creates a formatted chapter title in the PDF.

        Args:
            title (str): The text of the chapter title.

        Returns:
            None
        """
        self.set_font("Arial", "B", 12)  # Set font to Arial, bold, and size 12
        self.multi_cell(
            0, 5, title
        )  # Write the title with variable width and line height
        self.ln(1)  # Add a line break after the title

    def chapter_body(self, body):
        """
        Creates a formatted chapter body text in the PDF.

        Args:
            body (str): The text of the chapter body.

        Returns:
            None
        """
        self.set_font("Arial", "", 10)  # Set font to Arial, regular, and size 10
        self.multi_cell(
            0, 5, body
        )  # Write the body text with variable width and line height
        self.ln()  # Add a line break after the body text


# List of  queries that are fed into the search engine
queries = [
    "who are canno's customers and what is thier customer retention strategy and customer support",
    "What are canoo's strengths, weaknesses, opportunities, and threats",
    "who are the key players in EV industry and what including their market share, products or services offered",
    "Who are Canoo's main competitors and key players in its industry ",
    "What are the key trends in consumer behavior, technological advancements in  Canoo's industry?",
    "What is Canoo's recent revenue, profit margin, return on investment and market share growth rate?",
    "What is canoo's recent target market,products or services features, benefits,",
    "what are canoo's recent marketing,sales and pricing strategy with numbers",
    "give details about canoo's recent partnerships, collaborations and it's innovation",
]


def search_store_links(queries):
    """Searches the internet for links related to the given queries using a search tool and stores the links in a dictionary.

    Args:
        queries (list): A list of queries to search for.

    Returns:
        str: A JSON string containing the search results where each query maps to a list of related links.
    """
    results = {}
    tool = TavilySearchResults()
    for query in queries:
        result_list = tool.invoke({"query": query})
        links = [item["url"] for item in result_list]

        results[query] = links

    return json.dumps(results, indent=5)


# Storing links in JSON file
with open("data.json", "w") as json_file:
    json_file.write(search_store_links(queries))

with open("data.json", "r") as json_file:
    data = json.load(json_file)


def remove_duplicates(data):
    """
    Removes duplicate URLs from a dictionary containing lists of URLs mapped to keys.

    Args:
        data (dict): A dictionary where keys are query strings and values are lists of URLs.

    Returns:
        dict: A dictionary with duplicate URLs removed, where each key maps to a list of unique URLs.

    """
    unique_urls = set()
    cleaned_data = {}
    for key, urls in data.items():
        for url in urls:
            if url not in unique_urls:
                unique_urls.add(url)
                if key in cleaned_data:
                    cleaned_data[key].append(url)
                else:
                    cleaned_data[key] = [url]
    return cleaned_data


urls = remove_duplicates(data)


def url_to_doc(urljson):
    """
    Retrieves text content from URLs specified in a JSON object.

    Args:
        urljson (dict): A dictionary where keys are query strings and values are URLs.

    Returns:
        list: A list of text documents retrieved from the specified URLs.
    """
    for query, url in urljson.items():
        loader = WebBaseLoader(url)
        loader.requests_per_second = 1
        document = loader.aload()

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", ","], chunk_size=1000, chunk_overlap=100
        )
        docs = text_splitter.split_documents(document)

    return docs


docs = url_to_doc(urls)
# Loads data from the wikipedia page
wikidocs = WikipediaLoader(query="Canoo", load_max_docs=5).load()
docs.extend(wikidocs)


def format_docs(docs):
    """
    Formats a list of text documents.

    Args:
        docs (list): A list of text documents.

    Returns:
        str: Formatted text content.
    """
    return (
        "\n\n".join(doc.page_content for doc in docs)
        .encode("ascii", "ignore")
        .decode("ascii")
    )


def remove_markdown_symbols(text):
    """
    Removes Markdown symbols from the given text.

    Args:
        text (str): The input text containing Markdown symbols.

    Returns:
        str: Text with Markdown symbols removed
    """
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"#", "", text)
    text = text.strip()

    return text


def doc_to_vectorstore(clean_docs):
    """
    Converts cleaned documents into a vector store using pre-trained sentence embeddings.

    Args:
        clean_docs (list): A list of cleaned document texts.

    Returns:
        Chroma: A Chroma object containing document vectors.
    """
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(docs, embedding_function)
    return db


db = doc_to_vectorstore(docs)
retriever = db.as_retriever()
llm = GoogleGenerativeAI(model="gemini-pro")
template = """
You are tasked with generating a market analysis report in plain text format, avoiding symbols like * and #. Use the retrieved context from the vector store and the provided user query to answer the question comprehensively.

Context: {context} 

User Query: {question} 

Report Structure:
 Avoid the use of Symbols such as **,#
 Avoid the use of headings and subheadings. 
 The report should be structured in paragraphs, separated by newline characters.
 Use numbered lists where necessary to present information clearly.
 Ensure the report addresses the user query directly and provides clear insights using information from the context.
 Quantify your analysis with data from the context whenever possible.
 Aim for a minimum length of 1200 words, but extend the report as much as possible with relevant and necessary information.
 Include a list of sources used at the end of the report, ensuring each source is referenced only once.

Note: If insufficient context is available to answer the question comprehensively, state this clearly in the report.

---

Output:

Generate a plain text report following the above structure, avoiding the use of *, #, or markdown formatting entirely. Use numbered lists only where necessary to enhance clarity. Strive for a minimum length of 1200 words, but prioritize providing comprehensive and relevant information
"""

prompt = PromptTemplate.from_template(template)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
# Queries for the report
reports = [
    "Overview of canoo",
    "Define the primary industry Canoo operates in, outlining its size, growth trajectory, and key players",
    "Canoo's target market, including demographic information such as age, gender, income level",
    "Pinpoint the major trends and emerging technologies reshaping this industry, assessing how Canoo adapts and leverages them strategically",
    "Canoo's product offerings, pricing strategies, and marketing efforts against its key competitors",
    "Canoo's financial performance over recent years, focusing on revenue, profitability, and investment returns."
    "the primary cost drivers for Canoo, exploring potential strategies for optimizing its cost structure and enhancing profitability",
    "Canno's sales strategy, including its sales processes, distribution channels, and sales force",
    "Conduct a SWOT analysis to identify the Tesla's strengths, weaknesses, opportunities, and threats",
]
res = []
for questions in reports:
    res.append(remove_markdown_symbols(rag_chain.invoke(questions)))

pdf = PDF()
pdf.add_page()

for query, content in zip(reports, res):
    pdf.chapter_title(query)
    pdf.chapter_body(content)

pdf.output("output.pdf")
