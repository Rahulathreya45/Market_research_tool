# Market Researcher

Market Researcher is a tool which searches the internet and generates comprehensive report in PDF format.

## Deployment Link

- https://huggingface.co/spaces/rahulathreya45/Market_Research_tool

## Tech Stack

- **Langchain** : Framework
- **Google Gemini** : language model
- **Chroma** : Vector DB
- **Tavlily** : Search Engine
- **Beautifulsoup** : Web scraping
- **FPDF** : Report generation
- **Streamlit** : Frontend
- **Huggingface** : Embedding and Deployment

## Features

- Added wikipedia loader
- Parallel web scraping for faster results
- output as a PDF file in well structured manner
- Displays Original Sources

## Run Locally

Clone the project

Go to the project directory

Install dependencies

```bash
  pip install requirements.txt
```

To run this project, you will need to add the following environment variables to your .env file

`GOOGLE_API_KEY`

`TAVILY_API_KEY`

- GOOGLE_API_KEY can be obtained from makersuite.com
- TAVILY_API_KEY can be found in tavily.com

## Roadmap

- Faster Response

- More queries

- Improvements in Report formatting

## Screenshots

![Screenshot 1](screenshots/Screenshot_1.png)

![Screenshot 2](screenshots/Screenshot_2.png)
