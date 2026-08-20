from bs4 import BeautifulSoup
import logfire

def parse_html(file_path: str):
    """
    Parse HTML content using BeautifulSoup
    Cleans scripts, styles and extract readable text for RAG
    """
    with logfire.span("HTML parsing", filename=file_path):
        try:
            with open(file_path, 'r',encoding="UTF-8", errors="ignore") as f:
                content = f.read()
            
            soup = BeautifulSoup(content, "html.parser")
            
            #1. Remove Junk (Scripts, Styles and Metadata)
            for script in soup(["script", "style", "meta", "noscript"]):
                script.decompose()
                
            #2. extract text
            text = soup.get_text(separator="\n")
            
            #3. clean whitespace(collapse multiple newsline)
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split(" "))
            text_clean = "\n".join(chunk for chunk in chunks if chunk)
            
            return text_clean
        except Exception as e:
            logfire.error(f"HTML Parse failed: {e}")
            raise e