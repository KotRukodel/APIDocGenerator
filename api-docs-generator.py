import os
import logging
from typing import List, Optional
import re

from langchain.llms import OpenAI
from langchain.document_loaders import PythonLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('api_doc_generator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class APIDocumentationGenerator:
    """
    Advanced API documentation generator with comprehensive scenario handling.
    
    Enables automatic technical documentation creation using AI technologies.
    Supports code parsing, intelligent splitting, and documentation generation.
    """

    def __init__(
        self, 
        api_module_path: str, 
        openai_api_key: Optional[str] = None,
        model_name: str = 'gpt-3.5-turbo'
    ):
        """
        Initialize documentation generator with extended configuration.

        Args:
            api_module_path (str): Path to API source code file
            openai_api_key (Optional[str]): OpenAI API key. 
                If not provided, searches in environment variables
            model_name (str): Name of GPT model for generation
        """
        # Input parameters validation
        self._validate_input_path(api_module_path)
        
        # Retrieve API key
        self.openai_api_key = (
            openai_api_key or 
            os.getenv('OPENAI_API_KEY')
        )
        
        if not self.openai_api_key:
            raise ValueError(
                "OpenAI API key not found. "
                "Provide it explicitly or set OPENAI_API_KEY environment variable"
            )

        # Language model initialization
        try:
            self.llm = OpenAI(
                temperature=0.3, 
                model_name=model_name,
                openai_api_key=self.openai_api_key
            )
        except Exception as e:
            logger.error(f"Model initialization error: {e}")
            raise

        self.api_module_path = api_module_path

    def _validate_input_path(self, path: str) -> None:
        """
        Validate file path correctness.

        Args:
            path (str): Path to the file

        Raises:
            ValueError: On incorrect path
        """
        if not path or not isinstance(path, str):
            raise ValueError("File path must be a non-empty string")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        
        if not path.endswith('.py'):
            raise ValueError("Only Python files with .py extension are supported")

    def load_api_code(self) -> List:
        """
        Load and preprocess API source code with advanced handling.

        Returns:
            List: List of documents with split code
        """
        try:
            # Python file loading
            loader = PythonLoader(self.api_module_path)
            documents = loader.load()
            logger.info(f"Documents loaded: {len(documents)}")

            # Document splitting with advanced logic
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,     
                chunk_overlap=200,
                length_function=len,  # Precise length calculation
                separators=['\n\n', '\n', ' ', '']  # Separator hierarchy
            )
            texts = text_splitter.split_documents(documents)
            
            # Filtering empty and too short fragments
            texts = [
                text for text in texts 
                if len(text.page_content.strip()) > 50
            ]
            
            logger.info(f"Code fragments prepared: {len(texts)}")
            return texts

        except Exception as e:
            logger.error(f"Code loading error: {e}")
            raise

    def generate_endpoint_description(self, code_snippet: str) -> str:
        """
        Generate description with additional processing and validation.

        Args:
            code_snippet (str): Source code fragment

        Returns:
            str: Generated technical description
        """
        # Code preprocessing
        clean_code = self._preprocess_code_snippet(code_snippet)

        prompt_template = PromptTemplate(
            input_variables=["code"],
            template="""Analyze the following Python code and create a professional technical description of the API endpoint:

            Code:
            {code}

            MANDATORY description sections:
            1. Exact endpoint purpose
            2. Input parameters with types and validation
            3. Return data structure
            4. Possible HTTP status codes
            5. Correct invocation example
            6. Potential errors and exceptions
            """
        )

        try:
            chain = LLMChain(llm=self.llm, prompt=prompt_template)
            documentation = chain.run(code=clean_code)
            
            # Documentation postprocessing
            documentation = self._postprocess_documentation(documentation)
            
            return documentation

        except Exception as e:
            logger.error(f"Documentation generation error: {e}")
            return f"Failed to generate documentation. Details: {e}"

    def _preprocess_code_snippet(self, code: str) -> str:
        """
        Preliminary code cleaning before analysis.

        Args:
            code (str): Source code fragment

        Returns:
            str: Cleaned code
        """
        # Comments removal
        code = re.sub(r'#.*', '', code)
        
        # Removing extra spaces and line breaks
        code = '\n'.join(line.strip() for line in code.split('\n') if line.strip())
        
        return code

    def _postprocess_documentation(self, doc: str) -> str:
        """
        Documentation postprocessing.

        Args:
            doc (str): Generated documentation text

        Returns:
            str: Processed text
        """
        # Removing extra spaces
        doc = re.sub(r'\n{3,}', '\n\n', doc)
        
        # Adding headers for structure
        doc = f"## API Endpoint\n\n{doc}"
        
        return doc.strip()

    def generate_full_api_docs(self) -> str:
        """
        Generate complete documentation for entire API.

        Returns:
            str: Full technical documentation
        """
        try:
            api_code_parts = self.load_api_code()
            full_documentation = []

            # Parallel generation with limitation
            for part in api_code_parts[:10]:  # Limit to prevent excessive costs
                endpoint_doc = self.generate_endpoint_description(part.page_content)
                full_documentation.append(endpoint_doc)

            # Documentation combining
            result = "\n\n---\n\n".join(full_documentation)
            
            logger.info(f"Endpoint descriptions generated: {len(full_documentation)}")
            return result

        except Exception as e:
            logger.error(f"Critical documentation generation error: {e}")
            return f"Failed to generate complete documentation. Reason: {e}"

def main():
    """Demonstrate generator operation with various scenario handling."""
    try:
        # Configuration example
        generator = APIDocumentationGenerator(
            api_module_path='path/to/your/api_module.py',
            model_name='gpt-3.5-turbo'
        )

        # Generation with time measurement
        import time
        start_time = time.time()
        
        api_documentation = generator.generate_full_api_docs()
        
        execution_time = time.time() - start_time
        logger.info(f"Documentation generated in {execution_time:.2f} sec")

        # Saving with encoding
        with open('api_documentation.md', 'w', encoding='utf-8') as f:
            f.write(api_documentation)
        
        print("Documentation successfully generated and saved.")

    except Exception as e:
        logger.error(f"Critical error in main function: {e}")
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
