import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter, 
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    TokenTextSplitter
)

print("\n=== Basic Text Splitting Examples ===")
chunk_size = 26
chunk_overlap = 4

r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)
c_splitter = CharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)

text1 = 'abcdefghijklmnopqrstuvwxyz'
print("\nSplitting text1:", text1)
print("Recursive split result:", r_splitter.split_text(text1))

text2 = 'abcdefghijklmnopqrstuvwxyzabcdefg'
print("\nSplitting text2:", text2)
print("Recursive split result:", r_splitter.split_text(text2))

text3 = "a b c d e f g h i j k l m n o p q r s t u v w x y z"
print("\nSplitting text3:", text3)
print("Recursive split result:", r_splitter.split_text(text3))
print("Character split result:", c_splitter.split_text(text3))

print("\n=== Custom Separator Splitting ===")
c_splitter = CharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separator = ' '
)
print("Character split with space separator:", c_splitter.split_text(text3))

print("\n=== Document Structure Example ===")
some_text = """When writing documents, writers will use document structure to group content. \
This can convey to the reader, which idea's are related. For example, closely related ideas \
are in sentances. Similar ideas are in paragraphs. Paragraphs form a document. \n\n  \
Paragraphs are often delimited with a carriage return or two carriage returns. \
Carriage returns are the "backslash n" you see embedded in this string. \
Sentences have a period at the end, but also, have a space.\
and words are separated by space."""

print(f"Text length: {len(some_text)}")

print("\n=== Different Splitting Approaches ===")
c_splitter = CharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=0,
    separator = ' '
)
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=0, 
    separators=["\n\n", "\n", " ", ""]
)

print("\nCharacter split result:")
print(c_splitter.split_text(some_text))
print("\nRecursive split result:")
print(r_splitter.split_text(some_text))

print("\n=== Advanced Recursive Splitting ===")
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=0,
    separators=["\n\n", "\n", r"\.", " ", ""]
)
print("Recursive split with multiple separators:")
print(r_splitter.split_text(some_text))

print("\n=== PDF Document Splitting ===")
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("/home/aboelmagd/Downloads/ITI/Django/Django - Getting Python in The Web - lecture 1.pdf")
pages = loader.load()

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len
)

docs = text_splitter.split_documents(pages)
print(f"Original pages: {len(pages)}")
print(f"Split chunks: {len(docs)}")

print("\n=== Markdown Header Splitting ===")

markdown_document = """# Title\n\n \
## Chapter 1\n\n \
Hi this is Jim\n\n Hi this is Joe\n\n \
### Section \n\n \
Hi this is Lance \n\n \
## Chapter 2\n\n \
Hi this is Molly"""

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)
md_header_splits = markdown_splitter.split_text(markdown_document)

print("\nMarkdown splits:")
for i, split in enumerate(md_header_splits):
    print(f"\nSplit {i+1}:")
    print(split)