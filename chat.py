import os
import sys
sys.path.append('../..')

try:
    import panel as pn
    import param
    HAS_PANEL = True
except ImportError:
    print("Panel not found. Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    import panel as pn
    import param
    HAS_PANEL = True

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

pn.extension()

def load_db(file_path, k=4):
    """Initialize the database and QA system"""
    print(f"Loading document: {file_path}")
    
    embeddings = OpenAIEmbeddings()
    db = Chroma(
        persist_directory='docs/chroma/',
        embedding_function=embeddings
    )
    
    # Simplified chat model configuration
    llm = ChatOpenAI(
        model='o1-mini'  # Remove model_kwargs to use default configuration
    )
    
    # Create retrieval chain with simpler prompt
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": k}),
        return_source_documents=True,
        verbose=True,
        max_tokens_limit=1000,
        chain_type="stuff",  # Use simpler chain type
        combine_docs_chain_kwargs={
            "prompt": PromptTemplate.from_template(
                """Answer based on the following context:
                Context: {context}
                Question: {question}
                Answer:"""
            )
        }
    )
    
    return qa

class ChatBot(param.Parameterized):
    chat_history = param.List([])
    answer = param.String("")
    db_query = param.String("")
    db_response = param.List([])
    
    def __init__(self, **params):
        super().__init__(**params)
        self.panels = []
        self.loaded_file = "docs/pdf/Django - Getting Python in The Web - lecture 1.pdf"
        self.qa = load_db(self.loaded_file)
    
    def call_load_db(self, count):
        if count == 0 or file_input.value is None:  # init or no file specified :
            return pn.pane.Markdown(f"Loaded File: {self.loaded_file}")
        else:
            file_input.save("temp.pdf")  # local copy
            self.loaded_file = file_input.filename
            button_load.button_style = "outline"
            self.qa = load_db("temp.pdf", 4)
            button_load.button_style = "solid"
        self.clr_history()
        return pn.pane.Markdown(f"Loaded File: {self.loaded_file}")

    def convchain(self, query):
        if not query:
            return pn.WidgetBox(pn.Row('User:', pn.pane.Markdown("", width=600)), scroll=True)
        try:
            result = self.qa.invoke({
                "question": query,
                "chat_history": self.chat_history
            })
            self.chat_history.extend([(query, result["answer"])])
            self.db_query = result.get("generated_question", "")
            self.db_response = result.get("source_documents", [])
            self.answer = result['answer']
            
            # Updated Panel styling
            self.panels.extend([
                pn.Row('User:', pn.pane.Markdown(query, width=600)),
                pn.Row('ChatBot:', pn.pane.Markdown(
                    f'<div style="background-color: #F6F6F6; padding: 10px;">{self.answer}</div>',
                    width=600)
                )
            ])
        except Exception as e:
            print(f"Error in chat: {str(e)}")
            self.answer = "Sorry, I encountered an error. Please try again."
            self.panels.extend([
                pn.Row('User:', pn.pane.Markdown(query, width=600)),
                pn.Row('ChatBot:', pn.pane.Markdown(
                    f'<div style="background-color: #FFE6E6; padding: 10px;">{self.answer}</div>',
                    width=600)
                )
            ])
        inp.value = ''
        return pn.WidgetBox(*self.panels, scroll=True)

    @param.depends('db_query', )
    def get_lquest(self):
        if not self.db_query:
            return pn.Column(
                pn.Row(pn.pane.Markdown(f"Last question to DB:", styles={'background-color': '#F6F6F6'})),
                pn.Row(pn.pane.Str("no DB accesses so far"))
            )
        return pn.Column(
            pn.Row(pn.pane.Markdown(f"DB query:", styles={'background-color': '#F6F6F6'})),
            pn.pane.Str(self.db_query)
        )

    @param.depends('db_response', )
    def get_sources(self):
        if not self.db_response:
            return
        rlist = [pn.Row(pn.pane.Markdown(f"Result of DB lookup:", styles={'background-color': '#F6F6F6'}))]
        for doc in self.db_response:
            rlist.append(pn.Row(pn.pane.Str(doc)))
        return pn.WidgetBox(*rlist, width=600, scroll=True)

    @param.depends('convchain', 'clr_history')
    def get_chats(self):
        if not self.chat_history:
            return pn.WidgetBox(pn.Row(pn.pane.Str("No History Yet")), width=600, scroll=True)
        rlist = [pn.Row(pn.pane.Markdown(f"Current Chat History variable", styles={'background-color': '#F6F6F6'}))]
        for exchange in self.chat_history:
            rlist.append(pn.Row(pn.pane.Str(exchange)))
        return pn.WidgetBox(*rlist, width=600, scroll=True)

    def clr_history(self, count=0):
        self.chat_history = []
        return

cb = ChatBot()

file_input = pn.widgets.FileInput(accept='.pdf')
button_load = pn.widgets.Button(name="Load DB", button_type='primary')
button_clearhistory = pn.widgets.Button(name="Clear History", button_type='warning')
button_clearhistory.on_click(cb.clr_history)
inp = pn.widgets.TextInput(placeholder='Enter text here…')

bound_button_load = pn.bind(cb.call_load_db, button_load.param.clicks)
conversation = pn.bind(cb.convchain, inp)

jpg_pane = pn.pane.Image('./img/convchain.jpg')

tab1 = pn.Column(
    pn.Row(inp),
    pn.layout.Divider(),
    pn.panel(conversation, loading_indicator=True, height=300),
    pn.layout.Divider(),
)
tab2 = pn.Column(
    pn.panel(cb.get_lquest),
    pn.layout.Divider(),
    pn.panel(cb.get_sources),
)
tab3 = pn.Column(
    pn.panel(cb.get_chats),
    pn.layout.Divider(),
)
tab4 = pn.Column(
    pn.Row(file_input, button_load, bound_button_load),
    pn.Row(button_clearhistory, pn.pane.Markdown("Clears chat history. Can use to start a new topic")),
    pn.layout.Divider(),
    pn.Row(jpg_pane.clone(width=400))
)
dashboard = pn.Column(
    pn.Row(pn.pane.Markdown('# ChatWithYourData_Bot')),
    pn.Tabs(('Conversation', tab1), ('Database', tab2), ('Chat History', tab3), ('Configure', tab4))
)

if __name__ == "__main__":
    try:
        print("\nStarting chat interface...")
        pn.serve(dashboard, show=True, port=5006)
    except Exception as e:
        print(f"Error starting panel server: {str(e)}")
        # Fall back to simple display
        print("\nFalling back to simple display...")
        display = dashboard.show()
        print("Access the chat interface at: http://localhost:5006")

