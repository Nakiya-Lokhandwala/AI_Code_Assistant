import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

class LangChainClient:
    def __init__(self, mode="Debugger", gemini_version="2.5-flash"):
        """
        mode: one of your assistant modes (General, Debugger, etc.)
        gemini_version: string, e.g. "1.5-flash", "1.5-pro", "2.0-flash", "2.5-flash", "2.5-pro"
        """
        role_prompts = {
            "Code Generator": "You are a senior developer. Generate high-quality, production-ready code.",
            "Debugger": "You are a debugging assistant. Find issues, explain them, and suggest fixes.",
            "Code Guide": "You are a mentor. Teach coding best practices step by step.",
            "Explain Code": "You are a teacher. Explain code in simple terms with examples.",
            "Documentation": "You are a technical writer. Generate professional documentation."
        }

        self.system_prompt = role_prompts.get(mode, role_prompts["Debugger"])

        # Determine model name string
        # You may need to use “preview” suffixes in some cases (e.g. “gemini-2.5-pro-preview-03-25”) 
        # depending on Google’s rollout / your API access.
        model_name = f"gemini-{gemini_version}"

        # Instantiate ChatGoogleGenerativeAI with appropriate model
        self.model = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.2,
            api_key=os.getenv("GOOGLE_API_KEY")
        )

    def chat(self, messages):
        """Send messages to the Gemini model via LangChain and return reply."""
        prompt_messages = [SystemMessage(content=self.system_prompt)]
        for m in messages:
            if m["role"] == "user":
                prompt_messages.append(HumanMessage(content=m["content"]))
            else:
                prompt_messages.append(AIMessage(content=m["content"]))

        resp = self.model.invoke(prompt_messages)
        return resp.content
