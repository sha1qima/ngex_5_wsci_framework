from pathlib import Path
from ollama import chat
import json



question = """
I changed my university password this morning.
Now my Windows laptop won't connect to campus Wi-Fi,
but my phone still works.
"""

## WRITE ##
service_status = {
    "wifi": "operational"
}

state = {
    "problem": question,
    "wi_fi_status": "operational",
    "wi_fi_check": True
}

with open("state.json", "w") as file:
    json.dump(
        state,
        file,
        indent=2
    )

with open("state.json", "r") as file:
    state = json.load(file)

print(state)


## SELECT CONTEXT FILES BASED ON QUESTION
## Create the function that takes the student's question, takes some keywords and chooses the relevant files from the knowledge base. Return a list of the selected files.
## For example, if the question has the kyeword "print" or "printer", then the function should return the file "knowledge/printer_setup.txt" in a list.
def select_context(question):
    question_lower = question.lower()
    keyword_map = {
        "printer": "knowledge/printing.txt",
        "print": "knowledge/printing.txt",
        "projector": "knowledge/classroom_projectors.txt",
        "display": "knowledge/classroom_projectors.txt",
        "hdmi": "knowledge/classroom_projectors.txt",
        "email": "knowledge/email_setup.txt",
        "mail": "knowledge/email_setup.txt",
        "vpn": "knowledge/vpn.txt",
        "wi-fi": "knowledge/wifi_setup.txt",
        "wifi": "knowledge/wifi_setup.txt",
        "eduroam": "knowledge/wifi_setup.txt",
        "password": "knowledge/password_changes.txt",
        "credential": "knowledge/password_changes.txt",
        "login": "knowledge/password_changes.txt",
        "network": "knowledge/wifi_setup.txt",
        "connect": "knowledge/wifi_setup.txt",
    }

    selected = set()
    for keyword, filepath in keyword_map.items():
        if keyword in question_lower:
            selected.add(filepath)

    selected.add("knowledge/service_status.txt")
    selected.add("knowledge/password_changes.txt")
    return sorted(selected)

selected_files = select_context(question)

## READ SELECTED FILES and add their contents to the context variable.
context = ""
for file_path in selected_files:
    context += Path(file_path).read_text()
    context += "\n\n"

## 
## COMPRESS CONTEXT
## Add logic to compress the context from above by calling Qwen with "context" and the "question" as the parameter
## The response from Qwen should be the compressed context. Store it in a variable called "compressed_context" 

def compress_context(context, question):
    compression_prompt = f"""
You are a context compression assistant.

Given the following knowledge base and a student's question, extract ONLY the information that is directly relevant to solving the question. Remove all unrelated content. Be concise but complete.

Knowledge base:
{context}

Student question:
{question}

Return ONLY the compressed relevant information, nothing else.
"""
    response = chat(
        model = "qwen3:8b",
        messages = [
            {
                "role": "user",
                "content": compression_prompt
            }
        ],
    )
    return response.message.content

compressed_context = compress_context(context, question)

## Print the length of the compressed context
print(len(compressed_context))
print(compressed_context[:300], "...")
## Now, call Qwen again with the compressed context and the student's question. Store the response in a variable called "response" and print the response from Qwen.
## Ensure the model produces a structured output 
response = chat(
    model="qwen3:8b",
    messages=[
        {
            "role": "system",
            "content": (
                "You are a university IT support assistant. "
                "Answer the student's question using ONLY the provided compressed context. "
                "Return your answer as structured JSON with keys: "
                "'diagnosis', 'steps', 'expected_outcome'."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Compressed context:\n{compressed_context}\n\n"
                f"Student question:\n{question}"
            ),
        },
    ],
    format="json",
)

print(response.message.content)

## WRITE the above output in an artifact called "state"
try:
    answer_data = json.loads(response.message.content)
except json.JSONDecodeError:
    answer_data = {"raw_answer": response.message.content}

state.update({
    "diagnosis": answer_data.get("diagnosis", ""),
    "steps": answer_data.get("steps", []),
    "expected_outcome": answer_data.get("expected_outcome", ""),
    "compressed_context": compressed_context,
    "selected_files": selected_files,
})

with open("state.json", "w") as file:
    json.dump(state, file, indent=2)

## Update the rest of the code so that it uses the "state" artifact as part of the context. 
## It is important to ensure that the model uses only the relevant parts from the "state" artifact and not the entire artifact.
## For this, you may have to think of a good structure for the "state" artifact and how to use it in the context.
def build_isolated_context(state, task="diagnose"):
    if task == "diagnose":
        return {
            "problem": state["problem"],
            "wi_fi_status": state["wi_fi_status"],
            "diagnosis": state.get("diagnosis", ""),
            "steps": state.get("steps", []),
        }
    elif task == "report":
        return {
            "wi_fi_status": state["wi_fi_status"],
            "wi_fi_check": state["wi_fi_check"],
        }
    return {}

diagnostic_context = build_isolated_context(state, task="diagnose")
report_context = build_isolated_context(state, task="report")

print(json.dumps(diagnostic_context, indent=2))
print(json.dumps(report_context, indent=2))

