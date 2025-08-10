import os
import json
import traceback
import uuid
import zipfile
import html
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, session, send_from_directory
from flask_cors import CORS
import textwrap
from jinja2 import Environment, FileSystemLoader
import shutil
import bleach
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import tool
from langchain_google_genai import GoogleGenerativeAI
from langchain.memory import ConversationBufferWindowMemory
from pydantic.v1 import BaseModel, Field
from typing import Literal, Optional, Dict
from langchain_google_genai.llms import HarmBlockThreshold, HarmCategory


load_dotenv()
app = Flask(__name__)


CORS(app, origins=["http://127.0.0.1:5000", "http://localhost:5000"], supports_credentials=True)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "a-strong-default-secret-key-for-dev")


SESSION_STORE: dict[str, dict] = {}

def get_session_id() -> str:
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
    return session["session_id"]


def get_or_create_user_state() -> dict:
    sid = get_session_id()
    if sid not in SESSION_STORE:
        print(f"--- CREATING NEW SESSION: {sid} ---")
        SESSION_STORE[sid] = {
            "website_config": {
                "checklist": {"details": False, "style": False, "menu": False, "contact": False, "design": False},
                "data": {"details": {}, "style": "modern", "menu": [], "contact": {}}, 
                "design_prefs": {
                    "section_order": ["introduction", "menu", "contact"],
                    "menu_layout": "grid",
                    "brand_color": None
                }
            },
            "memory": ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True),
        }
    return SESSION_STORE[sid]


def is_safe_filename(name: str) -> bool:
    p = Path(name)
    if not name or ".." in p.parts or p.is_absolute() or not name.endswith('.zip'):
        return False
    return True




@tool
def get_website_build_status(dummy_input: str | None = None) -> str:
    """Checks the current status of the website build to decide what to ask next."""
    user_state = get_or_create_user_state()
    config = user_state["website_config"]
    if all(config["checklist"].values()):
        return json.dumps({"status": "READY_TO_GENERATE", "summary": config["data"]})
    return json.dumps({
        "status": "IN_PROGRESS",
        "collected_items": [k for k, v in config["checklist"].items() if v],
        "missing_items": [k for k, v in config["checklist"].items() if not v],
        "current_data": config["data"] 
    })



@tool
def save_website_style(tool_input: str) -> str:
    """Saves the visual style for the website. The input must be one of ['modern', 'rustic', 'elegant']."""
    
    style_str = tool_input.get("style", str(tool_input)) if isinstance(tool_input, dict) else str(tool_input)
    cleaned_style = style_str.strip().replace('"', '').lower()
    
    actual_style = None
    if 'modern' in cleaned_style:
        actual_style = 'modern'
    elif 'rustic' in cleaned_style:
        actual_style = 'rustic'
    elif 'elegant' in cleaned_style:
        actual_style = 'elegant'

    if not actual_style:
        return f"Error: Invalid style provided. I couldn't find 'modern', 'rustic', or 'elegant' in your choice."

    user_state = get_or_create_user_state()
    user_state["website_config"]["data"]["style"] = actual_style
    user_state["website_config"]["checklist"]["style"] = True
    return f"Style successfully saved as '{actual_style}'."

@tool
def add_menu_item(tool_input: str | dict) -> str:
    """
    Adds a single item (dish) to the restaurant's menu.
    The input MUST be a valid JSON string or dictionary with "name", "price", and optional "description" and "image_url".
    Example: '{"name": "Vada Pav", "price": "40 rupees"}'
    """
    data = None
    if isinstance(tool_input, dict):
        data = tool_input
    else:
        try:
            data = json.loads(tool_input)
        except json.JSONDecodeError:
            return json.dumps({
                "status": "error", "type": "JSONDecodeError",
                "message": "Invalid format for menu item. Please ensure it is a valid JSON."
            })

    try:
        name = data["name"]
        price = data["price"]
        description = data.get("description", "")
        image_url = data.get("image_url")
    except KeyError as e:
        return json.dumps({
            "status": "error", "type": "MissingKeyError",
            "message": f"Missing required menu item field: {e}."
        })
    
    user_state = get_or_create_user_state()
    
    if not isinstance(user_state["website_config"]["data"]["menu"], list):
        user_state["website_config"]["data"]["menu"] = []

    menu_item = { 
        "name": name, 
        "price": price, 
        "description": description, 
        "image_url": image_url  
    }
    user_state["website_config"]["data"]["menu"].append(menu_item)
    
    if not user_state["website_config"]["checklist"]["menu"]:
        user_state["website_config"]["checklist"]["menu"] = True
        
    return f"Added '{name}' to the menu. You can add another item, or say 'done with menu' when you are finished."


@tool
def finish_menu_section(dummy_input: str = "finish") -> str:
    """Call this tool ONLY when the user says they are done adding menu items."""
    user_state = get_or_create_user_state()
    menu = user_state["website_config"]["data"]["menu"]
    if isinstance(menu, list) and len(menu) > 0:
        user_state["website_config"]["checklist"]["menu"] = True
        return "Menu section is complete. I will now ask about the next section."
    else:
        return "Error: No items were added to the menu. Please add at least one item before finishing."
    
@tool
def save_restaurant_details(tool_input: str | dict) -> str:
    """Saves the restaurant's name, cuisine, tagline, and an optional introduction image."""
    data = None
    if isinstance(tool_input, dict):
        data = tool_input
    else:
        try:
            data = json.loads(tool_input)
        except json.JSONDecodeError:
            return json.dumps({"status": "error", "type": "JSONDecodeError", "message": "Invalid format for restaurant details."})

    try:
        name = data["name"]
        cuisine = data["cuisine"]
        tagline = data.get("tagline", "")
        introduction_image_url = data.get("introduction_image_url") or data.get("image_url") or data.get("intro_image_url")
    except KeyError as e:
        return json.dumps({"status": "error", "type": "MissingKeyError", "message": f"Missing required detail: {e}."})

    user_state = get_or_create_user_state()
    user_state["website_config"]["data"]["details"].update({
        "name": name, "cuisine": cuisine, "tagline": tagline, "introduction_image_url": introduction_image_url
    })
    user_state["website_config"]["checklist"]["details"] = True
    return "Successfully saved restaurant details."


@tool
def save_contact_info(tool_input: str | dict) -> str:
    """Saves contact information: address, phone, and hours."""
    data = None
    if isinstance(tool_input, dict):
        data = tool_input
    else:
        try:
            data = json.loads(tool_input)
        except json.JSONDecodeError:
            return json.dumps({"status": "error", "type": "JSONDecodeError", "message": "Invalid format for contact info."})

    try:
        address = data["address"]
        phone = data["phone"]
        hours = data["hours"]
    except KeyError as e:
        return json.dumps({"status": "error", "type": "MissingKeyError", "message": f"Missing required contact info: {e}."})

    user_state = get_or_create_user_state()
    user_state["website_config"]["data"]["contact"].update({
        "address": address, "phone": phone, "hours": hours
    })
    user_state["website_config"]["checklist"]["contact"] = True
    return "Contact information successfully saved."


@tool
def update_existing_information(tool_input: str | dict) -> str:
    """Updates existing information for sections like 'details', 'menu', or 'style'."""
    data = None
    if isinstance(tool_input, dict):
        data = tool_input
    else:
        try:
            data = json.loads(tool_input)
        except json.JSONDecodeError:
            return json.dumps({"status": "error", "type": "JSONDecodeError", "message": "Invalid format for updating information."})

    try:
        item_to_update = data["item_to_update"]
        new_data = data["new_data"]
    except KeyError as e:
        return json.dumps({"status": "error", "type": "MissingKeyError", "message": f"Missing 'item_to_update' or 'new_data'."})

    user_state = get_or_create_user_state()
    config = user_state["website_config"]

    if item_to_update not in config['data']:
        return f"Error: '{item_to_update}' is not a valid section."

    if item_to_update == 'menu':
        item_name_to_find = data.get("item_name")
        if not item_name_to_find:
            return "Error: To update the menu, you must provide the 'item_name' of the dish you want to change."
        menu_list = config['data']['menu']
        item_found = False
        for item in menu_list:
            if item_name_to_find.lower() in item.get('name', '').lower():
                item.update(new_data)
                item_found = True
                break
        if not item_found:
            return f"Error: Could not find a menu item named '{item_name_to_find}' to update."
    elif isinstance(config['data'][item_to_update], dict):
        config['data'][item_to_update].update(new_data)
    elif item_to_update == 'style':
        pass
    
    return f"Successfully updated the '{item_to_update}' section."

@tool
def finish_and_generate_website(tool_input: str) -> str:
    """
    Call this as the VERY LAST step to build the website and create a zip file.
    The input should be the desired filename, preferably without the .zip extension.
    Example: 'latur_ruchira_website'
    """
    try:
        
        if isinstance(tool_input, dict):
            filename_base = tool_input.get("filename", tool_input.get("tool_input", "website"))
        elif isinstance(tool_input, str) and tool_input.startswith('{'):
            data = json.loads(tool_input)
            filename_base = data.get("filename", data.get("tool_input", "website"))
        else:
            filename_base = tool_input
    except (json.JSONDecodeError, TypeError):
        filename_base = "website"

    
    cleaned_base = filename_base.strip().replace('"', '').replace('.zip', '')
    final_filename = f"{cleaned_base}.zip"
    if not is_safe_filename(final_filename):
        return f"Error: The filename '{filename_base}' contains invalid characters. Please use only letters, numbers, and underscores."

    user_state = get_or_create_user_state()

    try:
        
        html_content, css_content = _generate_website_files(user_state)
        
        downloads_dir = "downloads"
        os.makedirs(downloads_dir, exist_ok=True)
        zip_path = os.path.join(downloads_dir, final_filename)

        env = Environment(loader=FileSystemLoader('templates/'))
        preview_template = env.get_template('preview_shell.html')

        single_file_html = preview_template.render(
            title=user_state["website_config"]["data"]["details"].get('name', 'My Restaurant'),
            generated_body_html=html_content.split('<body>')[1].split('</body>')[0].strip(), 
            embedded_css=css_content
        )

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("index.html", html_content)
            zf.writestr("style.css", css_content)
            zf.writestr("_preview-in-one-file.html", single_file_html)

        download_url = f"/downloads/{final_filename}"
        
        return (f"Website generation complete! "
                f"<a href='{download_url}' target='_blank' download>Click here to download your website</a>. "
                f"Inside the zip, open `_preview-in-one-file.html` to see the styled page immediately.")

    except Exception as e:
        traceback.print_exc()
        return f"A critical error occurred during website generation: {e}"
    

@tool
def preview_website(dummy_input: str = "preview") -> str:
    """
    Generates a live, temporary preview of the website using AI to write the code.
    Use this after all info, including design preferences, is collected.
    """
    user_state = get_or_create_user_state()
    sid = get_session_id()
    preview_dir = Path("previews") / sid
    
    if preview_dir.exists():
        shutil.rmtree(preview_dir)
    os.makedirs(preview_dir, exist_ok=True)
    
    try:
        html_content, css_content = _generate_website_files(user_state)

        (preview_dir / "index.html").write_text(html_content, encoding='utf-8')
        (preview_dir / "style.css").write_text(css_content, encoding='utf-8')

        preview_url = f"/preview/{sid}/index.html"
        return f"I have generated a preview. <a href='{preview_url}' target='_blank'>Click here to open it.</a> How does this look?"
    except Exception as e:
        traceback.print_exc()
        return f"A critical error occurred during preview generation: {e}"   

@tool
def save_design_preferences(tool_input: str | dict) -> str:
    """Saves design preferences: section_order, menu_layout, brand_color."""
    data = None
    if isinstance(tool_input, dict):
        data = tool_input
    else:
        try:
            data = json.loads(tool_input)
        except json.JSONDecodeError:
            return json.dumps({"status": "error", "type": "JSONDecodeError", "message": "Invalid format for design preferences."})

    user_state = get_or_create_user_state()
    user_state["website_config"]["design_prefs"].update(data)
    user_state["website_config"]["checklist"]["design"] = True
    return "Design preferences have been saved. I am now ready to generate a preview."



def _generate_website_files(user_state):
    """
    A helper function that uses an LLM to generate HTML and CSS from scratch.
    """
    config = user_state["website_config"]
    full_summary = {
        "content": config["data"],
        "design": config["design_prefs"]
    }
    config_json = json.dumps(full_summary, indent=2)

    code_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.getenv("GEMINI_API_KEY"), temperature=0.1, safety_settings={
                                                                                                                                                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                                                                                                                                                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                                                                                                                                                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                                                                                                                                                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                                                                                                                                            },) 

    
    html_writer_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert frontend developer. Your task is to write the HTML for a restaurant website based on the provided JSON. You must follow the `design.section_order`.\n\n"
         "**CRITICAL INSTRUCTIONS:**\n"
         "1.  **DO NOT** include `<html>`, `<head>`, `<body>`, `<style>`, or ````html` tags. Output ONLY the raw HTML for the page content.\n"
         "2.  **Introduction Section:**\n"
         "    - Use the provided Jinja2-like syntax: `{{ content.details.name }}` to insert data.\n"
         "    - **IF `content.details.introduction_image_url` EXISTS:** Create a hero section like this: `<section id='introduction-hero'><div class='hero-overlay'><div class='hero-text'><h1>{{ content.details.name }}</h1><p>{{ content.details.tagline }}</p></div></div></section>`.\n"
         "    - **IF it is MISSING:** Create a simple section: `<section id='introduction'><h1>{{ content.details.name }}</h1><p>{{ content.details.tagline }}</p></section>`.\n"
         "3.  **Menu Section:**\n"
         "    - Create a section with `id='menu'`.\n"
         "    - Use the class `menu-grid` or `menu-list` on a container `div` based on `design.menu_layout`.\n"
         "    - Loop through `content.menu` items, creating a `div` with class `menu-item` for each.\n"
         "    - For each item, include an `<img>` with the `src` set to `{{ item.image_url }}` and `alt` set to `{{ item.name }}`.\n"
         "4.  **Contact Section:**\n"
         "    - Create a section with `id='contact'` and use an `<address>` tag for the information."
         ),
        ("user", "Here is the website configuration data:\n\n{config_json}")
    ])

   
    css_writer_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert CSS designer. Your task is to write a complete, professional CSS stylesheet based on the provided JSON data. Do not wrap your code in ```css markers.\n\n"
         "**CRITICAL STYLING INSTRUCTIONS:**\n"
         "1.  **Brand Color:** You MUST use the value from `design.brand_color` for the main accent color, especially for headings (`h1`, `h2`, `h3`). Use this placeholder in your CSS: `BRAND_COLOR_PLACEHOLDER`.\n"
         "2.  **Hero Section (`#introduction-hero`):**\n"
         "    - This section is a banner. Its `background-image` MUST be set using the URL from `content.details.introduction_image_url`.\n"
         "    - It must have a semi-transparent dark overlay (`.hero-overlay`).\n"
         "    - Text (`.hero-text`) must be white and centered with a text-shadow for readability.\n"
         "3.  **Menu Layout:**\n"
         "    - Style `.menu-grid` with `display: grid;`\n"
         "    - Style `.menu-list` with `display: flex; flex-direction: column;`"
        ),
        ("user", "{config_json}")
    ])

    html_chain = html_writer_prompt | code_llm | StrOutputParser()
    css_chain = css_writer_prompt | code_llm | StrOutputParser()

    print("--- Generating HTML body code from scratch... ---")
    generated_body_html_template = html_chain.invoke({"config_json": config_json}).strip()
    
    if generated_body_html_template.startswith("```html"):
        print("--- Cleaning HTML markdown fences ---")
        generated_body_html_template = "\n".join(generated_body_html_template.split('\n')[1:-1])
    elif generated_body_html_template.startswith("```"):
        print("--- Cleaning generic HTML markdown fences ---")
        generated_body_html_template = "\n".join(generated_body_html_template.split('\n')[1:-1])


    print("--- Generating CSS code from scratch... ---")
    generated_css_template = css_chain.invoke({"config_json": config_json}).strip()

    if generated_css_template.startswith("```css"):
        print("--- Cleaning CSS markdown fences ---")
        generated_css_template = "\n".join(generated_css_template.split('\n')[1:-1])
    elif generated_css_template.startswith("```"):
        print("--- Cleaning generic CSS markdown fences ---")
        generated_css_template = "\n".join(generated_css_template.split('\n')[1:-1])

    env = Environment(loader=FileSystemLoader('templates/'))
    body_template = env.from_string(generated_body_html_template)
    rendered_body = body_template.render(content=config["data"], design=config["design_prefs"])

    
    allowed_tags = list(bleach.ALLOWED_TAGS) + ['div', 'p', 'h1', 'h2', 'h3', 'header', 'main', 'section', 'footer', 'a', 'strong', 'em', 'ul', 'li', 'ol', 'img', 'address', 'br']
    allowed_attrs = {**bleach.ALLOWED_ATTRIBUTES, '*': ['class', 'id', 'style'], 'a': ['href', 'title'], 'img': ['src', 'alt']}
    sanitized_html_body = bleach.clean(rendered_body, tags=allowed_tags, attributes=allowed_attrs)

   
    brand_color = config["design_prefs"].get("brand_color", "#333") 
    final_css_code = generated_css_template.replace("BRAND_COLOR_PLACEHOLDER", brand_color)
    

    shell_template = env.get_template('shell.html')
    final_html_content = shell_template.render(
        title=config['data']['details'].get('name', 'My Restaurant'),
        generated_body_html=sanitized_html_body
    )
    
    return final_html_content, final_css_code

llm = GoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.getenv("GEMINI_API_KEY"), temperature=0.0, safety_settings={
                                                                                                                                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                                                                                                                                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                                                                                                                                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                                                                                                                                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                                                                                                                            },)

tools = [
    get_website_build_status, 
    preview_website,
    save_restaurant_details, 
    save_contact_info,
    update_existing_information,
    save_website_style,
    add_menu_item,
    finish_menu_section,
    finish_and_generate_website,
    save_design_preferences
]



react_prompt = PromptTemplate.from_template("""
You are "AI Web Designer", a methodical assistant that builds restaurant websites by asking questions in a specific order.

**WORKFLOW:** You MUST follow these phases strictly. Use the `get_website_build_status` tool to determine your current state and what to ask for next.

**Phase 1: Collect Content**
- GOAL: Collect all core content. The required items in this phase are `details`, `contact`, and `menu`.
- ACTION: Check the `missing_items` from `get_website_build_status`.
  - If `details` is missing, ask for it.
  - If `details` is complete but `contact` is missing, ask for contact info.
  - If `details` and `contact` are complete but `menu` is missing, ask about the menu.
- **IMPORTANT**: Do NOT proceed to Phase 2 until `details`, `contact`, AND `menu` are all marked as collected. Do NOT ask for style or design preferences yet.

**Phase 2: Collect Style**
- CONDITION: This phase begins ONLY when `details`, `contact`, and `menu` are complete, and `style` is the next missing item.
- ACTION: Ask for the website's visual `style`. The only valid options are 'modern', 'rustic', or 'elegant'.

**Phase 3: Design & Preview**
- CONDITION: This phase begins ONLY when all content and the style are collected, and `design` is the only remaining missing item.
- ACTION:
    1. Ask for ALL design preferences in a single question: `section_order`, `menu_layout`, and `brand_color`.
    2. After the user responds, use `save_design_preferences` to save their choices.
    3. Immediately after saving, use the `preview_website` tool to show the result. Do not ask for permission to show the preview.

**Phase 4: Finalize**
- CONDITION: You are in this phase after a preview has been successfully generated and shown to the user.
- ACTION: Ask for feedback on the preview.
    - If the user approves, ask for a filename and then use `finish_and_generate_website`.
    - If the user requests changes, use `update_existing_information` to apply them, and then use `preview_website` again to show the updated version.

**--- CRITICAL RULE FOR LINKS ---**
If a tool's `Observation` contains an HTML `<a>` tag (like a preview or download link), your next `Thought` and `Final Answer` MUST follow this format EXACTLY:
Thought: The user needs to see the link generated by the tool. I will copy the entire Observation text into my Final Answer without any changes or commentary.
Final Answer: [THE ENTIRE, UNMODIFIED TEXT FROM THE OBSERVATION, INCLUDING THE FULL `<a>` TAG]

**TOOL USAGE:**
- Your `Action Input` for tools that take JSON must be a valid raw JSON object, not a stringified JSON.
- If a tool returns a critical error (e.g., a message containing "A critical error occurred" or a Python traceback), DO NOT try other tools to fix it. Apologize to the user, state that a technical problem occurred, and stop.

**You have access to the following tools:**
{tools}

**RESPONSE FORMAT:**
**To use a tool:**
Thought: [Your reasoning, explicitly stating which Phase you are in and why you are choosing this tool based on the workflow.]
Action: The tool to use, one of [{tool_names}].
Action Input: The arguments for the tool, as a valid JSON object.

**To ask a question:**
Thought: [Your reasoning, stating which Phase you are in and what information you are missing according to the workflow.]
Final Answer: [Your question to the user.]

---
**Begin!**

Conversation History:
{chat_history}

User's Latest Message:
Question: {input}

Your Response:
{agent_scratchpad}
""")


agent = create_react_agent(llm, tools, react_prompt)

def _handle_pydantic_error(error: Exception) -> str:
    return f"I'm sorry, I couldn't understand some of the details you provided. Please check that you've provided all the required information for a section. For example, for contact info, I need the address, phone, and hours all at once. Error details: {error}"


@app.route("/")
def index():
    get_session_id()
    return render_template("bot.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"reply": "Invalid request. 'message' field is required."}), 400
    user_message = data.get("message", "")

    if user_message.lower().strip() == "restart":
        sid = get_session_id()
        if sid in SESSION_STORE:
            SESSION_STORE.pop(sid)
        get_or_create_user_state()
        return jsonify({"reply": "Ok, I've cleared everything. Let's start over! What is your restaurant's name and cuisine type?"})

    user_state = get_or_create_user_state()

    agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=user_state["memory"],
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10,
)

    try:
        response = agent_executor.invoke({"input": user_message})
        ai_reply = response.get("output", "I'm sorry, I encountered an error. Please try again.")
        return jsonify({"reply": ai_reply})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"reply": f"A critical error occurred: {e}"}), 500

@app.route('/downloads/<path:filename>')
def download_file(filename):
    if not is_safe_filename(filename):
        return "Invalid filename.", 400
    return send_from_directory("downloads", filename, as_attachment=True)

def is_safe_path_component(component):
    """Prevents directory traversal attacks."""
    return not (".." in component or component.startswith("/") or component.startswith("\\"))

@app.route('/preview/<session_id>/<path:filename>')
def preview_file(session_id, filename):
    if not is_safe_path_component(session_id) or not is_safe_path_component(filename):
        return "Invalid path.", 400
    
    preview_dir = Path("previews") / session_id
    if not preview_dir.is_dir():
        return "Preview not found or has expired.", 404
        
    return send_from_directory(preview_dir, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)