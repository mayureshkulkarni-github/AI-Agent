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
                },
                "is_complete": False,
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
    
    cleaned_style = tool_input.strip().replace('"', '').lower()
    
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
def add_menu_item(tool_input: str) -> str:
    """
    Adds a single item (dish) to the restaurant's menu.
    The input MUST be a valid JSON string with "name", "price", and optional "description" and "image_url".
    Example: '{"name": "Vada Pav", "price": "40 rupees", "image_url": "http://example.com/vada.jpg"}'
    """
    try:
        data = json.loads(tool_input)
        name = data["name"]
        price = data["price"]
        description = data.get("description", "")
        image_url = data.get("image_url")
    except json.JSONDecodeError:
        error_info = {
            "status": "error",
            "type": "JSONDecodeError",
            "message": "The input was not valid JSON. I need a well-formatted JSON string from you."
        }
        return json.dumps(error_info)
    except KeyError as e:
        error_info = {
            "status": "error",
            "type": "MissingKeyError",
            "message": f"I'm missing a required piece of information: {e}. I need a 'name' and a 'price'."
        }
        return json.dumps(error_info)
    
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
def save_restaurant_details(tool_input: str) -> str:
    """
Saves the restaurant's name, cuisine, tagline, and an optional introduction image.
The input MUST be a valid JSON string with "name", "cuisine", and optional "tagline" and "introduction_image_url".
Example: '{"name": "The Golden Spoon", "cuisine": "Italian", "introduction_image_url": "http://example.com/introduction.jpg"}'
"""
    try:
        data = json.loads(tool_input)
        name = data["name"]
        cuisine = data["cuisine"]
        tagline = data.get("tagline", "")
        introduction_image_url = data.get("introduction_image_url")
    except json.JSONDecodeError:
        error_info = {
            "status": "error",
            "type": "JSONDecodeError",
            "message": "The input was not valid JSON. I need a well-formatted JSON string from you."
        }
        return json.dumps(error_info)
    except KeyError as e:
        error_info = {
            "status": "error",
            "type": "MissingKeyError",
            "message": f"I'm missing a required piece of information: {e}. I need both a 'name' and a 'cuisine'."
        }
        return json.dumps(error_info)

    user_state = get_or_create_user_state()
    user_state["website_config"]["data"]["details"].update({
        "name": name,
        "cuisine": cuisine,
        "tagline": tagline,
        "introduction_image_url": introduction_image_url
    })
    user_state["website_config"]["checklist"]["details"] = True
    return "Successfully saved restaurant details."

@tool
def save_contact_info(tool_input: str) -> str:
    """
    Saves contact information.
    The input to this tool MUST be a valid JSON string with the keys "address", "phone", and "hours".
    Example: '{"address": "123 Main St", "phone": "555-1234", "hours": "9-5 M-F"}'
    """
    try:
        data = json.loads(tool_input)
        address = data["address"]
        phone = data["phone"]
        hours = data["hours"]
    except json.JSONDecodeError:
        error_info = {
            "status": "error",
            "type": "JSONDecodeError",
            "message": "The input was not valid JSON. I need a well-formatted JSON string from you."
        }
        return json.dumps(error_info)
    except KeyError as e:
        error_info = {
            "status": "error",
            "type": "MissingKeyError",
            "message": f"I'm missing a required piece of information: {e}. I need an 'address', 'phone', and 'hours'."
        }
        return json.dumps(error_info)

    user_state = get_or_create_user_state()
    user_state["website_config"]["data"]["contact"].update({
        "address": address,
        "phone": phone,
        "hours": hours
    })
    user_state["website_config"]["checklist"]["contact"] = True
    return "Contact information successfully saved."



@tool
def update_existing_information(tool_input: str) -> str:
    """
    Use this to make specific, granular changes to information that has already been provided.
    For single fields like 'style' or 'tagline', provide the section and the new data.
    Example for style: '{"item_to_update": "style", "new_data": {"style": "rustic"}}'
    Example for tagline: '{"item_to_update": "details", "new_data": {"tagline": "The best food in town."}}'
    
    For updating a specific menu item, provide the item's name to identify it.
    Example for menu price: '{"item_to_update": "menu", "item_name": "Vada Pav", "new_data": {"price": "40 rupees"}}'
    """
    try:
        data = json.loads(tool_input)
        item_to_update = data["item_to_update"]
        new_data = data["new_data"]
    except json.JSONDecodeError:
        error_info = {
            "status": "error",
            "type": "JSONDecodeError",
            "message": "The input was not valid JSON. I need a well-formatted JSON string from you."
        }
        return json.dumps(error_info)
    except KeyError as e:
        error_info = {
            "status": "error",
            "type": "MissingKeyError",
            "message": f"I'm missing a required piece of information: {e}. I need 'item_to_update' and 'new_data'."
        }
        return json.dumps(error_info)

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
        
        if item_found:
            return f"Successfully updated '{item_name_to_find}' in the menu."
        else:
            return f"Error: Could not find a menu item named '{item_name_to_find}' to update."
            
    elif isinstance(config['data'][item_to_update], dict):
        config['data'][item_to_update].update(new_data)
    elif item_to_update == 'style':
        new_style = new_data.get('style')
        if not new_style: return "Error: 'new_data' for style must contain a 'style' key."
        if new_style in ['modern', 'rustic', 'elegant']:
            config['data']['style'] = new_style
        else:
            return f"Error: Invalid style '{new_style}'."
    
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
    if user_state["website_config"]["is_complete"]:
        return "Website has already been generated."

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

        user_state["website_config"]["is_complete"] = True
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
def save_design_preferences(tool_input: str) -> str:
    """
    Saves the user's design preferences for the website layout.
    The input MUST be a JSON string.
    'section_order' must be a list of strings from ['introduction', 'menu', 'contact'].
    'menu_layout' must be either 'grid' or 'list'.
    'brand_color' can be a hex code like '#ff0000' or a color name.
    Example: '{"section_order": ["contact", "menu", "introduction"], "menu_layout": "list", "brand_color": "#A52A2A"}'
    """
    try:
        data = json.loads(tool_input)
    except json.JSONDecodeError:
        error_info = {
            "status": "error",
            "type": "JSONDecodeError",
            "message": "The input was not valid JSON. I need a well-formatted JSON string from you."
        }
        return json.dumps(error_info)
    except KeyError as e:
        error_info = {
            "status": "error",
            "type": "MissingKeyError",
            "message": f"I'm missing a required piece of information: {e}. The JSON must contain the keys for the preferences you wish to save."
        }
        return json.dumps(error_info)

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

    code_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.getenv("GEMINI_API_KEY"), temperature=0.2)

    html_writer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert frontend developer. Your task is to write the HTML content for a restaurant website's <body> based on the provided JSON data. You must follow the user's `design.section_order`. Do NOT include any CSS, <style>, <html>, <head>, or <body> tags.\n\n"
     "**CRITICAL INSTRUCTIONS FOR SECTIONS:**\n"
     "1.  **Introduction Section:**\n"
     "    - **IF `content.details.introduction_image_url` EXISTS:** You MUST create a special 'hero' section. The structure MUST be exactly: `<section id='introduction-hero' style=\"background-image: url('{{ content.details.introduction_image_url }}');\"><div class='hero-overlay'><div class='hero-text'><h1>{{ content.details.name }}</h1><p>{{ content.details.tagline }}</p></div></div></section>`.\n"
     "    - **IF `content.details.introduction_image_url` is MISSING or NULL:** You MUST create a simple text section: `<section id='introduction'><h1>{{ content.details.name }}</h1><p>{{ content.details.tagline }}</p></section>`.\n"
     "2.  **Menu & Contact Sections:** Generate the HTML for the `menu` and `contact` sections exactly as you have been instructed before, using the `menu-grid`/`menu-item` classes and the `<address>` tag."
    ),
    ("user", "Here is the website configuration data:\n\n{config_json}")
])
    
    css_writer_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are an expert CSS designer. Your task is to write a complete, professional, and visually appealing CSS stylesheet based on the provided JSON data.\n\n"
     "**CRITICAL STYLING INSTRUCTIONS:**\n"
     "1.  **Hero Section Styling (for `#introduction-hero`):**\n"
     "    - This section MUST be a banner: `min-height: 400px; background-size: cover; background-position: center; display: flex; align-items: center; justify-content: center; position: relative;`.\n"
     "    - The `.hero-overlay` class MUST cover the entire section and have a semi-transparent dark background: `position: absolute; top: 0; left: 0; width: 100%; height: 100%; background-color: rgba(0, 0, 0, 0.5);`.\n"
     "    - The `.hero-text` class MUST be centered (`text-align: center;`), positioned above the overlay (`position: relative; z-index: 2;`), and have **white text** (`color: white;`).\n"
     "    - The `h1` inside `.hero-text` MUST have a large, impactful font size (e.g., `font-size: 3.5rem;`) and a text-shadow for readability (`text-shadow: 2px 2px 4px black;`).\n"
     "2.  **Standard Section Styling (for `#introduction`, `#menu`, `#contact`):**\n"
     "    - Use the `design.brand_color` as the primary accent color for headings (`h1`, `h2`, `h3`).\n"
     "    - Style the simple `#introduction` section and the `#contact` section with `text-align: center;`.\n"
     "    - Style menu items and other elements as you have been instructed before.\n\n"
     "Do not wrap your code in ```css markers."),
    ("user", "{config_json}")
])
    
  
    html_chain = html_writer_prompt | code_llm | StrOutputParser()
    css_chain = css_writer_prompt | code_llm | StrOutputParser()

    print("--- Generating HTML body code from scratch... ---")
    generated_body_html = html_chain.invoke({"config_json": config_json})
    if generated_body_html.strip().startswith("```html"):
        print("--- Cleaning HTML markdown fences ---")
        generated_body_html = generated_body_html.split('\n', 1)[1]
        generated_body_html = generated_body_html.rsplit('```', 1)[0]
    
    print("--- Generating CSS code from scratch... ---")
    css_code = css_chain.invoke({"config_json": config_json})
    if css_code.strip().startswith("```css"):
        print("--- Cleaning CSS markdown fences ---")
        css_code = css_code.split('\n', 1)[1]
        css_code = css_code.rsplit('```', 1)[0]

    

    env = Environment(loader=FileSystemLoader('templates/'))
    body_template = env.from_string(generated_body_html)
    rendered_body = body_template.render(content=config["data"], design=config["design_prefs"])

    
    allowed_tags = ['div', 'p', 'h1', 'h2', 'h3', 'header', 'main', 'section', 'footer', 'a', 'strong', 'em', 'ul', 'li', 'ol', 'img', 'address', 'br']

    allowed_attrs = {'*': ['class', 'id', 'style'], 'a': ['href', 'title'], 'img': ['src', 'alt']}

    sanitized_html_body = bleach.clean(rendered_body, tags=allowed_tags, attributes=allowed_attrs)
    shell_template = env.get_template('shell.html')
    final_html_content = shell_template.render(
        title=config['data']['details'].get('name', 'My Restaurant'),
        generated_body_html=sanitized_html_body
    )
    
    return final_html_content, css_code



llm = GoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.getenv("GEMINI_API_KEY"), temperature=0.0)

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
You are "Zesty AI Web Designer", a creative assistant that builds unique, professional websites.

**CRITICAL FLOW & RULES:**
1.  **GATHER CONTENT:** Your first goal is to collect the base content. Use `get_website_build_status` to find `missing_items`. Collect `details`, `style`, `menu`, and `contact` information first. Do not ask about design yet.
    **1a. BE THOROUGH:** When you ask the user for information for a section (like 'details' or a 'menu item'), you MUST ask for all possible fields at once, clearly mentioning which are optional. Do not just ask for the minimum required fields.
2.  **GATHER STYLE:** After `details`, `menu`, and `contact` are collected, your next step is to ask for the website `style` ('modern', 'rustic', or 'elegant').
3.  **DESIGN INTERVIEW:** Once `style` is also collected and `design` is the only missing item, you MUST conduct the design interview.
4.  **SAVE DESIGN:** After the interview, your ONLY action is to use the `save_design_preferences` tool to save their choices.
5.  **PREVIEW:** When `get_website_build_status` shows `status` is `READY_TO_GENERATE`, your **next and only action** is to use `preview_website`.
6.  **GET FEEDBACK & UPDATE:** After generating a preview, ask for feedback. If they want changes to content or design, use `update_existing_information` or `save_design_preferences`, then go back to Step 4.
7.  **FINALIZE:** If the user approves and provides a filename, use `finish_and_generate_website`. If they approve but don't give a filename, ask for one.
8.  **SHOW HTML LINK:** If you have just used `preview_website` or `finish_and_generate_website`, you MUST format your response as follows, with no extra commentary:
    Thought: I have just used a tool that generates a link. I must pass the exact text from the Observation to the user in my Final Answer.
    Final Answer: [The exact, unmodified text from the Observation goes here]

**TOOL USAGE POLICY - VERY IMPORTANT!**
1.  **FOCUS ON THE LATEST MESSAGE:** When deciding to use a `save_` or `add_` tool, you MUST base your decision **ONLY on the information in the user's most recent message (`{input}`)**. Do not try to re-save information from the `Conversation History`.
2.  **ONE ACTION AT A TIME:** You MUST only perform a single `Action` per turn. Do not generate an `Action` and a `Final Answer` in the same turn.
3.  **MENU COLLECTION POLICY:** This is a strict multi-turn process.
    - **Turn A (Ask):** When asking for a menu item, you MUST explain *why* direct image URLs are important. Say something like: "Please provide a direct link to an image (ending in .jpg, .png, etc.). Links from pages like Google Drive or a standard Unsplash page will not display correctly."
    - **Turn B (Save):** You MUST accept and save whatever URL the user provides, even if it doesn't look like a direct link. Your ONLY action is to use the `add_menu_item` tool with the provided data.
    - **Turn C (Confirm):** After the tool succeeds, your `Final Answer` should be to ask for the next item or if they are done.
                                            
**ERROR HANDLING POLICY:**
If a tool call results in an `Observation` that is a JSON object with `status: "error"`, your job is to interpret the `message` field from that JSON. DO NOT simply repeat the error message to the user. Instead, formulate a clear, helpful question that asks for the specific missing or incorrect information.
**Example:**
Observation: {{"status": "error", "type": "MissingKeyError", "message": "I'm missing a required piece of information: 'cuisine'. I need both a 'name' and a 'cuisine'."}}
Thought: The tool failed because the user only gave a name but not a cuisine. I need to ask for the cuisine.
Final Answer: Thanks for the name! What type of cuisine does your restaurant serve?

**CAPABILITY POLICY:**
Your capabilities are strictly limited to the tools provided. You can gather text, save preferences, and generate a website with text and images (if URLs are provided).
If the user asks for a feature you do not have a tool for, you MUST NOT attempt to invent a solution.
Examples of unsupported requests:
- Video embedding
- Contact forms
- Online ordering systems
- Image uploads (you can only accept image URLs)
- Animations
If you receive an unsupported request, your ONLY response should be to state that you cannot perform that action and politely list what you *can* do.
**Example of a correct refusal:**
Final Answer: I'm sorry, I can't add a contact form to the website. My capabilities are focused on creating a static page with your restaurant's details, menu, contact info, and images. I can, however, change the style, content, or layout if you'd like.
                                            
**You have access to the following tools:**
{tools}

**RESPONSE FORMAT:**
**To use a tool:**
Thought: [Your reasoning, explicitly stating that you have all required arguments for the chosen tool based on the user's input.]
Action: The tool to use, one of [{tool_names}].
Action Input: A JSON object containing the arguments for the tool. For tools with a Pydantic schema (like `save_restaurant_details`), the JSON object should match the schema's fields.

**EXAMPLE of CORRECT `Action Input` for a tool with a Pydantic schema:**
Action: save_restaurant_details
Action Input: {{"name": "The Golden Spoon", "cuisine": "Italian", "tagline": "A taste of Rome."}}

**To ask a question:**
Thought: [Your reasoning, explicitly stating which arguments you are missing for a specific tool, which is why you must ask a question.]
Final Answer: [The question you will ask the user for the missing information.]

---
**BEGIN!**

**Conversation History:**
{chat_history}

**User's Latest Message:**
Question: {input}

**Your Response:**
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
    app.run(debug=True, use_reloader=False, port=5000)